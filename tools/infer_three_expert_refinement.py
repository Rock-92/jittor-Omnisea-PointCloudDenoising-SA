import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import jittor as jt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.noise_classifier import PatchNoiseClassifier  # noqa: E402
from src.model.refinement import MultiStageGeometryRefiner  # noqa: E402
from tools.evaluate_full_cloud_refinement import (  # noqa: E402
    build_patches,
    fuse_patches,
)
from tools.evaluate_three_expert_refinement import softmax  # noqa: E402
from tools.hard_patch_common import load_model, read_datalist  # noqa: E402


def package_result(result_dir, zip_path):
    files = sorted(Path(result_dir).rglob("denoised.npy"))
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(result_dir).as_posix())
    return len(files)


def load_refiner(path, args, adaptive_v2):
    model = MultiStageGeometryRefiner(
        num_stages=args.stages,
        stage_max_residuals=(
            args.stage1_max_residual,
            args.stage2_max_residual,
        ),
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        adaptive_v2=adaptive_v2,
        min_residual_ratio=args.min_residual_ratio,
    )
    model.load(path)
    model.eval()
    return model


def predict_patch_batches(vm, classifier, refiners, patches, args):
    outputs = []
    probabilities_all = []
    point_indices = np.linspace(
        0,
        patches.shape[1] - 1,
        min(args.classifier_points, patches.shape[1]),
        dtype=np.int32,
    )
    vm.eval()
    classifier.eval()
    with jt.no_grad():
        for start in range(0, patches.shape[0], args.batch_size):
            end = min(start + args.batch_size, patches.shape[0])
            noisy = patches[start:end]
            if args.coarse_mode == "fixed":
                sigma = jt.ones((end - start, 1)) * float(args.sigma)
                coarse = vm.predict_clean(noisy, sigma=sigma)
            else:
                coarse, _ = vm.denoise_langevin_dynamics(noisy)
            logits, _ = classifier(
                noisy[:, point_indices, :],
                coarse[:, point_indices, :],
            )
            probabilities = softmax(logits.numpy(), args.temperature)
            expert_predictions = [
                refiner(coarse, noisy)[0].numpy()
                for refiner in refiners
            ]
            blended = sum(
                expert_predictions[index]
                * probabilities[:, index, None, None]
                for index in range(3)
            )
            outputs.append(blended.astype(np.float32, copy=False))
            probabilities_all.append(probabilities)
    return np.concatenate(outputs), np.concatenate(probabilities_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/vm_ssl/checkpoint_best.pkl",
    )
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--low-refiner", required=True)
    parser.add_argument("--medium-refiner", required=True)
    parser.add_argument("--high-refiner", required=True)
    parser.add_argument("--low-refiner-v1", action="store_true")
    parser.add_argument("--medium-refiner-v1", action="store_true")
    parser.add_argument("--high-refiner-v1", action="store_true")
    parser.add_argument("--noisy-root", default="test_noisy")
    parser.add_argument("--datalist", default="datalist/test.txt")
    parser.add_argument(
        "--out-dir",
        default="outputs/three_expert/result/test_noisy",
    )
    parser.add_argument(
        "--result-zip",
        default="outputs/three_expert/result/result.zip",
    )
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--sigma", type=float, default=0.020)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--classifier-points", type=int, default=256)
    parser.add_argument("--classifier-k", type=int, default=24)
    parser.add_argument("--classifier-local-dim", type=int, default=96)
    parser.add_argument("--classifier-hidden-dim", type=int, default=192)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-max-residual", type=float, default=0.012)
    parser.add_argument("--stage2-max-residual", type=float, default=0.008)
    parser.add_argument("--min-residual-ratio", type=float, default=0.2)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for rel_path in read_datalist(args.datalist):
        noisy_path = Path(args.noisy_root) / rel_path / "noisy.npy"
        if noisy_path.exists():
            samples.append((rel_path, noisy_path))
    if args.max_shapes > 0:
        samples = samples[:args.max_shapes]
    if not samples:
        raise FileNotFoundError("no test noisy.npy files found")

    vm = load_model(args.checkpoint)
    for parameter in vm.parameters():
        parameter.stop_grad()
    classifier = PatchNoiseClassifier(
        k=args.classifier_k,
        local_dim=args.classifier_local_dim,
        hidden_dim=args.classifier_hidden_dim,
    )
    classifier.load(args.classifier_checkpoint)
    refiners = [
        load_refiner(
            args.low_refiner,
            args,
            adaptive_v2=not args.low_refiner_v1,
        ),
        load_refiner(
            args.medium_refiner,
            args,
            adaptive_v2=not args.medium_refiner_v1,
        ),
        load_refiner(
            args.high_refiner,
            args,
            adaptive_v2=not args.high_refiner_v1,
        ),
    ]

    records = []
    total_start = time.time()
    for index, (rel_path, noisy_path) in enumerate(samples):
        start = time.time()
        noisy = np.load(noisy_path).astype(np.float32, copy=False)
        patches, seeds, point_indices, patch_distances = build_patches(
            noisy,
            args.patch_size,
            args.seed_k,
        )
        refined_patches, probabilities = predict_patch_batches(
            vm,
            classifier,
            refiners,
            patches,
            args,
        )
        refined = fuse_patches(
            noisy,
            refined_patches,
            seeds.numpy().astype(np.float32, copy=False),
            point_indices,
            patch_distances,
            args.fusion_tau,
        )
        if refined.shape != noisy.shape or not np.isfinite(refined).all():
            raise ValueError(f"{rel_path}: invalid refinement output")
        output_path = out_dir / rel_path / "denoised.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, refined.astype(np.float32, copy=False))
        record = {
            "index": index,
            "rel_path": rel_path,
            "num_points": int(noisy.shape[0]),
            "num_patches": int(patches.shape[0]),
            "mean_low_probability": float(probabilities[:, 0].mean()),
            "mean_medium_probability": float(probabilities[:, 1].mean()),
            "mean_high_probability": float(probabilities[:, 2].mean()),
            "elapsed_seconds": time.time() - start,
        }
        records.append(record)
        print(
            f"[{index + 1}/{len(samples)}] {rel_path} "
            f"patches={record['num_patches']} "
            f"time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )

    packaged_count = package_result(out_dir, args.result_zip)
    summary = {
        "sample_count": len(records),
        "packaged_count": packaged_count,
        "elapsed_seconds": time.time() - total_start,
        "records": records,
        "args": vars(args),
    }
    (out_dir.parent / "inference_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
