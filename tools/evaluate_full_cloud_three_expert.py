import argparse
import csv
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.noise_classifier import PatchNoiseClassifier  # noqa: E402
from tools.build_refinement_probe_dataset import choose_paths  # noqa: E402
from tools.evaluate_full_cloud_refinement import (  # noqa: E402
    build_patches,
    fuse_patches,
    normalize_clean,
    score_prediction,
)
from tools.hard_patch_common import (  # noqa: E402
    load_model,
    quantile_summary,
    read_datalist,
)
from tools.infer_three_expert_refinement import load_refiner  # noqa: E402
from tools.evaluate_three_expert_refinement import routing_weights  # noqa: E402


BAND_NAMES = ["low", "medium", "high"]
BAND_RANGES = [
    (0.005, 0.010),
    (0.010, 0.015),
    (0.015, 0.020),
]


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def excluded_paths(dataset_paths):
    excluded = set()
    for dataset_path in dataset_paths:
        data = np.load(dataset_path, allow_pickle=True)
        excluded.update(str(path) for path in data["rel_path"].tolist())
    return excluded


def predict_patch_batches(vm, classifier, refiners, patches, args):
    coarse_all = []
    expert_all = [[], [], []]
    soft_all = []
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
                sigma = jt.ones((end - start, 1)) * float(args.noise_std)
                coarse = vm.predict_clean(noisy, sigma=sigma)
            else:
                coarse, _ = vm.denoise_langevin_dynamics(noisy)
            logits, classifier_output = classifier(
                noisy[:, point_indices, :],
                coarse[:, point_indices, :],
            )
            predicted_sigma = (
                0.005
                + 0.015
                * classifier_output["sigma_normalized"].numpy().reshape(-1)
            )
            probabilities = routing_weights(
                logits.numpy(),
                predicted_sigma,
                args.temperature,
                mode=args.routing_mode,
                high_threshold=args.high_route_threshold,
                adjacent_boundary=args.adjacent_boundary,
                high_sigma_boundary=args.high_sigma_boundary,
            )
            expert_predictions = [
                refiner(coarse, noisy)[0].numpy()
                for refiner in refiners
            ]
            soft = sum(
                expert_predictions[index]
                * probabilities[:, index, None, None]
                for index in range(3)
            )
            coarse_all.append(coarse.numpy().astype(np.float32, copy=False))
            for expert_index in range(3):
                expert_all[expert_index].append(
                    expert_predictions[expert_index].astype(
                        np.float32,
                        copy=False,
                    )
                )
            soft_all.append(soft.astype(np.float32, copy=False))
            probabilities_all.append(probabilities)
    return (
        np.concatenate(coarse_all),
        [
            np.concatenate(expert_predictions)
            for expert_predictions in expert_all
        ],
        np.concatenate(soft_all),
        np.concatenate(probabilities_all),
    )


def method_summary(rows, prefix):
    return {
        "cd_score": float(np.mean(
            [row[f"{prefix}_cd_score"] for row in rows]
        )),
        "p2s_score": float(np.mean(
            [row[f"{prefix}_p2s_score"] for row in rows]
        )),
        "final_score": float(np.mean(
            [row[f"{prefix}_final_score"] for row in rows]
        )),
        "final_gain": quantile_summary(
            [row[f"{prefix}_final_gain"] for row in rows]
        ),
        "improved_rate": float(np.mean(
            [row[f"{prefix}_final_gain"] > 0.0 for row in rows]
        )),
        "degraded_ge_1_rate": float(np.mean(
            [row[f"{prefix}_final_gain"] <= -1.0 for row in rows]
        )),
    }


def grouped_summary(rows, prefix):
    result = {"overall": method_summary(rows, prefix), "by_noise_band": {}}
    for band in BAND_NAMES:
        selected = [row for row in rows if row["noise_band"] == band]
        result["by_noise_band"][band] = method_summary(selected, prefix)
    return result


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
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--datalist", default="datalist/train.txt")
    parser.add_argument(
        "--category-reference-datalist",
        default="datalist/test.txt",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/full_cloud_three_expert_eval",
    )
    parser.add_argument("--max-shapes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--routing-mode",
        choices=["soft", "sparse"],
        default="soft",
    )
    parser.add_argument("--high-route-threshold", type=float, default=0.45)
    parser.add_argument("--adjacent-boundary", type=float, default=0.0125)
    parser.add_argument("--high-sigma-boundary", type=float, default=0.0145)
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
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded = excluded_paths(args.exclude_dataset)
    candidates = [
        rel_path
        for rel_path in read_datalist(args.datalist)
        if rel_path not in excluded
        and (Path(args.clean_root) / rel_path / "clean.npy").exists()
        and (
            Path(args.mesh_root)
            / rel_path
            / "models/model_normalized.obj"
        ).exists()
    ]
    category_reference = read_datalist(args.category_reference_datalist)
    usable = choose_paths(
        candidates,
        args.max_shapes,
        rng,
        category_reference=category_reference,
    )
    if not usable:
        raise FileNotFoundError("no complete held-out shapes found")
    band_indices = np.arange(len(usable)) % 3
    rng.shuffle(band_indices)

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

    rows = []
    for shape_index, rel_path in enumerate(usable):
        band_index = int(band_indices[shape_index])
        lower, upper = BAND_RANGES[band_index]
        args.noise_band_index = band_index
        args.noise_std = float(rng.uniform(lower, upper))
        print(
            f"[{shape_index + 1}/{len(usable)}] {rel_path} "
            f"band={BAND_NAMES[band_index]} sigma={args.noise_std:.5f}",
            flush=True,
        )
        clean_raw = np.load(
            Path(args.clean_root) / rel_path / "clean.npy"
        ).astype(np.float32, copy=False)
        clean, normalize_center, normalize_scale = normalize_clean(clean_raw)
        noisy = (
            clean
            + rng.standard_normal(clean.shape).astype(np.float32)
            * args.noise_std
        ).astype(np.float32, copy=False)
        mesh = trimesh.load(
            str(
                Path(args.mesh_root)
                / rel_path
                / "models/model_normalized.obj"
            ),
            process=False,
        )
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh_vertices = (
            np.asarray(mesh.vertices, dtype=np.float32) - normalize_center
        ) / max(normalize_scale, 1e-12)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int32)

        patches, seeds, point_indices, patch_distances = build_patches(
            noisy,
            args.patch_size,
            args.seed_k,
        )
        (
            coarse_patches,
            expert_patches,
            soft_patches,
            probabilities,
        ) = predict_patch_batches(vm, classifier, refiners, patches, args)
        seeds_np = seeds.numpy().astype(np.float32, copy=False)
        outputs = {
            "coarse": fuse_patches(
                noisy,
                coarse_patches,
                seeds_np,
                point_indices,
                patch_distances,
                args.fusion_tau,
            ),
            "soft": fuse_patches(
                noisy,
                soft_patches,
                seeds_np,
                point_indices,
                patch_distances,
                args.fusion_tau,
            ),
        }
        for expert_index, expert_name in enumerate(BAND_NAMES):
            outputs[f"expert_{expert_name}"] = fuse_patches(
                noisy,
                expert_patches[expert_index],
                seeds_np,
                point_indices,
                patch_distances,
                args.fusion_tau,
            )
        outputs["oracle"] = outputs[f"expert_{BAND_NAMES[band_index]}"]
        scores = {
            name: score_prediction(
                noisy,
                clean,
                prediction,
                mesh_vertices,
                mesh_faces,
            )
            for name, prediction in outputs.items()
        }
        row = {
            "rel_path": rel_path,
            "noise_band": BAND_NAMES[band_index],
            "noise_std": args.noise_std,
            "num_points": int(clean.shape[0]),
            "num_patches": int(patches.shape[0]),
            "mean_low_probability": float(probabilities[:, 0].mean()),
            "mean_medium_probability": float(probabilities[:, 1].mean()),
            "mean_high_probability": float(probabilities[:, 2].mean()),
        }
        for name, method_scores in scores.items():
            for key, value in method_scores.items():
                row[f"{name}_{key}"] = value
        expert_scores = [
            scores[f"expert_{expert_name}"]["final_score"]
            for expert_name in BAND_NAMES
        ]
        best_expert_index = int(np.argmax(expert_scores))
        outputs["best_expert"] = outputs[
            f"expert_{BAND_NAMES[best_expert_index]}"
        ]
        scores["best_expert"] = scores[
            f"expert_{BAND_NAMES[best_expert_index]}"
        ]
        row["best_expert"] = BAND_NAMES[best_expert_index]
        for key, value in scores["best_expert"].items():
            row[f"best_expert_{key}"] = value
        for name in [
            "oracle",
            "soft",
            "best_expert",
            "expert_low",
            "expert_medium",
            "expert_high",
        ]:
            row[f"{name}_cd_gain"] = (
                row[f"{name}_cd_score"] - row["coarse_cd_score"]
            )
            row[f"{name}_p2s_gain"] = (
                row[f"{name}_p2s_score"] - row["coarse_p2s_score"]
            )
            row[f"{name}_final_gain"] = (
                row[f"{name}_final_score"] - row["coarse_final_score"]
            )
        rows.append(row)
        write_csv(out_dir / "shape_eval.csv", rows)
        for name, prediction in outputs.items():
            output_path = out_dir / name / rel_path / "denoised.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, prediction)
        print(row, flush=True)

    summary = {
        "shape_count": len(rows),
        "noise_band_counts": {
            band: sum(row["noise_band"] == band for row in rows)
            for band in BAND_NAMES
        },
        "coarse": {
            key: float(np.mean([row[f"coarse_{key}"] for row in rows]))
            for key in ["cd_score", "p2s_score", "final_score"]
        },
        "oracle": grouped_summary(rows, "oracle"),
        "soft": grouped_summary(rows, "soft"),
        "best_expert": grouped_summary(rows, "best_expert"),
        "expert_matrix": {
            expert_name: grouped_summary(
                rows,
                f"expert_{expert_name}",
            )
            for expert_name in BAND_NAMES
        },
        "best_expert_counts": {
            true_band: {
                expert_name: sum(
                    row["noise_band"] == true_band
                    and row["best_expert"] == expert_name
                    for row in rows
                )
                for expert_name in BAND_NAMES
            }
            for true_band in BAND_NAMES
        },
        "routing_loss": {
            "overall": (
                grouped_summary(rows, "oracle")["overall"]["final_gain"]["mean"]
                - grouped_summary(rows, "soft")["overall"]["final_gain"]["mean"]
            ),
            "by_noise_band": {
                band: (
                    grouped_summary(rows, "oracle")["by_noise_band"][band][
                        "final_gain"
                    ]["mean"]
                    - grouped_summary(rows, "soft")["by_noise_band"][band][
                        "final_gain"
                    ]["mean"]
                )
                for band in BAND_NAMES
            },
        },
        "soft_to_best_loss": {
            "overall": (
                grouped_summary(rows, "best_expert")["overall"][
                    "final_gain"
                ]["mean"]
                - grouped_summary(rows, "soft")["overall"][
                    "final_gain"
                ]["mean"]
            ),
            "by_noise_band": {
                band: (
                    grouped_summary(rows, "best_expert")["by_noise_band"][
                        band
                    ]["final_gain"]["mean"]
                    - grouped_summary(rows, "soft")["by_noise_band"][band][
                        "final_gain"
                    ]["mean"]
                )
                for band in BAND_NAMES
            },
        },
        "excluded_shape_count": len(excluded),
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in ["noise_band_index", "noise_std"]
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
