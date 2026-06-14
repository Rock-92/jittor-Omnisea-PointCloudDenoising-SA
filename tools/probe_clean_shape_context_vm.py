import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.shape_context import (  # noqa: E402
    CleanShapeRegionProcessor,
    ShapeContextVMAdapter,
)
from tools.hard_patch_common import load_model, read_datalist  # noqa: E402
from tools.probe_fusion_aware_residual_oracle import (  # noqa: E402
    fuse_absolute,
    make_instance,
    predict_stage1,
)
from tools.train_full_cloud_fusion_probe import (  # noqa: E402
    choose_balanced_paths,
    farthest_point_indices,
    load_shape,
    parse_noise_bands,
    score_instance,
    usable_paths,
    validate_checkpoint_compatibility,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/vm/checkpoint_best.pkl"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/vm_pure_global.yaml"
DEFAULT_TRANSFORM_CONFIG = PROJECT_ROOT / "configs/transform/vm_pure_laplace.yaml"


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_clean_regions(clean, region_count, points_per_region):
    center_indices = farthest_point_indices(clean, region_count)
    centers = clean[center_indices]
    _, point_indices = cKDTree(clean).query(
        centers,
        k=min(int(points_per_region), clean.shape[0]),
    )
    point_indices = np.asarray(point_indices, dtype=np.int32)
    if point_indices.ndim == 1:
        point_indices = point_indices[:, None]
    region_points = clean[point_indices] - centers[:, None, :]
    return {
        "region_centers": centers.astype(np.float32, copy=False),
        "region_points": region_points.astype(np.float32, copy=False),
    }


def add_fixed_instance(shape, sigma, args, rng):
    instance = make_instance(
        shape,
        sigma=sigma,
        patch_size=args.patch_size,
        seed_k=args.seed_k,
        noise_type="laplace",
        rng=rng,
    )
    instance.update(
        build_clean_regions(
            instance["clean"],
            region_count=args.region_count,
            points_per_region=args.points_per_region,
        )
    )
    return instance


def cache_vm_predictions(vm, instance, args):
    stage1_patch = predict_stage1(
        vm,
        instance,
        batch_size=args.vm_batch_size,
        mode="one_step",
    )
    instance["vm_patch"] = stage1_patch
    instance["vm_prediction"] = fuse_absolute(
        instance,
        stage1_patch + instance["seeds"][:, None, :],
        fusion_tau=args.fusion_tau,
    )


def adapter_loss(prediction, coarse, clean_patch, aux, max_residual):
    target = clean_patch - coarse
    target_norm = np.sqrt(
        (target ** 2.0).sum(axis=-1, keepdims=True)
    )
    target_scale = np.minimum(
        1.0,
        float(max_residual) / np.maximum(target_norm, 1e-12),
    )
    target = jt.array((target * target_scale).astype(np.float32))
    residual = aux["residual"]
    scale = max(float(max_residual), 1e-6)
    vector = ((residual - target) ** 2.0).sum(dim=-1).mean() / (scale ** 2.0)
    residual_norm = jt.sqrt(
        (residual ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
    )
    target_length = jt.sqrt(
        (target ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
    )
    cosine = (residual * target).sum(dim=-1, keepdims=True) / (
        residual_norm * target_length + 1e-8
    )
    direction_weight = jt.minimum(
        target_length / scale,
        jt.ones_like(target_length),
    )
    direction = (
        ((1.0 - cosine) * direction_weight).sum()
        / (direction_weight.sum() + 1e-6)
    )
    length = (jt.abs(residual_norm - target_length) / scale).mean()
    paired = (
        ((prediction - jt.array(clean_patch)) ** 2.0)
        .sum(dim=-1)
        .mean()
        / (scale ** 2.0)
    )
    loss = vector + 0.5 * direction + 0.5 * length + 0.05 * paired
    return loss, {
        "vector": vector,
        "direction": direction,
        "length": length,
        "paired": paired,
        "cosine": cosine.mean(),
        "residual_mean": residual_norm.mean(),
        "target_mean": target_length.mean(),
        "gate_mean": aux["gate"].mean(),
    }


def shape_tokens(processor, instance):
    return processor(
        jt.array(instance["region_points"][None, :, :, :]),
        jt.array(instance["region_centers"][None, :, :]),
    )


def refine_patch_batch(
    adapter,
    instance,
    patch_indices,
    region_tokens,
    global_token,
):
    patch_indices = np.asarray(patch_indices, dtype=np.int32)
    seeds = instance["seeds"][patch_indices]
    noisy_local = instance["patches"][patch_indices]
    coarse_local = instance["vm_patch"][patch_indices]
    point_global = coarse_local + seeds[:, None, :]
    prediction, aux = adapter(
        jt.array(noisy_local),
        jt.array(coarse_local),
        jt.array(point_global.astype(np.float32)),
        region_tokens,
        jt.array(instance["region_centers"][None, :, :]),
        global_token,
    )
    return prediction, aux


def refine_full_cloud(
    processor,
    adapter,
    target_instance,
    context_instance,
    args,
):
    region_tokens, global_token = shape_tokens(
        processor,
        context_instance,
    )
    outputs = []
    for start in range(0, target_instance["patches"].shape[0], args.batch_size):
        end = min(
            start + args.batch_size,
            target_instance["patches"].shape[0],
        )
        patch_indices = np.arange(start, end, dtype=np.int32)
        seeds = target_instance["seeds"][patch_indices]
        noisy_local = target_instance["patches"][patch_indices]
        coarse_local = target_instance["vm_patch"][patch_indices]
        point_global = coarse_local + seeds[:, None, :]
        prediction, _ = adapter(
            jt.array(noisy_local),
            jt.array(coarse_local),
            jt.array(point_global.astype(np.float32)),
            region_tokens,
            jt.array(context_instance["region_centers"][None, :, :]),
            global_token,
        )
        outputs.append(prediction.numpy().astype(np.float32, copy=False))
    patch_prediction = np.concatenate(outputs, axis=0)
    return fuse_absolute(
        target_instance,
        patch_prediction + target_instance["seeds"][:, None, :],
        fusion_tau=args.fusion_tau,
    )


def summarize(rows, prefix):
    return {
        metric: float(np.mean([row[f"{prefix}_{metric}"] for row in rows]))
        for metric in ("cd_score", "p2s_score", "final_score")
    }


def evaluate(processor, adapter, instances, args):
    processor.eval()
    adapter.eval()
    rows = []
    with jt.no_grad():
        for index, instance in enumerate(instances):
            shuffled = instances[(index + 1) % len(instances)]
            correct_prediction = refine_full_cloud(
                processor,
                adapter,
                instance,
                instance,
                args,
            )
            shuffled_prediction = refine_full_cloud(
                processor,
                adapter,
                instance,
                shuffled,
                args,
            )
            vm_score = score_instance(instance, instance["vm_prediction"])
            correct_score = score_instance(instance, correct_prediction)
            shuffled_score = score_instance(instance, shuffled_prediction)
            row = {
                "rel_path": instance["rel_path"],
                "sigma": instance["sigma"],
            }
            for prefix, score in (
                ("vm", vm_score),
                ("correct", correct_score),
                ("shuffled", shuffled_score),
            ):
                for metric in ("cd_score", "p2s_score", "final_score"):
                    row[f"{prefix}_{metric}"] = score[metric]
            row["correct_gain"] = (
                correct_score["final_score"] - vm_score["final_score"]
            )
            row["shuffled_gain"] = (
                shuffled_score["final_score"] - vm_score["final_score"]
            )
            row["context_specific_gain"] = (
                correct_score["final_score"] - shuffled_score["final_score"]
            )
            rows.append(row)
            print(
                f"  eval [{index + 1}/{len(instances)}] "
                f"vm={vm_score['final_score']:.3f} "
                f"correct={correct_score['final_score']:.3f} "
                f"shuffled={shuffled_score['final_score']:.3f}",
                flush=True,
            )
            jt.gc()
    summary = {
        "count": len(rows),
        "vm": summarize(rows, "vm"),
        "correct": summarize(rows, "correct"),
        "shuffled": summarize(rows, "shuffled"),
        "correct_gain": float(np.mean([row["correct_gain"] for row in rows])),
        "shuffled_gain": float(np.mean([row["shuffled_gain"] for row in rows])),
        "context_specific_gain": float(
            np.mean([row["context_specific_gain"] for row in rows])
        ),
        "correct_improved_rate": float(
            np.mean([row["correct_gain"] > 0.0 for row in rows])
        ),
        "correct_min_gain": float(
            np.min([row["correct_gain"] for row in rows])
        ),
    }
    return rows, summary


def optimizer_grad_norm(optimizer):
    gradients = []
    for group in optimizer.param_groups:
        for parameter, gradient in zip(group["params"], group["grads"]):
            if not parameter.is_stop_grad():
                gradients.append(gradient.flatten())
    if not gradients:
        return 0.0
    return float(jt.norm(jt.concat(gradients), 2).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--transform-config", default=str(DEFAULT_TRANSFORM_CONFIG))
    parser.add_argument("--train-list", default="datalist/train.txt")
    parser.add_argument("--val-list", default="datalist/validate.txt")
    parser.add_argument("--category-reference-list", default="datalist/test.txt")
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument(
        "--out-dir",
        default="outputs/clean_shape_context_vm_probe",
    )
    parser.add_argument("--train-shapes", type=int, default=40)
    parser.add_argument("--val-shapes", type=int, default=10)
    parser.add_argument("--patches-per-shape", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--vm-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--region-count", type=int, default=256)
    parser.add_argument("--points-per-region", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--context-knn", type=int, default=4)
    parser.add_argument("--max-residual", type=float, default=0.008)
    parser.add_argument("--noise-bands", default="0.005:0.010,0.010:0.015,0.015:0.020")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--allow-nonstandard-protocol", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    if (
        args.num_points != 32768
        or args.patch_size != 1000
        or args.seed_k != 6.0
    ) and not args.allow_nonstandard_protocol:
        raise ValueError(
            "probe must use 32768 points / 1000-point patches / seed_k=6; "
            "use --allow-nonstandard-protocol only for smoke tests"
        )
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    selection_rng = np.random.default_rng(args.seed)
    train_rng = np.random.default_rng(args.seed + 1)
    val_rng = np.random.default_rng(args.seed + 2)
    bands = parse_noise_bands(args.noise_bands)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vm = load_model(
        args.checkpoint,
        model_config=args.model_config,
        transform_config=args.transform_config,
    )
    compatibility = validate_checkpoint_compatibility(vm, args.checkpoint)
    if vm.use_edm:
        raise RuntimeError("probe requires pure VM with use_edm=false")
    vm.eval()
    for parameter in vm.parameters():
        parameter.stop_grad()

    reference = read_datalist(args.category_reference_list)
    train_candidates = usable_paths(
        read_datalist(args.train_list),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    val_candidates = usable_paths(
        read_datalist(args.val_list),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    train_paths = choose_balanced_paths(
        train_candidates,
        args.train_shapes,
        reference,
        selection_rng,
    )
    val_paths = choose_balanced_paths(
        val_candidates,
        args.val_shapes,
        reference,
        selection_rng,
    )
    print("Building fixed Laplace instances and caching pure VM...", flush=True)
    train_instances = []
    val_instances = []
    for output, paths, rng in (
        (train_instances, train_paths, train_rng),
        (val_instances, val_paths, val_rng),
    ):
        for index, path in enumerate(tqdm(paths, unit="shape")):
            shape = load_shape(
                path,
                args.clean_root,
                args.mesh_root,
                args.num_points,
                rng,
                args.sample_missing_clean,
            )
            lower, upper = bands[index % len(bands)]
            instance = add_fixed_instance(
                shape,
                sigma=float(rng.uniform(lower, upper)),
                args=args,
                rng=rng,
            )
            instance["noise_band"] = (lower, upper)
            cache_vm_predictions(vm, instance, args)
            output.append(instance)

    processor = CleanShapeRegionProcessor(token_dim=args.token_dim)
    adapter = ShapeContextVMAdapter(
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        context_knn=args.context_knn,
        max_residual=args.max_residual,
    )
    parameters = list(processor.parameters()) + list(adapter.parameters())
    optimizer = jt.optim.Adam(parameters, lr=args.lr)
    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "train_paths": train_paths,
            "val_paths": val_paths,
            "checkpoint_compatibility": compatibility,
            "protocol": {
                "vm_use_edm": False,
                "noise_type": "laplace",
            },
        },
    )

    best_score = -1e9
    best_summary = None
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        optimizer.lr = float(
            args.min_lr
            + 0.5 * (args.lr - args.min_lr) * (1.0 + np.cos(np.pi * progress))
        )
        processor.train()
        adapter.train()
        order = train_rng.permutation(len(train_instances))
        metrics = []
        start_time = time.time()
        bar = tqdm(order, desc=f"Epoch {epoch}", unit="shape")
        for shape_index in bar:
            instance = train_instances[int(shape_index)]
            patch_count = instance["patches"].shape[0]
            patch_indices = train_rng.choice(
                patch_count,
                size=min(args.patches_per_shape, patch_count),
                replace=False,
            ).astype(np.int32)
            region_tokens, global_token = shape_tokens(processor, instance)
            prediction, aux = refine_patch_batch(
                adapter,
                instance,
                patch_indices,
                region_tokens,
                global_token,
            )
            clean_patch = (
                instance["clean"][instance["point_indices"][patch_indices]]
                - instance["seeds"][patch_indices, None, :]
            ).astype(np.float32)
            coarse = instance["vm_patch"][patch_indices]
            loss, parts = adapter_loss(
                prediction,
                coarse,
                clean_patch,
                aux,
                args.max_residual,
            )
            optimizer.zero_grad()
            optimizer.backward(loss)
            grad_norm = optimizer_grad_norm(optimizer)
            if not np.isfinite(grad_norm):
                optimizer.zero_grad()
                continue
            optimizer.clip_grad_norm(args.grad_clip)
            optimizer.step()
            row = {
                "loss": float(loss.item()),
                "grad_norm": grad_norm,
                **{key: float(value.item()) for key, value in parts.items()},
            }
            metrics.append(row)
            bar.set_postfix(
                loss=f"{row['loss']:.3f}",
                cosine=f"{row['cosine']:.3f}",
                residual=f"{row['residual_mean']:.4f}",
            )
            jt.gc()
        record = {
            "epoch": epoch,
            "lr": float(optimizer.lr),
            "seconds": float(time.time() - start_time),
            **{
                f"train_{key}": float(np.mean([row[key] for row in metrics]))
                for key in metrics[0]
            },
        }
        if (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        ):
            print(f"Epoch {epoch} full-cloud context evaluation:", flush=True)
            rows, summary = evaluate(
                processor,
                adapter,
                val_instances,
                args,
            )
            record.update(
                {
                    "val_vm": summary["vm"]["final_score"],
                    "val_correct": summary["correct"]["final_score"],
                    "val_shuffled": summary["shuffled"]["final_score"],
                    "val_correct_gain": summary["correct_gain"],
                    "val_context_specific_gain": summary[
                        "context_specific_gain"
                    ],
                }
            )
            if summary["correct"]["final_score"] > best_score:
                best_score = summary["correct"]["final_score"]
                best_summary = summary
                best_epoch = epoch
                processor.save(str(out_dir / "processor_best.pkl"))
                adapter.save(str(out_dir / "adapter_best.pkl"))
                write_csv(out_dir / "best_rows.csv", rows)
                write_json(out_dir / "best_summary.json", summary)
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        processor.save(str(out_dir / "processor_last.pkl"))
        adapter.save(str(out_dir / "adapter_last.pkl"))
        print(record, flush=True)

    decision = {
        "shape_context_has_signal": (
            best_summary["correct_gain"] >= 2.0
            and best_summary["context_specific_gain"] >= 1.0
        ),
        "criteria": {
            "correct_gain_ge_2": best_summary["correct_gain"] >= 2.0,
            "correct_minus_shuffled_ge_1": (
                best_summary["context_specific_gain"] >= 1.0
            ),
        },
        "best_epoch": best_epoch,
        "best": best_summary,
        "args": vars(args),
    }
    write_json(out_dir / "probe_summary.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
