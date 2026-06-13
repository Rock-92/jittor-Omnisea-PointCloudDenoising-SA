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

from src.model.refinement import FusionAwareResidualRefiner  # noqa: E402
from tools.hard_patch_common import load_model, read_datalist  # noqa: E402
from tools.probe_fusion_aware_residual_oracle import (  # noqa: E402
    fuse_absolute,
    make_instance,
    predict_stage1,
)
from tools.train_full_cloud_fusion_probe import (  # noqa: E402
    choose_balanced_paths,
    load_shape,
    normalize_clean,
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


def finite(value):
    return bool(np.isfinite(float(value.item())))


def optimizer_grad_norm(optimizer):
    gradients = []
    for group in optimizer.param_groups:
        for parameter, gradient in zip(group["params"], group["grads"]):
            if parameter.is_stop_grad():
                continue
            gradients.append(gradient.flatten())
    if not gradients:
        return 0.0
    return float(jt.norm(jt.concat(gradients), 2).item())


def build_training_overlap_layout(noisy, patch_size, group_size, rng):
    point_count = noisy.shape[0]
    patch_size = min(int(patch_size), point_count)
    group_size = min(max(int(group_size), 1), point_count)
    tree = cKDTree(noisy)
    anchor_seed = int(rng.integers(0, point_count))
    anchor_distances, anchor_points = tree.query(
        noisy[anchor_seed],
        k=patch_size,
    )
    anchor_points = np.asarray(anchor_points, dtype=np.int64).reshape(-1)
    anchor_distances = np.asarray(
        anchor_distances,
        dtype=np.float32,
    ).reshape(-1)
    inner_count = max(group_size, int(0.6 * patch_size))
    inner_points = anchor_points[:inner_count]
    candidates = inner_points[inner_points != anchor_seed]
    extra_count = min(group_size - 1, candidates.shape[0])
    extra_seeds = rng.choice(
        candidates,
        size=extra_count,
        replace=False,
    ).astype(np.int64)
    seed_indices = np.concatenate(
        [np.asarray([anchor_seed], dtype=np.int64), extra_seeds]
    )
    all_indices = [anchor_points]
    all_distances = [anchor_distances ** 2.0]
    for seed_index in seed_indices[1:]:
        distances, indices = tree.query(noisy[int(seed_index)], k=patch_size)
        all_indices.append(
            np.asarray(indices, dtype=np.int64).reshape(-1)
        )
        all_distances.append(
            np.asarray(distances, dtype=np.float32).reshape(-1) ** 2.0
        )
    point_indices = np.stack(all_indices)
    distances = np.stack(all_distances)
    normalized_distances = distances / np.maximum(
        distances[:, -1:],
        1e-8,
    )
    seeds = noisy[seed_indices]
    return {
        "patches": (noisy[point_indices] - seeds[:, None, :]).astype(
            np.float32,
            copy=False,
        ),
        "seeds": seeds.astype(np.float32, copy=False),
        "point_indices": point_indices.astype(np.int32, copy=False),
        "normalized_distances": normalized_distances.astype(
            np.float32,
            copy=False,
        ),
    }


def fuse_group_numpy(instance, patch_absolute, patch_indices, fusion_tau):
    point_count = instance["noisy"].shape[0]
    selected_points = instance["point_indices"][patch_indices]
    selected_distances = instance["normalized_distances"][patch_indices]
    weights = np.exp(-float(fusion_tau) * selected_distances).astype(np.float64)
    weighted_sum = np.zeros((point_count, 3), dtype=np.float64)
    weight_sum = np.zeros((point_count,), dtype=np.float64)
    np.add.at(
        weighted_sum,
        selected_points.reshape(-1),
        (patch_absolute * weights[:, :, None]).reshape(-1, 3),
    )
    np.add.at(weight_sum, selected_points.reshape(-1), weights.reshape(-1))
    fused = instance["noisy"].copy()
    covered = weight_sum > 0.0
    fused[covered] = (
        weighted_sum[covered] / weight_sum[covered, None]
    ).astype(np.float32)
    return fused


def fuse_group_jittor(instance, patch_absolute, patch_indices, fusion_tau):
    point_indices_np = instance["point_indices"][patch_indices]
    weights_np = np.exp(
        -float(fusion_tau)
        * instance["normalized_distances"][patch_indices]
    ).astype(np.float32)
    flat_indices = jt.array(point_indices_np.reshape(-1)).int32()
    flat_absolute = patch_absolute.reshape(-1, 3)
    flat_weights = jt.array(weights_np.reshape(-1, 1))
    index_3d = flat_indices.reshape(-1, 1).broadcast(flat_absolute.shape)
    weighted_sum = jt.zeros(
        (instance["noisy"].shape[0], 3)
    ).scatter(
        0,
        index_3d,
        flat_absolute * flat_weights,
        reduce="add",
    )
    weight_sum_np = np.zeros(
        (instance["noisy"].shape[0], 1),
        dtype=np.float32,
    )
    np.add.at(
        weight_sum_np[:, 0],
        point_indices_np.reshape(-1),
        weights_np.reshape(-1),
    )
    return weighted_sum / (jt.array(weight_sum_np) + 1e-8)


def run_refiner(refiner, instance, stage1_patch, patch_indices, fusion_tau):
    patch_indices = np.asarray(patch_indices, dtype=np.int32)
    seeds = instance["seeds"][patch_indices]
    coarse = stage1_patch[patch_indices]
    coarse_absolute = coarse + seeds[:, None, :]
    consensus_absolute = fuse_group_numpy(
        instance,
        coarse_absolute,
        patch_indices,
        fusion_tau=fusion_tau,
    )
    point_indices = instance["point_indices"][patch_indices]
    consensus = (
        consensus_absolute[point_indices] - seeds[:, None, :]
    ).astype(np.float32)
    prediction, aux = refiner(
        jt.array(coarse),
        jt.array(instance["patches"][patch_indices]),
        jt.array(consensus),
        jt.array(
            instance["normalized_distances"][patch_indices, :, None]
        ),
    )
    return prediction, aux


def stage2_loss(
    instance,
    patch_indices,
    anchor_index,
    prediction,
    aux,
    fusion_tau,
    max_residual,
    args,
):
    seeds = jt.array(instance["seeds"][patch_indices]).unsqueeze(1)
    absolute = prediction + seeds
    fused = fuse_group_jittor(
        instance,
        absolute,
        patch_indices,
        fusion_tau=fusion_tau,
    )
    anchor_points_np = instance["point_indices"][int(anchor_index)]
    anchor_points = jt.array(anchor_points_np).int32()
    clean = jt.array(instance["clean"])
    scale = max(float(max_residual), 1e-6)
    scale2 = scale ** 2.0

    fused_paired = (
        ((fused[anchor_points] - clean[anchor_points]) ** 2.0)
        .sum(dim=-1)
        .mean()
        / scale2
    )
    point_indices = jt.array(instance["point_indices"][patch_indices]).int32()
    clean_patch = clean[point_indices] - seeds
    patch_paired = (
        ((prediction - clean_patch) ** 2.0).sum(dim=-1).mean() / scale2
    )
    coarse = jt.array(
        instance["stage1_patch"][patch_indices]
    )
    target = clean_patch - coarse
    target_length = jt.sqrt(
        (target ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
    )
    residual = aux["residual"]
    residual_length = jt.sqrt(
        (residual ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
    )
    cosine = (residual * target).sum(dim=-1, keepdims=True) / (
        residual_length * target_length + 1e-8
    )
    direction_map = jt.minimum(
        target_length / scale,
        jt.ones_like(target_length),
    )
    direction = (
        ((1.0 - cosine) * direction_map).sum()
        / (direction_map.sum() + 1e-6)
    )
    length = (jt.abs(residual_length - target_length) / scale).mean()
    keep_map = jt.exp(
        -target_length / max(float(args.keep_threshold), 1e-6)
    )
    keep = (
        keep_map * (residual_length / scale) ** 2.0
    ).sum() / (keep_map.sum() + 1e-6)

    gate_target = jt.minimum(
        target_length / scale,
        jt.ones_like(target_length),
    )
    gate_target.stop_grad()
    gate = jt.minimum(jt.maximum(aux["gate"], 1e-5), 1.0 - 1e-5)
    gate_loss = -(
        gate_target * jt.log(gate)
        + (1.0 - gate_target) * jt.log(1.0 - gate)
    ).mean()

    fused_patch = fused[point_indices]
    overlap = (
        ((absolute - fused_patch) ** 2.0).sum(dim=-1).mean() / scale2
    )
    total = (
        float(args.fused_weight) * fused_paired
        + float(args.patch_weight) * patch_paired
        + float(args.direction_weight) * direction
        + float(args.length_weight) * length
        + float(args.keep_weight) * keep
        + float(args.gate_weight) * gate_loss
        + float(args.overlap_weight) * overlap
    )
    return total, {
        "fused_paired": fused_paired,
        "patch_paired": patch_paired,
        "direction": direction,
        "length": length,
        "keep": keep,
        "gate": gate_loss,
        "overlap": overlap,
        "residual_mean": residual_length.mean(),
        "gate_mean": aux["gate"].mean(),
    }


def refine_full_cloud(
    refiner,
    instance,
    stage1_patch,
    batch_size,
    fusion_tau,
):
    stage1_absolute = stage1_patch + instance["seeds"][:, None, :]
    consensus_absolute = fuse_absolute(
        instance,
        stage1_absolute,
        fusion_tau=fusion_tau,
    )
    outputs = []
    refiner.eval()
    with jt.no_grad():
        for start in range(0, stage1_patch.shape[0], batch_size):
            end = min(start + batch_size, stage1_patch.shape[0])
            indices = np.arange(start, end, dtype=np.int32)
            points = instance["point_indices"][indices]
            seeds = instance["seeds"][indices]
            consensus = (
                consensus_absolute[points] - seeds[:, None, :]
            ).astype(np.float32)
            prediction, _ = refiner(
                jt.array(stage1_patch[indices]),
                jt.array(instance["patches"][indices]),
                jt.array(consensus),
                jt.array(
                    instance["normalized_distances"][indices, :, None]
                ),
            )
            outputs.append(prediction.numpy().astype(np.float32, copy=False))
    refined_patch = np.concatenate(outputs, axis=0)
    refined_absolute = refined_patch + instance["seeds"][:, None, :]
    return fuse_absolute(
        instance,
        refined_absolute,
        fusion_tau=fusion_tau,
    )


def evaluate(refiner, stage1, instances, args):
    rows = []
    for index, instance in enumerate(instances):
        stage1_patch = predict_stage1(
            stage1,
            instance,
            batch_size=args.stage1_batch_size,
            mode=args.stage1_mode,
        )
        baseline = fuse_absolute(
            instance,
            stage1_patch + instance["seeds"][:, None, :],
            fusion_tau=args.fusion_tau,
        )
        refined = refine_full_cloud(
            refiner,
            instance,
            stage1_patch,
            batch_size=args.stage2_batch_size,
            fusion_tau=args.fusion_tau,
        )
        base_score = score_instance(instance, baseline)
        refined_score = score_instance(instance, refined)
        row = {
            "rel_path": instance["rel_path"],
            "sigma": instance["sigma"],
            **{f"base_{key}": value for key, value in base_score.items()
               if key.endswith("_score")},
            **{f"refined_{key}": value for key, value in refined_score.items()
               if key.endswith("_score")},
            "final_gain": (
                refined_score["final_score"] - base_score["final_score"]
            ),
        }
        rows.append(row)
        print(
            f"  eval [{index + 1}/{len(instances)}] "
            f"base={base_score['final_score']:.3f} "
            f"refined={refined_score['final_score']:.3f} "
            f"gain={row['final_gain']:+.3f}",
            flush=True,
        )
        jt.gc()
    return rows, summarize(rows)


def summarize(rows):
    return {
        "count": len(rows),
        "base": {
            metric: float(np.mean([row[f"base_{metric}"] for row in rows]))
            for metric in ("cd_score", "p2s_score", "final_score")
        },
        "refined": {
            metric: float(
                np.mean([row[f"refined_{metric}"] for row in rows])
            )
            for metric in ("cd_score", "p2s_score", "final_score")
        },
        "gain": {
            "mean": float(np.mean([row["final_gain"] for row in rows])),
            "median": float(np.median([row["final_gain"] for row in rows])),
            "min": float(np.min([row["final_gain"] for row in rows])),
            "improved_rate": float(
                np.mean([row["final_gain"] > 0.0 for row in rows])
            ),
        },
    }


def load_selected_shape(path, args, rng):
    return load_shape(
        path,
        clean_root=args.clean_root,
        mesh_root=args.mesh_root,
        num_points=args.num_points,
        rng=rng,
        sample_missing_clean=args.sample_missing_clean,
    )


def load_training_shape(path, args, rng):
    clean_path = Path(args.clean_root) / path / "clean.npy"
    if not clean_path.exists():
        return load_selected_shape(path, args, rng)
    clean_raw = np.load(clean_path).astype(np.float32, copy=False)
    clean, _, _ = normalize_clean(clean_raw)
    if args.num_points > 0 and clean.shape[0] > args.num_points:
        indices = rng.choice(
            clean.shape[0],
            size=int(args.num_points),
            replace=False,
        )
        clean = clean[indices]
    return {
        "rel_path": path,
        "clean": clean.astype(np.float32, copy=False),
    }


def make_training_instance(shape, bands, args, rng):
    lower, upper = bands[int(rng.integers(0, len(bands)))]
    sigma = float(rng.uniform(lower, upper))
    clean = shape["clean"]
    if args.noise_type == "laplace":
        noise = rng.laplace(0.0, sigma, clean.shape)
    else:
        noise = rng.standard_normal(clean.shape) * sigma
    noisy = (clean + noise.astype(np.float32)).astype(
        np.float32,
        copy=False,
    )
    return {
        **shape,
        **build_training_overlap_layout(
            noisy,
            patch_size=args.patch_size,
            group_size=args.group_size,
            rng=rng,
        ),
        "noisy": noisy,
        "sigma": sigma,
        "noise_band": (lower, upper),
    }


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
        default="outputs/fusion_aware_stage2_full",
    )
    parser.add_argument(
        "--train-shapes",
        type=int,
        default=0,
        help="Candidate pool size; 0 uses every usable training shape.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=800)
    parser.add_argument("--val-shapes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--stage1-batch-size", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument(
        "--stage1-mode",
        choices=["one_step", "heun"],
        default="one_step",
    )
    parser.add_argument(
        "--noise-bands",
        default="0.005:0.010,0.010:0.015,0.015:0.020",
    )
    parser.add_argument(
        "--noise-type",
        choices=["laplace", "gaussian"],
        default="laplace",
    )
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--max-residual", type=float, default=0.008)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--load-refiner", default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--fused-weight", type=float, default=1.0)
    parser.add_argument("--patch-weight", type=float, default=0.5)
    parser.add_argument("--direction-weight", type=float, default=0.2)
    parser.add_argument("--length-weight", type=float, default=0.2)
    parser.add_argument("--keep-weight", type=float, default=0.3)
    parser.add_argument("--keep-threshold", type=float, default=0.002)
    parser.add_argument("--gate-weight", type=float, default=0.2)
    parser.add_argument("--overlap-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260613)
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
            "training must use 32768 points / 1000-point patches / seed_k=6; "
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
    checkpoint = Path(args.checkpoint)
    stage1 = load_model(
        checkpoint,
        model_config=args.model_config,
        transform_config=args.transform_config,
    )
    compatibility = validate_checkpoint_compatibility(stage1, checkpoint)
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.stop_grad()

    reference_paths = read_datalist(args.category_reference_list)
    train_candidates = usable_paths(
        read_datalist(args.train_list),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    if args.train_shapes > 0:
        train_paths = choose_balanced_paths(
            train_candidates,
            args.train_shapes,
            reference_paths,
            selection_rng,
        )
    else:
        train_paths = list(train_candidates)
        selection_rng.shuffle(train_paths)
    val_candidates = usable_paths(
        read_datalist(args.val_list),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    val_paths = choose_balanced_paths(
        val_candidates,
        args.val_shapes,
        reference_paths,
        selection_rng,
    )
    if not train_paths or not val_paths:
        raise FileNotFoundError(
            "no usable shapes; generate cache_clean_points or pass "
            "--sample-missing-clean"
        )

    print(
        f"Training candidate pool: {len(train_paths)} shapes; "
        f"steps/epoch: {args.steps_per_epoch}",
        flush=True,
    )
    val_instances = []
    for index, path in enumerate(val_paths):
        shape = load_selected_shape(path, args, val_rng)
        lower, upper = bands[index % len(bands)]
        sigma = float(val_rng.uniform(lower, upper))
        instance = make_instance(
            shape,
            sigma=sigma,
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            noise_type=args.noise_type,
            rng=val_rng,
        )
        instance["noise_band"] = (lower, upper)
        val_instances.append(instance)

    refiner = FusionAwareResidualRefiner(
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        max_residual=args.max_residual,
    )
    if args.load_refiner:
        refiner.load(str(args.load_refiner))
        print(f"Loaded Stage2 checkpoint: {args.load_refiner}", flush=True)
    parameters = list(refiner.parameters())
    optimizer = jt.optim.Adam(parameters, lr=args.lr)
    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "train_candidate_count": len(train_paths),
            "val_paths": val_paths,
            "checkpoint_compatibility": compatibility,
        },
    )

    print("Baseline full-cloud validation:", flush=True)
    baseline_rows, baseline_summary = evaluate(
        refiner,
        stage1,
        val_instances,
        args,
    )
    write_csv(out_dir / "baseline_rows.csv", baseline_rows)
    write_json(out_dir / "baseline_summary.json", baseline_summary)
    write_csv(out_dir / "best_rows.csv", baseline_rows)
    write_json(out_dir / "best_summary.json", baseline_summary)
    best_score = baseline_summary["refined"]["final_score"]
    best_summary = baseline_summary
    best_epoch = -1
    refiner.save(str(out_dir / "refiner_best.pkl"))
    history = []
    shape_schedule = train_rng.permutation(len(train_paths))
    schedule_cursor = 0

    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1.0 + np.cos(np.pi * progress)
        )
        optimizer.lr = float(lr)
        refiner.train()
        epoch_steps = min(args.steps_per_epoch, len(train_paths))
        order_parts = []
        remaining = epoch_steps
        while remaining > 0:
            available = len(shape_schedule) - schedule_cursor
            take = min(remaining, available)
            order_parts.append(
                shape_schedule[schedule_cursor:schedule_cursor + take]
            )
            schedule_cursor += take
            remaining -= take
            if schedule_cursor == len(shape_schedule):
                shape_schedule = train_rng.permutation(len(train_paths))
                schedule_cursor = 0
        order = np.concatenate(order_parts)
        metrics = []
        skipped = 0
        epoch_start = time.time()
        bar = tqdm(order, desc=f"Epoch {epoch}", unit="shape")
        for shape_index in bar:
            shape = load_training_shape(
                train_paths[int(shape_index)],
                args,
                train_rng,
            )
            instance = make_training_instance(shape, bands, args, train_rng)
            anchor_index = 0
            patch_indices = np.arange(
                instance["patches"].shape[0],
                dtype=np.int32,
            )
            stage1_patch = predict_stage1(
                stage1,
                instance,
                batch_size=args.stage1_batch_size,
                mode=args.stage1_mode,
            )
            instance["stage1_patch"] = stage1_patch
            prediction, aux = run_refiner(
                refiner,
                instance,
                stage1_patch,
                patch_indices,
                fusion_tau=args.fusion_tau,
            )
            loss, parts = stage2_loss(
                instance,
                patch_indices,
                anchor_index,
                prediction,
                aux,
                fusion_tau=args.fusion_tau,
                max_residual=args.max_residual,
                args=args,
            )
            if not finite(loss):
                skipped += 1
                optimizer.zero_grad()
                jt.gc()
                continue
            optimizer.zero_grad()
            optimizer.backward(loss)
            grad_norm = optimizer_grad_norm(optimizer)
            if not np.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad()
                jt.gc()
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
                gate=f"{row['gate_mean']:.3f}",
                residual=f"{row['residual_mean']:.4f}",
            )
            jt.gc()

        if not metrics:
            raise RuntimeError("all training steps were skipped")
        record = {
            "epoch": epoch,
            "lr": float(lr),
            "seconds": float(time.time() - epoch_start),
            "steps": len(metrics),
            "skipped": skipped,
            **{
                f"train_{key}": float(np.mean([row[key] for row in metrics]))
                for key in metrics[0]
            },
        }
        should_eval = (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        )
        if should_eval:
            print(f"Epoch {epoch} full-cloud validation:", flush=True)
            val_rows, val_summary = evaluate(
                refiner,
                stage1,
                val_instances,
                args,
            )
            val_score = val_summary["refined"]["final_score"]
            record.update(
                {
                    "val_base": val_summary["base"]["final_score"],
                    "val_refined": val_score,
                    "val_gain": val_summary["gain"]["mean"],
                    "val_improved_rate": val_summary["gain"]["improved_rate"],
                }
            )
            if val_score > best_score:
                best_score = val_score
                best_summary = val_summary
                best_epoch = epoch
                refiner.save(str(out_dir / "refiner_best.pkl"))
                write_csv(out_dir / "best_rows.csv", val_rows)
                write_json(out_dir / "best_summary.json", val_summary)
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        refiner.save(str(out_dir / "refiner_last.pkl"))
        print(record, flush=True)

    train_summary = {
        "best_epoch": best_epoch,
        "baseline": baseline_summary,
        "best": best_summary,
        "best_gain_over_vm": (
            best_summary["refined"]["final_score"]
            - baseline_summary["base"]["final_score"]
        ),
        "go_no_go": {
            "expand_or_submit": best_summary["gain"]["mean"] >= 5.0,
            "target_gain": 5.0,
        },
        "args": vars(args),
    }
    write_json(out_dir / "train_summary.json", train_summary)
    print(json.dumps(train_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
