import argparse
import csv
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.hard_patch_common import load_model, read_datalist  # noqa: E402
from tools.train_full_cloud_fusion_probe import (  # noqa: E402
    build_patch_layout,
    category_of,
    choose_balanced_paths,
    load_shape,
    parse_noise_bands,
    score_instance,
    usable_paths,
    validate_checkpoint_compatibility,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/vm/checkpoint_best.pkl"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/vm_pure_global.yaml"
DEFAULT_TRANSFORM_CONFIG = (
    PROJECT_ROOT / "configs/transform/vm_pure_laplace.yaml"
)


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


def parse_budgets(value):
    budgets = sorted(
        {
            float(item)
            for item in str(value).split(",")
            if str(item).strip()
        }
    )
    if not budgets or budgets[0] <= 0:
        raise ValueError("residual budgets must be positive")
    return budgets


def budget_key(budget):
    return f"{float(budget):.4f}".replace(".", "p")


def make_instance(shape, sigma, patch_size, seed_k, noise_type, rng):
    clean = shape["clean"]
    if noise_type == "laplace":
        noise = rng.laplace(0.0, float(sigma), clean.shape)
    elif noise_type == "gaussian":
        noise = rng.standard_normal(clean.shape) * float(sigma)
    else:
        raise ValueError(f"unsupported noise type: {noise_type}")
    noisy = (clean + noise.astype(np.float32)).astype(
        np.float32,
        copy=False,
    )
    layout = build_patch_layout(
        noisy,
        patch_size=patch_size,
        seed_k=seed_k,
    )
    return {
        **shape,
        **layout,
        "noisy": noisy,
        "sigma": float(sigma),
    }


def predict_stage1(model, instance, batch_size, mode):
    predictions = []
    model.eval()
    with jt.no_grad():
        for start in range(0, instance["patches"].shape[0], batch_size):
            end = min(start + batch_size, instance["patches"].shape[0])
            patches = jt.array(instance["patches"][start:end])
            if mode == "one_step":
                if getattr(model, "use_edm", False):
                    sigma = (
                        jt.ones((end - start, 1))
                        * float(instance["sigma"])
                    )
                    prediction = model.predict_clean(
                        patches,
                        sigma=sigma,
                    )
                else:
                    prediction = model.predict_clean(patches)
            elif mode == "heun":
                if not getattr(model, "use_edm", False):
                    raise ValueError("heun mode requires an EDM checkpoint")
                prediction = model.edm_heun_sampler(patches)
            else:
                raise ValueError(f"unsupported stage1 mode: {mode}")
            predictions.append(
                prediction.numpy().astype(np.float32, copy=False)
            )
    return np.concatenate(predictions, axis=0)


def fusion_weights(instance, fusion_tau):
    return np.exp(
        -float(fusion_tau) * instance["normalized_distances"]
    ).astype(np.float64)


def fuse_absolute(instance, absolute_prediction, fusion_tau):
    point_count = instance["noisy"].shape[0]
    weights = fusion_weights(instance, fusion_tau)
    weighted_sum = np.zeros((point_count, 3), dtype=np.float64)
    weight_sum = np.zeros((point_count,), dtype=np.float64)
    for patch_index, indices in enumerate(instance["point_indices"]):
        patch_weights = weights[patch_index]
        np.add.at(
            weighted_sum,
            indices,
            absolute_prediction[patch_index]
            * patch_weights[:, None],
        )
        np.add.at(weight_sum, indices, patch_weights)
    output = instance["noisy"].copy()
    covered = weight_sum > 0
    output[covered] = (
        weighted_sum[covered] / weight_sum[covered, None]
    ).astype(np.float32)
    return output


def clip_residual(residual, budget):
    norm = np.sqrt((residual**2.0).sum(axis=-1, keepdims=True))
    scale = np.minimum(1.0, float(budget) / np.maximum(norm, 1e-12))
    return (residual * scale).astype(np.float32, copy=False)


def smooth_patch_residual(
    absolute_prediction,
    residual,
    k,
    alpha,
):
    if k <= 1 or alpha <= 0:
        return residual
    smoothed = np.empty_like(residual)
    for patch_index in range(absolute_prediction.shape[0]):
        points = absolute_prediction[patch_index]
        neighbor_count = min(int(k), points.shape[0])
        _, neighbors = cKDTree(points).query(
            points,
            k=neighbor_count,
        )
        neighbors = np.asarray(neighbors, dtype=np.int64)
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        local_mean = residual[patch_index][neighbors].mean(axis=1)
        smoothed[patch_index] = (
            (1.0 - float(alpha)) * residual[patch_index]
            + float(alpha) * local_mean
        )
    return smoothed


def overlap_rms(instance, absolute_prediction, fused):
    squared = []
    for patch_index, indices in enumerate(instance["point_indices"]):
        delta = absolute_prediction[patch_index] - fused[indices]
        squared.append((delta**2.0).sum(axis=-1))
    return float(np.sqrt(np.concatenate(squared).mean()))


def add_score(row, prefix, score, baseline):
    for key in ("cd_score", "p2s_score", "final_score"):
        row[f"{prefix}_{key}"] = score[key]
    row[f"{prefix}_final_gain"] = (
        score["final_score"] - baseline["final_score"]
    )


def evaluate_instance(
    model,
    instance,
    budgets,
    batch_size,
    stage1_mode,
    fusion_tau,
    smooth_k,
    smooth_alpha,
):
    patch_prediction = predict_stage1(
        model,
        instance,
        batch_size=batch_size,
        mode=stage1_mode,
    )
    absolute_prediction = (
        patch_prediction + instance["seeds"][:, None, :]
    )
    baseline_prediction = fuse_absolute(
        instance,
        absolute_prediction,
        fusion_tau=fusion_tau,
    )
    baseline = score_instance(instance, baseline_prediction)
    clean_patch = instance["clean"][instance["point_indices"]]
    raw_residual = clean_patch - absolute_prediction
    smooth_residual = smooth_patch_residual(
        absolute_prediction,
        raw_residual,
        k=smooth_k,
        alpha=smooth_alpha,
    )
    paired_norm = np.sqrt(
        ((instance["clean"] - baseline_prediction) ** 2.0).sum(axis=-1)
    )
    row = {
        "rel_path": instance["rel_path"],
        "sigma": float(instance["sigma"]),
        "patch_count": int(instance["patches"].shape[0]),
        "baseline_cd_score": baseline["cd_score"],
        "baseline_p2s_score": baseline["p2s_score"],
        "baseline_final_score": baseline["final_score"],
        "baseline_overlap_rms": overlap_rms(
            instance,
            absolute_prediction,
            baseline_prediction,
        ),
        "remaining_residual_mean": float(paired_norm.mean()),
        "remaining_residual_p50": float(np.percentile(paired_norm, 50)),
        "remaining_residual_p90": float(np.percentile(paired_norm, 90)),
        "remaining_residual_p99": float(np.percentile(paired_norm, 99)),
    }
    for budget in budgets:
        key = budget_key(budget)
        global_prediction = (
            baseline_prediction
            + clip_residual(
                instance["clean"] - baseline_prediction,
                budget,
            )
        )
        patch_absolute = (
            absolute_prediction + clip_residual(raw_residual, budget)
        )
        patch_prediction_fused = fuse_absolute(
            instance,
            patch_absolute,
            fusion_tau=fusion_tau,
        )
        smooth_absolute = (
            absolute_prediction + clip_residual(smooth_residual, budget)
        )
        smooth_prediction_fused = fuse_absolute(
            instance,
            smooth_absolute,
            fusion_tau=fusion_tau,
        )
        global_score = score_instance(instance, global_prediction)
        patch_score = score_instance(instance, patch_prediction_fused)
        smooth_score = score_instance(instance, smooth_prediction_fused)
        add_score(row, f"global_{key}", global_score, baseline)
        add_score(row, f"patch_{key}", patch_score, baseline)
        add_score(row, f"smooth_{key}", smooth_score, baseline)
        global_gain = global_score["final_score"] - baseline["final_score"]
        smooth_gain = smooth_score["final_score"] - baseline["final_score"]
        row[f"smooth_{key}_fusion_efficiency"] = (
            smooth_gain / global_gain if global_gain > 1e-8 else 0.0
        )
        row[f"smooth_{key}_overlap_rms"] = overlap_rms(
            instance,
            smooth_absolute,
            smooth_prediction_fused,
        )
        row[f"residual_within_{key}_rate"] = float(
            np.mean(paired_norm <= float(budget))
        )
    return row


def summarize(rows, budgets):
    summary = {
        "shape_count": len(rows),
        "baseline": {
            key: float(np.mean([row[f"baseline_{key}"] for row in rows]))
            for key in ("cd_score", "p2s_score", "final_score")
        },
        "remaining_residual": {
            key: float(
                np.mean([row[f"remaining_residual_{key}"] for row in rows])
            )
            for key in ("mean", "p50", "p90", "p99")
        },
        "baseline_overlap_rms": float(
            np.mean([row["baseline_overlap_rms"] for row in rows])
        ),
        "budgets": {},
    }
    for budget in budgets:
        key = budget_key(budget)
        budget_summary = {
            "budget": float(budget),
            "residual_within_rate": float(
                np.mean([row[f"residual_within_{key}_rate"] for row in rows])
            ),
        }
        for method in ("global", "patch", "smooth"):
            budget_summary[method] = {
                metric: float(
                    np.mean(
                        [
                            row[f"{method}_{key}_{metric}"]
                            for row in rows
                        ]
                    )
                )
                for metric in (
                    "cd_score",
                    "p2s_score",
                    "final_score",
                    "final_gain",
                )
            }
        budget_summary["smooth_fusion_efficiency"] = float(
            np.mean(
                [
                    row[f"smooth_{key}_fusion_efficiency"]
                    for row in rows
                ]
            )
        )
        budget_summary["smooth_overlap_rms"] = float(
            np.mean([row[f"smooth_{key}_overlap_rms"] for row in rows])
        )
        summary["budgets"][key] = budget_summary
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument(
        "--transform-config",
        default=str(DEFAULT_TRANSFORM_CONFIG),
    )
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument(
        "--category-reference-list",
        default="datalist/test.txt",
    )
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument(
        "--out-dir",
        default="outputs/fusion_aware_residual_oracle_v1",
    )
    parser.add_argument("--max-shapes", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument(
        "--residual-budgets",
        default="0.002,0.004,0.006,0.008",
    )
    parser.add_argument("--smooth-k", type=int, default=24)
    parser.add_argument("--smooth-alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--allow-nonstandard-protocol", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    standard_protocol = (
        args.num_points == 32768
        and args.patch_size == 1000
        and args.seed_k == 6.0
    )
    if not standard_protocol and not args.allow_nonstandard_protocol:
        raise ValueError(
            "oracle probe must use 32768 points / 1000-point patches / "
            "seed_k=6 unless --allow-nonstandard-protocol is set"
        )
    if not 0.0 <= args.smooth_alpha <= 1.0:
        raise ValueError("smooth_alpha must be in [0, 1]")

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    budgets = parse_budgets(args.residual_budgets)
    bands = parse_noise_bands(args.noise_bands)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = usable_paths(
        read_datalist(args.datalist),
        clean_root=args.clean_root,
        mesh_root=args.mesh_root,
        sample_missing_clean=args.sample_missing_clean,
    )
    paths = choose_balanced_paths(
        candidates,
        args.max_shapes,
        read_datalist(args.category_reference_list),
        rng,
    )
    if not paths:
        raise FileNotFoundError(
            "no usable shapes; generate cache_clean_points or pass "
            "--sample-missing-clean"
        )

    checkpoint = Path(args.checkpoint)
    model = load_model(
        checkpoint,
        model_config=args.model_config,
        transform_config=args.transform_config,
    )
    compatibility = validate_checkpoint_compatibility(model, checkpoint)
    print(
        "Checkpoint compatibility: exact match "
        f"({compatibility['model_parameter_count']} parameters)",
        flush=True,
    )

    rows = []
    for index, rel_path in enumerate(paths):
        shape = load_shape(
            rel_path,
            clean_root=args.clean_root,
            mesh_root=args.mesh_root,
            num_points=args.num_points,
            rng=rng,
            sample_missing_clean=args.sample_missing_clean,
        )
        lower, upper = bands[index % len(bands)]
        sigma = float(rng.uniform(lower, upper))
        instance = make_instance(
            shape,
            sigma=sigma,
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            noise_type=args.noise_type,
            rng=rng,
        )
        instance["noise_band"] = (lower, upper)
        row = evaluate_instance(
            model,
            instance,
            budgets=budgets,
            batch_size=args.batch_size,
            stage1_mode=args.stage1_mode,
            fusion_tau=args.fusion_tau,
            smooth_k=args.smooth_k,
            smooth_alpha=args.smooth_alpha,
        )
        rows.append(row)
        write_csv(out_dir / "rows.csv", rows)
        write_json(out_dir / "summary.json", summarize(rows, budgets))
        print(
            f"[{index + 1}/{len(paths)}] {rel_path} "
            f"sigma={sigma:.4f} "
            f"baseline={row['baseline_final_score']:.3f}",
            flush=True,
        )

    summary = summarize(rows, budgets)
    summary["args"] = vars(args)
    summary["checkpoint_compatibility"] = compatibility
    summary["paths"] = paths
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
