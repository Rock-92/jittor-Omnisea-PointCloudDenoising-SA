import argparse
import csv
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.refinement import MultiStageGeometryRefiner  # noqa: E402
from tools.hard_patch_common import (  # noqa: E402
    displacement_metrics,
    load_model,
    quantile_summary,
    score_prediction,
)
from tools.train_refinement_probe import generate_coarse  # noqa: E402


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_patch_file(path):
    data = np.load(path, allow_pickle=True)
    return {
        "pc_noisy": data["pc_noisy"].astype(np.float32, copy=False),
        "pc_clean": data["pc_clean"].astype(np.float32, copy=False),
        "score_sigma": data["score_sigma"].astype(np.float32, copy=False),
        "rel_path": data["rel_path"],
        "seed_idx": data["seed_idx"],
        **{
            key: data[key]
            for key in ["patch_center", "normalize_center", "normalize_scale"]
            if key in data.files
        },
    }


def pairwise_sqdist(first, second):
    return (
        (first.unsqueeze(2) - second.unsqueeze(1)) ** 2.0
    ).sum(dim=-1)


def sampled_chamfer(prediction, clean, max_points):
    point_count = prediction.shape[1]
    sample_count = min(int(max_points), point_count)
    if sample_count < point_count:
        indices = np.linspace(
            0,
            point_count - 1,
            sample_count,
            dtype=np.int32,
        )
        prediction = prediction[:, indices, :]
        clean = clean[:, indices, :]
    distance = pairwise_sqdist(prediction, clean)
    return (
        distance.min(dim=2).mean()
        + distance.min(dim=1).mean()
    )


def stage_loss(
    previous,
    prediction,
    clean,
    aux,
    loss_scale,
    chamfer_points,
    chamfer_weight,
    direction_weight,
    length_weight,
    keep_weight,
    keep_threshold,
):
    scale = max(float(loss_scale), 1e-6)
    scale2 = scale ** 2.0
    target = clean - previous
    target_length = jt.sqrt(
        (target ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
    )
    residual = aux["residual"]
    residual_length = jt.sqrt(
        (residual ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
    )

    paired = ((prediction - clean) ** 2.0).sum(dim=-1).mean() / scale2
    chamfer = sampled_chamfer(
        prediction,
        clean,
        chamfer_points,
    ) / scale2
    cosine = (residual * target).sum(dim=-1, keepdims=True) / (
        residual_length * target_length + 1e-8
    )
    direction_weight_map = jt.minimum(
        target_length / scale,
        jt.ones_like(target_length),
    )
    direction = (
        (1.0 - cosine) * direction_weight_map
    ).sum() / (direction_weight_map.sum() + 1e-6)
    length = (
        jt.abs(residual_length - target_length) / scale
    ).mean()
    keep_map = jt.exp(
        -target_length / max(float(keep_threshold), 1e-6)
    )
    keep = (
        keep_map * (residual_length / scale) ** 2.0
    ).sum() / (keep_map.sum() + 1e-6)

    total = (
        paired
        + float(chamfer_weight) * chamfer
        + float(direction_weight) * direction
        + float(length_weight) * length
        + float(keep_weight) * keep
    )
    return total, {
        "paired": paired,
        "chamfer": chamfer,
        "direction": direction,
        "length": length,
        "keep": keep,
    }


def refinement_loss(model, coarse, noisy, clean, args):
    prediction, output = model(coarse, noisy)
    previous = coarse
    total = jt.zeros(())
    metrics = {}
    stage_count = len(output["stages"])
    for stage_index, stage in enumerate(output["stages"]):
        stage_total, parts = stage_loss(
            previous=previous,
            prediction=stage["prediction"],
            clean=clean,
            aux=stage,
            loss_scale=model.stage_max_residuals[stage_index],
            chamfer_points=args.chamfer_points,
            chamfer_weight=args.chamfer_weight,
            direction_weight=args.direction_weight,
            length_weight=args.length_weight,
            keep_weight=args.keep_weight,
            keep_threshold=args.keep_threshold,
        )
        deep_weight = 1.0 if stage_index == stage_count - 1 else 0.5
        total = total + deep_weight * stage_total
        for key, value in parts.items():
            metrics[f"stage{stage_index + 1}_{key}"] = value
        metrics[f"stage{stage_index + 1}_confidence"] = stage[
            "confidence"
        ].mean()
        metrics[f"stage{stage_index + 1}_residual"] = jt.sqrt(
            (stage["residual"] ** 2.0).sum(dim=-1) + 1e-8
        ).mean()
        previous = stage["prediction"]
    return total, prediction, metrics


def predict(model, coarse_np, noisy_np, batch_size):
    model.eval()
    outputs = []
    with jt.no_grad():
        for start in range(0, coarse_np.shape[0], batch_size):
            end = min(start + batch_size, coarse_np.shape[0])
            prediction, _ = model(
                jt.array(coarse_np[start:end]),
                jt.array(noisy_np[start:end]),
            )
            outputs.append(prediction.numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def score_rows(noisy_np, clean_np, coarse_np, prediction_np, paths):
    rows = []
    for index in range(noisy_np.shape[0]):
        coarse_score = score_prediction(
            noisy_np[index],
            clean_np[index],
            coarse_np[index],
        )
        refined_score = score_prediction(
            noisy_np[index],
            clean_np[index],
            prediction_np[index],
        )
        rows.append(
            {
                "index": index,
                "rel_path": str(paths[index]),
                "coarse_score": coarse_score["cd_score"],
                "refined_score": refined_score["cd_score"],
                "score_gain": (
                    refined_score["cd_score"] - coarse_score["cd_score"]
                ),
                **{
                    f"refined_{key}": value
                    for key, value in displacement_metrics(
                        noisy_np[index],
                        clean_np[index],
                        prediction_np[index],
                    ).items()
                },
            }
        )
    return rows


def add_score_bands(rows, thresholds=None):
    scores = np.asarray(
        [row["coarse_score"] for row in rows],
        dtype=np.float64,
    )
    if thresholds is None:
        thresholds = np.quantile(scores, [1 / 3, 2 / 3])
    for row in rows:
        if row["coarse_score"] <= thresholds[0]:
            row["score_band"] = "hard"
        elif row["coarse_score"] <= thresholds[1]:
            row["score_band"] = "ordinary"
        else:
            row["score_band"] = "high_score"
    return np.asarray(thresholds, dtype=np.float64)


def summarize_rows(rows):
    gains = np.asarray([row["score_gain"] for row in rows])
    return {
        "count": len(rows),
        "coarse_score_mean": float(
            np.mean([row["coarse_score"] for row in rows])
        ),
        "refined_score_mean": float(
            np.mean([row["refined_score"] for row in rows])
        ),
        "score_gain": quantile_summary(gains),
        "improved_rate": float(np.mean(gains > 0.0)),
        "degraded_ge_1_rate": float(np.mean(gains <= -1.0)),
    }


def full_summary(rows):
    result = {"overall": summarize_rows(rows), "by_score_band": {}}
    for band in ["hard", "ordinary", "high_score"]:
        selected = [row for row in rows if row["score_band"] == band]
        result["by_score_band"][band] = summarize_rows(selected)
    return result


def cache_coarse(checkpoint, data, cache_path, batch_size, coarse_mode):
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["pc_coarse"].astype(np.float32, copy=False)
    model = load_model(checkpoint)
    for parameter in model.parameters():
        parameter.stop_grad()
    coarse = generate_coarse(
        model,
        data["pc_noisy"],
        batch_size,
        coarse_mode,
        data["score_sigma"],
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, pc_coarse=coarse)
    return coarse


def balanced_epoch_indices(rows, rng, band_weights):
    probabilities = np.asarray(
        [
            band_weights[row["score_band"]]
            / sum(item["score_band"] == row["score_band"] for item in rows)
            for row in rows
        ],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum()
    return rng.choice(
        np.arange(len(rows)),
        size=len(rows),
        replace=True,
        p=probabilities,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs_result/outputs_hardware/"
            "checkpoints/vm_ssl/checkpoint_best.pkl"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=(
            "outputs_result/outputs_analysis/"
            "multistage_refinement_probe_dataset"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs_result/outputs_analysis/"
            "multistage_refinement_probe"
        ),
    )
    parser.add_argument("--stages", type=int, choices=[1, 2], default=2)
    parser.add_argument("--load-refiner", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="fixed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-max-residual", type=float, default=0.012)
    parser.add_argument("--stage2-max-residual", type=float, default=0.008)
    parser.add_argument("--chamfer-points", type=int, default=128)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--direction-weight", type=float, default=0.2)
    parser.add_argument("--length-weight", type=float, default=0.1)
    parser.add_argument("--keep-weight", type=float, default=0.2)
    parser.add_argument("--keep-threshold", type=float, default=0.003)
    parser.add_argument("--hard-sample-weight", type=float, default=0.25)
    parser.add_argument("--ordinary-sample-weight", type=float, default=0.60)
    parser.add_argument("--high-score-sample-weight", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_patch_file(dataset_dir / "train_patches.npz")
    val = load_patch_file(dataset_dir / "val_patches.npz")
    train_coarse = cache_coarse(
        args.checkpoint,
        train,
        dataset_dir / "train_coarse.npz",
        args.batch_size,
        args.coarse_mode,
    )
    val_coarse = cache_coarse(
        args.checkpoint,
        val,
        dataset_dir / "val_coarse.npz",
        args.batch_size,
        args.coarse_mode,
    )

    identity_train_rows = score_rows(
        train["pc_noisy"],
        train["pc_clean"],
        train_coarse,
        train_coarse,
        train["rel_path"],
    )
    train_thresholds = add_score_bands(identity_train_rows)
    identity_val_rows = score_rows(
        val["pc_noisy"],
        val["pc_clean"],
        val_coarse,
        val_coarse,
        val["rel_path"],
    )
    val_thresholds = add_score_bands(identity_val_rows)

    model = MultiStageGeometryRefiner(
        num_stages=args.stages,
        stage_max_residuals=(
            args.stage1_max_residual,
            args.stage2_max_residual,
        ),
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
    )
    if args.load_refiner:
        model.load(args.load_refiner)
        print(f"Loaded refiner: {args.load_refiner}", flush=True)
    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_score = full_summary(identity_val_rows)["overall"][
        "coarse_score_mean"
    ]
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        epoch_indices = balanced_epoch_indices(
            identity_train_rows,
            rng,
            {
                "hard": args.hard_sample_weight,
                "ordinary": args.ordinary_sample_weight,
                "high_score": args.high_score_sample_weight,
            },
        )
        batch_metrics = []
        for start in range(0, len(epoch_indices), args.batch_size):
            indices = epoch_indices[start:start + args.batch_size]
            loss, _, metrics = refinement_loss(
                model,
                jt.array(train_coarse[indices]),
                jt.array(train["pc_noisy"][indices]),
                jt.array(train["pc_clean"][indices]),
                args,
            )
            optimizer.step(loss)
            row = {"loss": float(loss.item())}
            row.update(
                {key: float(value.item()) for key, value in metrics.items()}
            )
            batch_metrics.append(row)

        if (
            epoch != 0
            and (epoch + 1) % args.eval_every != 0
            and epoch != args.epochs - 1
        ):
            continue

        prediction = predict(
            model,
            val_coarse,
            val["pc_noisy"],
            args.batch_size,
        )
        rows = score_rows(
            val["pc_noisy"],
            val["pc_clean"],
            val_coarse,
            prediction,
            val["rel_path"],
        )
        add_score_bands(rows, val_thresholds)
        summary = full_summary(rows)
        metric_keys = batch_metrics[0].keys()
        record = {
            "epoch": epoch,
            **{
                key: float(np.mean([item[key] for item in batch_metrics]))
                for key in metric_keys
            },
            "val_score": summary["overall"]["refined_score_mean"],
            "val_gain": summary["overall"]["score_gain"]["mean"],
            "hard_gain": summary["by_score_band"]["hard"]["score_gain"]["mean"],
            "ordinary_gain": summary["by_score_band"]["ordinary"][
                "score_gain"
            ]["mean"],
            "high_score_gain": summary["by_score_band"]["high_score"][
                "score_gain"
            ]["mean"],
            "improved_rate": summary["overall"]["improved_rate"],
        }
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        print(record, flush=True)

        if record["val_score"] >= best_score:
            best_score = record["val_score"]
            best_epoch = epoch
            model.save(str(out_dir / "refiner_best.pkl"))
            write_csv(out_dir / "best_val_eval.csv", rows)
            (out_dir / "best_val_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

    model.save(str(out_dir / "refiner_last.pkl"))
    final = {
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "coarse_val_score": full_summary(identity_val_rows)["overall"][
            "coarse_score_mean"
        ],
        "train_score_band_thresholds": train_thresholds.tolist(),
        "val_score_band_thresholds": val_thresholds.tolist(),
        "train_patches": len(identity_train_rows),
        "val_patches": len(identity_val_rows),
        "train_shapes": len(set(train["rel_path"].tolist())),
        "val_shapes": len(set(val["rel_path"].tolist())),
        "shape_overlap": len(
            set(train["rel_path"].tolist()) & set(val["rel_path"].tolist())
        ),
        "args": vars(args),
    }
    (out_dir / "probe_summary.json").write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
