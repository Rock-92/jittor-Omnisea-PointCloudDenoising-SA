import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jittor as jt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.check_patch_score_distribution_outputs62 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    chamfer_parts,
    evaluate_patch,
    load_model,
    metric_to_score,
    read_datalist,
    sample_patch,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_result/outputs_analysis/low_patch_modulation_outputs6.2"


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def split_gate(gate):
    num_blocks = gate.shape[-1] // 6
    blocks = []
    for block_idx in range(num_blocks):
        block = gate[:, block_idx * 6 : (block_idx + 1) * 6]
        blocks.append({"scale": block[:, :3], "temperature": block[:, 3:]})
    return blocks


def gate_metrics(model, patch):
    x = jt.array(patch["patch_noisy"][None, :, :].astype(np.float32, copy=False))
    with jt.no_grad():
        _, geometry, gate = model.encoder(x, return_condition=True)
    geometry = geometry.detach().numpy()[0].astype(np.float64, copy=False)
    gate = gate.detach().numpy()[0].astype(np.float64, copy=False)
    row = {
        "geometry_point_std_mean": float(geometry.std(axis=0).mean()),
        "geometry_mean_l2": float(np.sqrt((geometry.mean(axis=0) ** 2).sum())),
        "gate_point_std_mean": float(gate.std(axis=0).mean()),
        "gate_mean_l2": float(np.sqrt((gate.mean(axis=0) ** 2).sum())),
    }
    for block_idx, block in enumerate(split_gate(gate)):
        scale = block["scale"]
        temperature = block["temperature"]
        row[f"block_{block_idx}_scale_abs_from_1"] = float(np.abs(scale - 1.0).mean())
        row[f"block_{block_idx}_temperature_abs_from_1"] = float(
            np.abs(temperature - 1.0).mean()
        )
        row[f"block_{block_idx}_scale_point_std"] = float(scale.std(axis=0).mean())
        row[f"block_{block_idx}_temperature_point_std"] = float(
            temperature.std(axis=0).mean()
        )
    row["scale_abs_from_1_mean"] = float(
        np.mean([row[f"block_{i}_scale_abs_from_1"] for i in range(len(split_gate(gate)))])
    )
    row["temperature_abs_from_1_mean"] = float(
        np.mean([row[f"block_{i}_temperature_abs_from_1"] for i in range(len(split_gate(gate)))])
    )
    row["modulation_strength"] = row["scale_abs_from_1_mean"] + row[
        "temperature_abs_from_1_mean"
    ]
    return row


def zero_geometry_modulation(model):
    for block in model.encoder.blocks:
        block._zero_linear(block.scale_gate_proj)
        block._zero_linear(block.temperature_proj)
    model.eval()
    return model


def evaluate_patch_score_only(model, patch):
    pc_noisy = jt.array(patch["patch_noisy"][None, :, :])
    with jt.no_grad():
        pc_pred, _ = model.denoise_langevin_dynamics(pc_noisy)
    pred = pc_pred.detach().numpy()[0].astype(np.float32, copy=False)
    clean = patch["patch_clean"]
    noisy = patch["patch_noisy"]
    noisy_p2c, noisy_c2p = chamfer_parts(noisy, clean)
    pred_p2c, pred_c2p = chamfer_parts(pred, clean)
    cd_noisy = noisy_p2c + noisy_c2p
    cd_pred = pred_p2c + pred_c2p
    return {
        "ablate_cd_pred": cd_pred,
        "ablate_cd_score": metric_to_score(cd_pred, cd_noisy),
        "ablate_cd_delta": cd_pred - cd_noisy,
    }


def group_summary(rows, mask, feature_keys):
    selected = [row for row, keep in zip(rows, mask) if keep]
    out = {"count": len(selected)}
    for key in feature_keys:
        out[key] = summarize_values([row[key] for row in selected])
    out["category_counts"] = dict(Counter(row["geometry_category"] for row in selected))
    return out


def compare_feature_means(rows, low_mask, feature_keys):
    low_rows = [row for row, keep in zip(rows, low_mask) if keep]
    other_rows = [row for row, keep in zip(rows, low_mask) if not keep]
    comparisons = []
    for key in feature_keys:
        low_vals = np.asarray([row[key] for row in low_rows], dtype=np.float64)
        other_vals = np.asarray([row[key] for row in other_rows], dtype=np.float64)
        comparisons.append(
            {
                "feature": key,
                "low_mean": float(np.nanmean(low_vals)),
                "other_mean": float(np.nanmean(other_vals)),
                "low_median": float(np.nanmedian(low_vals)),
                "other_median": float(np.nanmedian(other_vals)),
                "mean_diff_low_minus_other": float(
                    np.nanmean(low_vals) - np.nanmean(other_vals)
                ),
            }
        )
    comparisons.sort(key=lambda row: abs(row["mean_diff_low_minus_other"]), reverse=True)
    return comparisons


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_report(summary, out_path):
    lines = [
        "# Low Patch Modulation Analysis: outputs6.2",
        "",
        f"Low score definition: bottom 30%, threshold `{summary['low_score_threshold']:.4f}`",
        "",
        "## Conclusion",
        "",
        f"- commonality: `{summary['commonality_conclusion']}`",
        f"- modulation effect on low patches: `{summary['modulation_conclusion']}`",
        "",
        "## Low vs Other",
        "",
        f"- low count: `{summary['low_group']['count']}`",
        f"- other count: `{summary['other_group']['count']}`",
        f"- low category counts: `{summary['low_group']['category_counts']}`",
        "",
        "## Strongest Feature Differences",
        "",
        "| Feature | Low mean | Other mean | Low median | Other median | Diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["feature_comparisons"][:12]:
        lines.append(
            "| {feature} | {low_mean:.4f} | {other_mean:.4f} | {low_median:.4f} | "
            "{other_median:.4f} | {mean_diff_low_minus_other:.4f} |".format(**row)
        )
    lines += [
        "",
        "## Modulation",
        "",
        f"- normal low mean score: `{summary['ablation']['low_normal_score_mean']:.4f}`",
        f"- ablated low mean score: `{summary['ablation']['low_ablate_score_mean']:.4f}`",
        f"- low normal minus ablated: `{summary['ablation']['low_score_gain_from_modulation']:.4f}`",
        f"- low modulation strength mean: `{summary['ablation']['low_modulation_strength_mean']:.4f}`",
        f"- other modulation strength mean: `{summary['ablation']['other_modulation_strength_mean']:.4f}`",
        "",
        "## Correlations",
        "",
    ]
    for key, value in summary["correlations"].items():
        lines.append(f"- {key}: `{value:.4f}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mesh-root", default=str(PROJECT_ROOT / "dataset_clean"))
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--low-quantile", type=float, default=0.30)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    mesh_root = Path(args.mesh_root)
    usable = [
        rel for rel in read_datalist(PROJECT_ROOT / args.datalist)
        if (mesh_root / rel / "models/model_normalized.obj").exists()
    ]
    chosen = [usable[int(rng.integers(0, len(usable)))] for _ in range(args.candidates)]

    model = load_model(checkpoint)
    ablated_model = zero_geometry_modulation(load_model(checkpoint))

    rows = []
    for idx, rel_path in enumerate(chosen, start=1):
        print(f"[{idx}/{len(chosen)}] {rel_path}", flush=True)
        patch = sample_patch(rel_path, mesh_root, rng, args.patch_size)
        row, _ = evaluate_patch(model, patch, args.seed + idx)
        row["index"] = idx
        row.update(gate_metrics(model, patch))
        row.update(evaluate_patch_score_only(ablated_model, patch))
        row["score_gain_from_modulation"] = row["cd_score"] - row["ablate_cd_score"]
        rows.append(row)

    scores = np.asarray([row["cd_score"] for row in rows], dtype=np.float64)
    low_threshold = float(np.quantile(scores, args.low_quantile))
    low_mask = scores <= low_threshold

    feature_keys = [
        "cd_score",
        "noise_std",
        "local_curv_p90",
        "local_curv_mean",
        "normal_var",
        "curvature",
        "linearity",
        "planarity",
        "scattering",
        "bbox_ratio_min",
        "bbox_ratio_mid",
        "geometry_point_std_mean",
        "gate_point_std_mean",
        "scale_abs_from_1_mean",
        "temperature_abs_from_1_mean",
        "modulation_strength",
        "score_gain_from_modulation",
    ]
    low_rows = [row for row, keep in zip(rows, low_mask) if keep]
    other_rows = [row for row, keep in zip(rows, low_mask) if not keep]
    low_gain = np.asarray([row["score_gain_from_modulation"] for row in low_rows])
    other_gain = np.asarray([row["score_gain_from_modulation"] for row in other_rows])
    low_mod = np.asarray([row["modulation_strength"] for row in low_rows])
    other_mod = np.asarray([row["modulation_strength"] for row in other_rows])

    correlations = {}
    for key in [
        "local_curv_p90",
        "local_curv_mean",
        "normal_var",
        "noise_std",
        "modulation_strength",
        "scale_abs_from_1_mean",
        "temperature_abs_from_1_mean",
        "score_gain_from_modulation",
    ]:
        vals = [row[key] for row in rows]
        correlations[f"pearson_score_vs_{key}"] = pearson(scores, vals)
        correlations[f"spearman_score_vs_{key}"] = spearman(scores, vals)

    commonity = "LOW_PATCHES_ARE_CURVATURE_NORMAL_VAR_HEAVY"
    if Counter(row["geometry_category"] for row in low_rows).most_common(1)[0][1] < 0.6 * len(low_rows):
        commonity = "LOW_PATCHES_HAVE_WEAK_CATEGORY_CONCENTRATION"
    modulation = "MODULATION_ACTIVE_BUT_NOT_RESCUING_LOW_PATCHES"
    if float(np.nanmean(low_gain)) > 1.0:
        modulation = "MODULATION_IMPROVES_LOW_PATCHES"
    elif float(np.nanmean(low_gain)) < -1.0:
        modulation = "MODULATION_HURTS_LOW_PATCHES"

    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "seed": args.seed,
        "patch_size": args.patch_size,
        "candidates": args.candidates,
        "low_quantile": args.low_quantile,
        "low_score_threshold": low_threshold,
        "commonality_conclusion": commonity,
        "modulation_conclusion": modulation,
        "low_group": group_summary(rows, low_mask, feature_keys),
        "other_group": group_summary(rows, ~low_mask, feature_keys),
        "feature_comparisons": compare_feature_means(rows, low_mask, feature_keys),
        "ablation": {
            "low_normal_score_mean": float(np.nanmean([row["cd_score"] for row in low_rows])),
            "low_ablate_score_mean": float(np.nanmean([row["ablate_cd_score"] for row in low_rows])),
            "low_score_gain_from_modulation": float(np.nanmean(low_gain)),
            "low_score_gain_from_modulation_median": float(np.nanmedian(low_gain)),
            "other_score_gain_from_modulation": float(np.nanmean(other_gain)),
            "other_score_gain_from_modulation_median": float(np.nanmedian(other_gain)),
            "low_modulation_strength_mean": float(np.nanmean(low_mod)),
            "other_modulation_strength_mean": float(np.nanmean(other_mod)),
        },
        "correlations": correlations,
    }

    write_csv(out_dir / "low_patch_modulation_records.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_report(summary, out_dir / "report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
