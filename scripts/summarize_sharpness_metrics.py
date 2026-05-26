import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key, value in row.items():
                if key != "rel_path":
                    parsed[key] = parse_float(value)
            rows.append(parsed)
    return rows


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def bucket_stats(rows, key, num_buckets=4):
    values = np.asarray([r[key] for r in rows], dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return []
    quantiles = np.quantile(finite_values, np.linspace(0, 1, num_buckets + 1))
    stats = []
    for i in range(num_buckets):
        lo = quantiles[i]
        hi = quantiles[i + 1]
        if i == num_buckets - 1:
            bucket = [r for r in rows if np.isfinite(r[key]) and lo <= r[key] <= hi]
        else:
            bucket = [r for r in rows if np.isfinite(r[key]) and lo <= r[key] < hi]
        scores = [r["cd_score"] for r in bucket]
        deltas = [r["cd_delta"] for r in bucket]
        stats.append(
            {
                "bucket": i + 1,
                "sharp_min": float(lo),
                "sharp_max": float(hi),
                "count": len(bucket),
                "mean_score": float(np.mean(scores)) if scores else math.nan,
                "median_score": float(np.median(scores)) if scores else math.nan,
                "worse_rate": float(np.mean([d > 0 for d in deltas])) if deltas else math.nan,
                "mean_cd_delta": float(np.mean(deltas)) if deltas else math.nan,
            }
        )
    return stats


def sharpness_summary(rows, key):
    sharp = [r[key] for r in rows]
    scores = [r["cd_score"] for r in rows]
    deltas = [r["cd_delta"] for r in rows]
    return {
        "key": key,
        "score_pearson": pearson(sharp, scores),
        "score_spearman": spearman(sharp, scores),
        "cd_delta_pearson": pearson(sharp, deltas),
        "cd_delta_spearman": spearman(sharp, deltas),
        "buckets": bucket_stats(rows, key),
    }


def write_bucket_csv(stats_by_key, path):
    fieldnames = [
        "sharpness_key",
        "bucket",
        "sharp_min",
        "sharp_max",
        "count",
        "mean_score",
        "median_score",
        "worse_rate",
        "mean_cd_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, summary in stats_by_key.items():
            for row in summary["buckets"]:
                writer.writerow({"sharpness_key": key, **row})


def save_plots(rows, stats, out_dir, key, label, prefix):
    sharp = np.asarray([r[key] for r in rows], dtype=np.float64)
    scores = np.asarray([r["cd_score"] for r in rows], dtype=np.float64)
    deltas = np.asarray([r["cd_delta"] for r in rows], dtype=np.float64)
    mask = np.isfinite(sharp) & np.isfinite(scores) & np.isfinite(deltas)
    sharp = sharp[mask]
    scores = scores[mask]
    deltas = deltas[mask]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    colors = np.where(deltas > 0, "#d62728", "#1f77b4")
    ax.scatter(sharp, scores, s=22, c=colors, alpha=0.72)
    ax.set_xlabel(label)
    ax.set_ylabel("CD improvement score")
    ax.set_title(f"{label} vs denoising score")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_vs_score.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))
    labels = [f"Q{s['bucket']}" for s in stats]
    med = [s["median_score"] for s in stats]
    worse = [100.0 * s["worse_rate"] for s in stats]
    x = np.arange(len(labels))
    ax1.bar(x - 0.18, med, width=0.36, color="#4e79a7", label="median score")
    ax1.set_ylabel("Median CD score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, worse, width=0.36, color="#e15759", label="worse rate")
    ax2.set_ylabel("Worse than noisy (%)")
    ax1.set_title(f"Performance by {label} quartile")
    ax1.grid(True, axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_bucket_summary.png", dpi=180)
    plt.close(fig)


def load_existing_summary(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_report(summary, path):
    q = summary["buckets"]
    lines = [
        "# Sharpness diagnostic report",
        "",
        f"Checkpoint: `{summary.get('checkpoint', '')}`",
        "",
        f"Dataset root used by the run: `{summary.get('mesh_root', '')}`",
        "",
        (
            f"Sample: {summary['candidates']} validation patches, patch size "
            f"{summary['patch_size']}, seed {summary['seed']}."
        ),
        "",
        "## Main finding",
        "",
        (
            "The current evidence only weakly supports the idea that the trained "
            "`best_checkpoint` performs worse on sharper, edge-like patches."
        ),
        "",
        "Using point-normal variation as the primary sharpness measure:",
        "",
        f"- Pearson(score, sharpness): {summary['score_pearson_vs_sharpness']:.4f}",
        f"- Spearman(score, sharpness): {summary['score_spearman_vs_sharpness']:.4f}",
        f"- Pearson(CD delta, sharpness): {summary['cd_delta_pearson_vs_sharpness']:.4f}",
        f"- Spearman(CD delta, sharpness): {summary['cd_delta_spearman_vs_sharpness']:.4f}",
        "",
        "The negative score correlations mean sharper patches tend to have slightly lower",
        "denoising scores, and the positive CD-delta correlations point in the same",
        "direction. The effect is small, though.",
        "",
        "## Quartiles",
        "",
        "| Sharpness bucket | Count | Mean score | Median score | Worse than noisy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    names = ["Q1 smoothest", "Q2", "Q3", "Q4 sharpest"]
    for name, row in zip(names, q):
        lines.append(
            f"| {name} | {row['count']} | {row['mean_score']:.2f} | "
            f"{row['median_score']:.2f} | {100.0 * row['worse_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Performance gets worse from Q1 to Q3, but the sharpest quartile recovers",
            "rather than continuing the trend. This makes the result non-monotonic.",
            "",
            "## Interpretation",
            "",
            "The model is not uniformly failing on sharp or angular regions. There is a",
            "mild degradation signal, especially compared with the smoothest quartile,",
            "but the sharpest quartile is not the worst by median score or failure rate.",
            "",
            "Useful outputs:",
            "",
            "- `sharpness_metrics.csv`",
            "- `summary.json`",
            "- `sharpness_bucket_summary.csv`",
            "- `point_normal_sharpness_vs_score.png`",
            "- `point_normal_sharpness_bucket_summary.png`",
            "- `mesh_normal_sharpness_vs_score.png`",
            "- `mesh_normal_sharpness_bucket_summary.png`",
            "- `sharp_worst_01.png` through `sharp_worst_08.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs1.1/patch_diagnostics/sharpness_test")
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    metrics_path = Path(args.metrics) if args.metrics else out_dir / "sharpness_metrics.csv"
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path

    rows = read_rows(metrics_path)
    previous = load_existing_summary(out_dir / "summary.json")
    stats_by_key = {
        "point_normal_var": sharpness_summary(rows, "point_normal_var"),
        "point_surface_var": sharpness_summary(rows, "point_surface_var"),
        "mesh_normal_var": sharpness_summary(rows, "mesh_normal_var"),
    }
    primary_key = "point_normal_var"
    primary = stats_by_key[primary_key]
    scores = [r["cd_score"] for r in rows]
    deltas = [r["cd_delta"] for r in rows]
    summary = {
        "checkpoint": previous.get("checkpoint"),
        "mesh_root": previous.get("mesh_root"),
        "candidates": len(rows),
        "patch_size": previous.get("patch_size"),
        "seed": previous.get("seed"),
        "primary_sharpness_key": primary_key,
        "score_pearson_vs_sharpness": primary["score_pearson"],
        "score_spearman_vs_sharpness": primary["score_spearman"],
        "cd_delta_pearson_vs_sharpness": primary["cd_delta_pearson"],
        "cd_delta_spearman_vs_sharpness": primary["cd_delta_spearman"],
        "overall_mean_score": float(np.mean(scores)),
        "overall_median_score": float(np.median(scores)),
        "overall_worse_rate": float(np.mean([d > 0 for d in deltas])),
        "buckets": primary["buckets"],
        "sharpness": stats_by_key,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_bucket_csv(stats_by_key, out_dir / "sharpness_bucket_summary.csv")
    save_plots(
        rows,
        stats_by_key["point_normal_var"]["buckets"],
        out_dir,
        key="point_normal_var",
        label="Point normal variation sharpness",
        prefix="point_normal_sharpness",
    )
    save_plots(
        rows,
        stats_by_key["mesh_normal_var"]["buckets"],
        out_dir,
        key="mesh_normal_var",
        label="Mesh normal variation sharpness",
        prefix="mesh_normal_sharpness",
    )
    write_report(summary, out_dir / "sharpness_report.md")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
