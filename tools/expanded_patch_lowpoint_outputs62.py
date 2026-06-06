import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jittor as jt
import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_result/analysis_outputs/expanded_patch_lowpoint_outputs6.2"


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def fine_tags(row, thresholds):
    tags = [row["geometry_category"]]
    if row["local_curv_p90"] >= thresholds["local_curv_p90_p75"]:
        tags.append("q75_high_local_curv_p90")
    if row["local_curv_p90"] >= thresholds["local_curv_p90_p90"]:
        tags.append("q90_high_local_curv_p90")
    if row["normal_var"] >= thresholds["normal_var_p75"]:
        tags.append("q75_high_normal_var")
    if row["normal_var"] >= thresholds["normal_var_p90"]:
        tags.append("q90_high_normal_var")
    if row["scattering"] >= thresholds["scattering_p75"]:
        tags.append("q75_high_scattering")
    if row["bbox_ratio_min"] >= thresholds["bbox_ratio_min_p75"]:
        tags.append("q75_thick_bbox_min")
    if (
        row["local_curv_p90"] >= thresholds["local_curv_p90_p75"]
        and row["normal_var"] >= thresholds["normal_var_p75"]
    ):
        tags.append("q75_high_curv_and_normal")
    if (
        row["scattering"] >= thresholds["scattering_p75"]
        and row["bbox_ratio_min"] >= thresholds["bbox_ratio_min_p75"]
    ):
        tags.append("q75_scattered_and_thick")
    if (
        row["local_curv_p90"] >= thresholds["local_curv_p90_p75"]
        and row["scattering"] >= thresholds["scattering_p75"]
        and row["bbox_ratio_min"] >= thresholds["bbox_ratio_min_p75"]
    ):
        tags.append("q75_curv_scattered_thick")
    return tags


def tag_low_rate(rows, low_threshold, min_count):
    feature_names = [
        "local_curv_p90",
        "normal_var",
        "scattering",
        "bbox_ratio_min",
    ]
    thresholds = {}
    for name in feature_names:
        values = [row[name] for row in rows]
        thresholds[f"{name}_p75"] = safe_quantile(values, 0.75)
        thresholds[f"{name}_p90"] = safe_quantile(values, 0.90)

    tagged = []
    grouped = defaultdict(list)
    for row in rows:
        tags = fine_tags(row, thresholds)
        row["fine_tags"] = ";".join(tags)
        tagged.append(row)
        for tag in tags:
            grouped[tag].append(row)

    summaries = []
    for tag, group in sorted(grouped.items()):
        if len(group) < min_count:
            continue
        scores = np.asarray([row["cd_score"] for row in group], dtype=np.float64)
        low = scores <= low_threshold
        summaries.append(
            {
                "tag": tag,
                "count": len(group),
                "low_count": int(low.sum()),
                "low_rate": float(low.mean()),
                "mean_score": float(scores.mean()),
                "median_score": float(np.median(scores)),
                "min_score": float(scores.min()),
            }
        )
    summaries.sort(key=lambda row: (-row["low_rate"], -row["count"], row["tag"]))
    return tagged, summaries, thresholds


def evaluate_patch_with_pred(model, patch, geom_seed):
    row, _ = evaluate_patch(model, patch, geom_seed)
    pc_noisy = jt.array(patch["patch_noisy"][None, :, :])
    with jt.no_grad():
        pc_pred, _ = model.denoise_langevin_dynamics(pc_noisy)
    pred = pc_pred.detach().numpy()[0].astype(np.float32, copy=False)
    return row, pred


def point_local_features(clean, noisy, pred, point_idx, tree, k):
    _, nn_idx = tree.query(clean[point_idx], k=min(k, clean.shape[0]))
    nn_idx = np.asarray(nn_idx).reshape(-1)
    neigh = clean[nn_idx]
    centered = neigh - neigh.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(neigh.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    total = float(eigvals.sum())
    local_curv = float(eigvals[0] / total) if total > 1e-12 else np.nan
    normal = eigvecs[:, 0]
    normals = []
    for idx in nn_idx:
        _, sub_idx = tree.query(clean[idx], k=min(k, clean.shape[0]))
        sub = clean[np.asarray(sub_idx).reshape(-1)]
        sub_centered = sub - sub.mean(axis=0, keepdims=True)
        sub_cov = sub_centered.T @ sub_centered / max(sub.shape[0] - 1, 1)
        _, sub_vecs = np.linalg.eigh(sub_cov)
        normals.append(sub_vecs[:, 0])
    normals = np.asarray(normals)
    normal_dot = np.abs(normals @ normal)
    radius = np.linalg.norm(neigh - clean[point_idx], axis=1)
    target = clean[point_idx] - noisy[point_idx]
    displacement = pred[point_idx] - noisy[point_idx]
    target_norm = float(np.linalg.norm(target))
    disp_norm = float(np.linalg.norm(displacement))
    denom = max(target_norm * disp_norm, 1e-12)
    return {
        "point_error_sq": float(((pred[point_idx] - clean[point_idx]) ** 2).sum()),
        "noisy_error_sq": float(((noisy[point_idx] - clean[point_idx]) ** 2).sum()),
        "error_gain_sq": float(
            ((noisy[point_idx] - clean[point_idx]) ** 2).sum()
            - ((pred[point_idx] - clean[point_idx]) ** 2).sum()
        ),
        "target_norm": target_norm,
        "disp_norm": disp_norm,
        "disp_target_cos": float((target * displacement).sum() / denom),
        "local_curvature": local_curv,
        "local_linearity": float((eigvals[2] - eigvals[1]) / max(eigvals[2], 1e-12)),
        "local_planarity": float((eigvals[1] - eigvals[0]) / max(eigvals[2], 1e-12)),
        "local_scattering": float(eigvals[0] / max(eigvals[2], 1e-12)),
        "local_normal_variation": float(1.0 - normal_dot.mean()),
        "local_radius_mean": float(radius.mean()),
        "local_radius_std": float(radius.std()),
        "local_radius_max": float(radius.max()),
    }


def set_axes_equal(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def render_worst_points(row, patch, pred, worst_idx, out_path):
    clean = patch["patch_clean"]
    noisy = patch["patch_noisy"]
    pts = np.concatenate([clean, pred], axis=0)
    fig = plt.figure(figsize=(12.5, 4.0))
    panels = [
        ("clean + worst points", clean, "#222222"),
        ("noisy + worst points", noisy, "#d62728"),
        ("pred + worst points", pred, "#1f77b4"),
    ]
    for i, (title, points, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        set_axes_equal(ax, pts)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3, c=color, alpha=0.35)
        ax.scatter(
            points[worst_idx, 0],
            points[worst_idx, 1],
            points[worst_idx, 2],
            s=28,
            c="#ffbf00",
            edgecolors="#111111",
            linewidths=0.35,
            alpha=0.95,
        )
        ax.view_init(elev=18, azim=35)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        "patch score={:.2f} | {} | {}".format(
            row["cd_score"], row["geometry_category"], row["rel_path"]
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def analyze_worst_points(model, rows, patches, preds, out_dir, num_patches, num_points, k):
    image_dir = out_dir / "worst_point_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    point_rows = []
    for rank, row in enumerate(sorted(rows, key=lambda item: item["cd_score"])[:num_patches], start=1):
        patch = patches[row["index"]]
        pred = preds[row["index"]]
        clean = patch["patch_clean"]
        noisy = patch["patch_noisy"]
        point_error = ((pred - clean) ** 2).sum(axis=1)
        worst_idx = np.argsort(point_error)[-num_points:][::-1]
        tree = cKDTree(clean)
        image_name = f"{rank:02d}_patch_{row['index']:03d}_score_{row['cd_score']:.2f}.png"
        render_worst_points(row, patch, pred, worst_idx, image_dir / image_name)
        for local_rank, point_idx in enumerate(worst_idx, start=1):
            item = {
                "patch_rank": rank,
                "point_rank": local_rank,
                "patch_index": row["index"],
                "point_idx": int(point_idx),
                "patch_score": row["cd_score"],
                "geometry_category": row["geometry_category"],
                "fine_tags": row.get("fine_tags", ""),
                "rel_path": row["rel_path"],
                "image": f"worst_point_images/{image_name}",
            }
            item.update(point_local_features(clean, noisy, pred, int(point_idx), tree, k))
            point_rows.append(item)
    return point_rows


def summarize_points(point_rows):
    keys = [
        "point_error_sq",
        "noisy_error_sq",
        "error_gain_sq",
        "target_norm",
        "disp_norm",
        "disp_target_cos",
        "local_curvature",
        "local_scattering",
        "local_normal_variation",
        "local_radius_mean",
        "local_radius_max",
    ]
    return {key: summarize([row[key] for row in point_rows]) for key in keys}


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def make_report(summary, out_path):
    lines = [
        "# Expanded Patch Low-Point Analysis: outputs6.2",
        "",
        f"Samples: `{summary['candidates']}`, patch_size: `{summary['patch_size']}`",
        f"Bottom-30 threshold: `{summary['low_score_threshold']:.4f}`",
        "",
        "## Low-Rate Tags",
        "",
        "| Tag | Count | Low count | Low rate | Median score | Min score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["tag_low_rate"][:18]:
        lines.append(
            "| {tag} | {count} | {low_count} | {low_rate:.2%} | "
            "{median_score:.2f} | {min_score:.2f} |".format(**row)
        )
    lines += [
        "",
        "## Worst Local Points",
        "",
    ]
    for key, stats in summary["worst_point_summary"].items():
        lines.append(
            f"- {key}: mean `{stats['mean']:.6f}`, median `{stats['median']:.6f}`, "
            f"min `{stats['min']:.6f}`, max `{stats['max']:.6f}`"
        )
    lines += [
        "",
        "## Notes",
        "",
        f"- tags_over_80_low_rate: `{summary['tags_over_80_low_rate']}`",
        f"- worst_patch_category_counts: `{summary['worst_patch_category_counts']}`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mesh-root", default=str(PROJECT_ROOT / "dataset_clean"))
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--candidates", type=int, default=160)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--min-tag-count", type=int, default=8)
    parser.add_argument("--worst-patches", type=int, default=6)
    parser.add_argument("--worst-points", type=int, default=24)
    parser.add_argument("--point-knn", type=int, default=32)
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
        rel
        for rel in read_datalist(PROJECT_ROOT / args.datalist)
        if (mesh_root / rel / "models/model_normalized.obj").exists()
    ]
    if not usable:
        raise FileNotFoundError(f"No mesh files found under {mesh_root}")
    chosen = [usable[int(rng.integers(0, len(usable)))] for _ in range(args.candidates)]
    model = load_model(checkpoint)

    rows = []
    patches = {}
    preds = {}
    for idx, rel_path in enumerate(chosen, start=1):
        print(f"[{idx}/{len(chosen)}] {rel_path}", flush=True)
        patch = sample_patch(rel_path, mesh_root, rng, args.patch_size)
        row, pred = evaluate_patch_with_pred(model, patch, args.seed + idx)
        row["index"] = idx
        rows.append(row)
        patches[idx] = patch
        preds[idx] = pred

    scores = np.asarray([row["cd_score"] for row in rows], dtype=np.float64)
    low_threshold = float(np.quantile(scores, 0.30))
    rows, tag_rows, thresholds = tag_low_rate(rows, low_threshold, args.min_tag_count)
    point_rows = analyze_worst_points(
        model=model,
        rows=rows,
        patches=patches,
        preds=preds,
        out_dir=out_dir,
        num_patches=args.worst_patches,
        num_points=args.worst_points,
        k=args.point_knn,
    )
    worst_patch_count = max(1, int(round(args.candidates * 0.30)))
    worst_patches = sorted(rows, key=lambda row: row["cd_score"])[:worst_patch_count]
    tags_over_80 = [
        row for row in tag_rows if row["low_rate"] >= 0.80 and row["count"] >= args.min_tag_count
    ]
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "seed": args.seed,
        "candidates": args.candidates,
        "patch_size": args.patch_size,
        "low_score_threshold": low_threshold,
        "score_summary": summarize(scores),
        "thresholds": thresholds,
        "tag_low_rate": tag_rows,
        "tags_over_80_low_rate": tags_over_80,
        "worst_patch_category_counts": dict(
            Counter(row["geometry_category"] for row in worst_patches)
        ),
        "worst_point_summary": summarize_points(point_rows),
    }

    write_csv(out_dir / "expanded_patch_records.csv", rows)
    write_csv(out_dir / "tag_low_rate.csv", tag_rows)
    write_csv(out_dir / "worst_local_points.csv", point_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_report(summary, out_dir / "report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
