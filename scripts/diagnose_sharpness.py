import argparse
import csv
import json
import random
import sys
from pathlib import Path

import jittor as jt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups
from scripts.legacy_vm import load_legacy_model


def normalize_pc_with_params(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / scale).astype(np.float32, copy=False), center, float(scale)


def chamfer_distance(pc_a, pc_b):
    tree_b = cKDTree(pc_b)
    dist_a2b, _ = tree_b.query(pc_a, k=1)
    tree_a = cKDTree(pc_a)
    dist_b2a, _ = tree_a.query(pc_b, k=1)
    return float((dist_a2b**2).mean() + (dist_b2a**2).mean())


def metric_to_score(val_pred, val_noisy):
    if val_noisy < 1e-15:
        return 100.0 if val_pred < 1e-15 else 0.0
    score = 100.0 * (1.0 - val_pred / val_noisy)
    return max(0.0, min(100.0, float(score)))


def orientation_variation(normals, weights=None):
    if normals.shape[0] == 0:
        return np.nan
    if weights is None:
        weights = np.ones((normals.shape[0],), dtype=np.float64)
    weights = weights.astype(np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    tensor = np.zeros((3, 3), dtype=np.float64)
    for n, w in zip(normals, weights):
        n = n.astype(np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        tensor += w * np.outer(n, n)
    eigvals = np.linalg.eigvalsh(tensor)
    eigvals = np.sort(eigvals)[::-1]
    return float(1.0 - eigvals[0])


def estimate_point_sharpness(points, k=24, max_points=500, seed=123):
    if points.shape[0] <= k + 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    if points.shape[0] > max_points:
        sample_idx = rng.choice(points.shape[0], size=max_points, replace=False)
    else:
        sample_idx = np.arange(points.shape[0])
    tree = cKDTree(points)
    normals = []
    surface_variations = []
    for idx in sample_idx:
        _, nn_idx = tree.query(points[idx], k=min(k, points.shape[0]))
        neigh = points[nn_idx]
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = float(eigvals.sum())
        if total > 1e-12:
            surface_variations.append(float(eigvals[0] / total))
        normals.append(eigvecs[:, 0])
    normals = np.asarray(normals)
    normal_variation = orientation_variation(normals)
    surface_variation = float(np.mean(surface_variations)) if surface_variations else np.nan
    return normal_variation, surface_variation


def crop_mesh_faces(vertices, faces, radius, max_faces=5000):
    face_vertices = vertices[faces]
    face_centers = face_vertices.mean(axis=1)
    vertex_radius = np.sqrt((face_vertices**2).sum(axis=2))
    face_radius_max = vertex_radius.max(axis=1)
    center_radius = np.sqrt((face_centers**2).sum(axis=1))
    edge_01 = np.linalg.norm(face_vertices[:, 0] - face_vertices[:, 1], axis=1)
    edge_12 = np.linalg.norm(face_vertices[:, 1] - face_vertices[:, 2], axis=1)
    edge_20 = np.linalg.norm(face_vertices[:, 2] - face_vertices[:, 0], axis=1)
    edge_max = np.maximum(np.maximum(edge_01, edge_12), edge_20)
    context_radius = radius * 2.0
    mask = (
        (center_radius <= context_radius)
        & (face_radius_max <= context_radius * 1.15)
        & (edge_max <= max(radius * 1.2, 1e-6))
    )
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        selected = np.argsort(center_radius)[: min(max_faces, len(faces))]
    if selected.size > max_faces:
        selected = selected[np.argsort(center_radius[selected])[:max_faces]]
    return selected


def estimate_mesh_sharpness(vertices, faces, radius):
    selected = crop_mesh_faces(vertices, faces, radius)
    if selected.size == 0:
        return np.nan, 0
    face_vertices = vertices[faces[selected]]
    normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    areas = np.linalg.norm(normals, axis=1) * 0.5
    valid = areas > 1e-12
    if not valid.any():
        return np.nan, int(selected.size)
    normals = normals[valid] / np.linalg.norm(normals[valid], axis=1, keepdims=True)
    areas = areas[valid]
    return orientation_variation(normals, areas), int(selected.size)


def sample_patch(rel_path, mesh_root, rng, patch_size):
    mesh_path = mesh_root / rel_path / "models/model_normalized.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int32)
    clean, _, _, _ = sample_vertex_groups(
        vertices=mesh_vertices,
        faces=mesh_faces,
        num_samples=32768,
        num_vertex_samples=1024,
    )
    clean, center, scale = normalize_pc_with_params(clean.astype(np.float32, copy=False))
    mesh_vertices = ((mesh_vertices - center) / scale).astype(np.float32, copy=False)
    noise_std = float(rng.uniform(0.005, 0.020))
    noisy = clean + rng.laplace(0, noise_std, size=clean.shape).astype(np.float32)

    seed_idx = int(rng.integers(0, noisy.shape[0]))
    seed_point = noisy[seed_idx].astype(np.float32, copy=False)
    _, nn_idx = cKDTree(noisy).query(seed_point[None, :], k=patch_size)
    nn_idx = nn_idx[0]

    patch_noisy = (noisy[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    patch_clean = (clean[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    mesh_local = (mesh_vertices - seed_point[None, :]).astype(np.float32, copy=False)
    patch_radius = float(np.sqrt((patch_noisy**2).sum(axis=1)).max())
    return {
        "rel_path": rel_path,
        "noise_std": noise_std,
        "seed_idx": seed_idx,
        "patch_noisy": patch_noisy,
        "patch_clean": patch_clean,
        "mesh_vertices": mesh_local,
        "mesh_faces": mesh_faces,
        "patch_radius": patch_radius,
    }


def evaluate_patch(model, patch, sharp_seed):
    pc_noisy = jt.array(patch["patch_noisy"][None, :, :])
    with jt.no_grad():
        pc_pred, _ = model.denoise_langevin_dynamics(pc_noisy)
    pred = pc_pred.detach().numpy()[0].astype(np.float32, copy=False)
    clean = patch["patch_clean"]
    noisy = patch["patch_noisy"]
    cd_noisy = chamfer_distance(noisy, clean)
    cd_pred = chamfer_distance(pred, clean)
    point_normal_var, point_surface_var = estimate_point_sharpness(
        clean,
        seed=sharp_seed,
    )
    mesh_normal_var, mesh_face_count = estimate_mesh_sharpness(
        patch["mesh_vertices"],
        patch["mesh_faces"],
        patch["patch_radius"],
    )
    return {
        "rel_path": patch["rel_path"],
        "noise_std": patch["noise_std"],
        "seed_idx": patch["seed_idx"],
        "patch_radius": patch["patch_radius"],
        "cd_noisy": cd_noisy,
        "cd_pred": cd_pred,
        "cd_delta": cd_pred - cd_noisy,
        "cd_score": metric_to_score(cd_pred, cd_noisy),
        "point_normal_var": point_normal_var,
        "point_surface_var": point_surface_var,
        "mesh_normal_var": mesh_normal_var,
        "mesh_face_count": mesh_face_count,
        "patch_clean": clean,
        "patch_pred": pred,
        "mesh_vertices": patch["mesh_vertices"],
        "mesh_faces": patch["mesh_faces"],
    }


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
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
        return np.nan
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def bucket_stats(records, key, num_buckets=4):
    values = np.asarray([r[key] for r in records], dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return []
    quantiles = np.quantile(finite_values, np.linspace(0, 1, num_buckets + 1))
    stats = []
    for i in range(num_buckets):
        lo = quantiles[i]
        hi = quantiles[i + 1]
        if i == num_buckets - 1:
            bucket = [r for r in records if np.isfinite(r[key]) and lo <= r[key] <= hi]
        else:
            bucket = [r for r in records if np.isfinite(r[key]) and lo <= r[key] < hi]
        scores = [r["cd_score"] for r in bucket]
        deltas = [r["cd_delta"] for r in bucket]
        stats.append({
            "bucket": i + 1,
            "sharp_min": float(lo),
            "sharp_max": float(hi),
            "count": len(bucket),
            "mean_score": float(np.mean(scores)) if scores else np.nan,
            "median_score": float(np.median(scores)) if scores else np.nan,
            "worse_rate": float(np.mean([d > 0 for d in deltas])) if deltas else np.nan,
            "mean_cd_delta": float(np.mean(deltas)) if deltas else np.nan,
        })
    return stats


def sharpness_summary(records, key, num_buckets=4):
    sharp = [r[key] for r in records]
    scores = [r["cd_score"] for r in records]
    deltas = [r["cd_delta"] for r in records]
    return {
        "key": key,
        "score_pearson": pearson(sharp, scores),
        "score_spearman": spearman(sharp, scores),
        "cd_delta_pearson": pearson(sharp, deltas),
        "cd_delta_spearman": spearman(sharp, deltas),
        "buckets": bucket_stats(records, key=key, num_buckets=num_buckets),
    }


def write_bucket_csv(stats_by_key, out_path):
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
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, summary in stats_by_key.items():
            for row in summary["buckets"]:
                writer.writerow({"sharpness_key": key, **row})


def set_axes_equal(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    if radius < 1e-8:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def render_triptych(item, rank, out_path):
    clean = item["patch_clean"]
    pred = item["patch_pred"]
    vertices = item["mesh_vertices"]
    faces = item["mesh_faces"]
    selected = crop_mesh_faces(vertices, faces, item["patch_radius"])
    selected_faces = faces[selected]
    unique_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    mesh_v = vertices[unique_vertices]
    mesh_f = inverse.reshape(selected_faces.shape)
    axis_pts = np.concatenate([clean, pred], axis=0)

    fig = plt.figure(figsize=(14.5, 4.6))
    axes = [fig.add_subplot(1, 3, i, projection="3d") for i in range(1, 4)]
    for ax in axes:
        set_axes_equal(ax, axis_pts)
        ax.view_init(elev=18, azim=35)
    axes[0].scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=5, c="#222222", alpha=0.9)
    axes[0].set_title("Clean points", fontsize=11)
    mesh_collection = Poly3DCollection(
        mesh_v[mesh_f],
        facecolor="#9ecae1",
        edgecolor="#6f93a8",
        linewidth=0.05,
        alpha=0.78,
    )
    axes[1].add_collection3d(mesh_collection)
    axes[1].set_title("True mesh surface", fontsize=11)
    axes[2].scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=5, c="#4e79a7", alpha=0.9)
    axes[2].set_title("Pred points", fontsize=11)
    fig.suptitle(
        f"Sharp-worst {rank} | sharp={item['point_normal_var']:.4f} | "
        f"score={item['cd_score']:.2f} | {item['rel_path']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plots(records, stats, out_dir, key, label, prefix):
    sharp = np.asarray([r[key] for r in records])
    scores = np.asarray([r["cd_score"] for r in records])
    deltas = np.asarray([r["cd_delta"] for r in records])
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


def strip_arrays(record):
    return {
        k: v
        for k, v in record.items()
        if not isinstance(v, np.ndarray)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs1.1/checkpoints/vm/checkpoint_best.pkl")
    parser.add_argument("--mesh-root", default="E:/Code/competition2_EdgeConv/dataset_clean")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default="outputs1.1/patch_diagnostics/sharpness_test")
    parser.add_argument("--candidates", type=int, default=250)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_root = Path(args.mesh_root)
    rel_paths = [
        line.strip()
        for line in (PROJECT_ROOT / args.datalist).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    chosen = [rel_paths[int(rng.integers(0, len(rel_paths)))] for _ in range(args.candidates)]
    model = load_legacy_model(PROJECT_ROOT / args.checkpoint)

    records = []
    for i, rel_path in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {rel_path}", flush=True)
        patch = sample_patch(rel_path, mesh_root, rng, args.patch_size)
        records.append(evaluate_patch(model, patch, sharp_seed=args.seed + i))

    rows = [strip_arrays(r) for r in records]
    fieldnames = list(rows[0].keys())
    with (out_dir / "sharpness_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats_by_key = {
        "point_normal_var": sharpness_summary(records, "point_normal_var"),
        "point_surface_var": sharpness_summary(records, "point_surface_var"),
        "mesh_normal_var": sharpness_summary(records, "mesh_normal_var"),
    }
    write_bucket_csv(stats_by_key, out_dir / "sharpness_bucket_summary.csv")

    primary_key = "point_normal_var"
    primary_stats = stats_by_key[primary_key]
    sharp = [r[primary_key] for r in records]
    scores = [r["cd_score"] for r in records]
    deltas = [r["cd_delta"] for r in records]
    summary = {
        "checkpoint": str((PROJECT_ROOT / args.checkpoint).resolve()),
        "mesh_root": str(mesh_root.resolve()),
        "candidates": args.candidates,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "primary_sharpness_key": primary_key,
        "score_pearson_vs_sharpness": primary_stats["score_pearson"],
        "score_spearman_vs_sharpness": primary_stats["score_spearman"],
        "cd_delta_pearson_vs_sharpness": primary_stats["cd_delta_pearson"],
        "cd_delta_spearman_vs_sharpness": primary_stats["cd_delta_spearman"],
        "overall_mean_score": float(np.mean(scores)),
        "overall_median_score": float(np.median(scores)),
        "overall_worse_rate": float(np.mean([d > 0 for d in deltas])),
        "buckets": primary_stats["buckets"],
        "sharpness": stats_by_key,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plots(
        records,
        stats_by_key["point_normal_var"]["buckets"],
        out_dir,
        key="point_normal_var",
        label="Point normal variation sharpness",
        prefix="point_normal_sharpness",
    )
    save_plots(
        records,
        stats_by_key["mesh_normal_var"]["buckets"],
        out_dir,
        key="mesh_normal_var",
        label="Mesh normal variation sharpness",
        prefix="mesh_normal_sharpness",
    )

    records_by_sharp_bad = sorted(
        records,
        key=lambda r: (-r[primary_key], r["cd_score"]),
    )
    examples = records_by_sharp_bad[:8]
    for rank, item in enumerate(examples, 1):
        render_triptych(item, rank, out_dir / f"sharp_worst_{rank:02d}.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
