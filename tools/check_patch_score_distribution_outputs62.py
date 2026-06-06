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
import trimesh
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups  # noqa: E402
from src.model.parse import get_model  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs_result/outputs6.2/checkpoints/vm/checkpoint_best.pkl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_result/analysis_outputs/patch_score_distribution_outputs6.2"


def normalize_pc(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / max(float(scale), 1e-12)).astype(np.float32, copy=False)


def metric_to_score(val_pred, val_noisy):
    if val_noisy < 1e-15:
        return 100.0 if val_pred < 1e-15 else 0.0
    score = 100.0 * (1.0 - val_pred / val_noisy)
    return max(0.0, min(100.0, float(score)))


def chamfer_parts(pc_a, pc_b):
    tree_b = cKDTree(pc_b)
    dist_a2b, _ = tree_b.query(pc_a, k=1)
    tree_a = cKDTree(pc_a)
    dist_b2a, _ = tree_a.query(pc_b, k=1)
    return float((dist_a2b**2).mean()), float((dist_b2a**2).mean())


def load_model(checkpoint):
    model_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs/model/vm.yaml"),
        resolve=True,
    )
    transform_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs/transform/vm.yaml"),
        resolve=True,
    )
    model = get_model(model_config=model_cfg, transform_config=transform_cfg)
    model.load(str(checkpoint))
    model.eval()
    return model


def read_datalist(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def pca_geometry(points):
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)[::-1]
    l1, l2, l3 = eigvals
    total = float(l1 + l2 + l3)
    if l1 <= 1e-15 or total <= 1e-15:
        return {
            "linearity": np.nan,
            "planarity": np.nan,
            "scattering": np.nan,
            "curvature": np.nan,
            "bbox_ratio_min": np.nan,
            "bbox_ratio_mid": np.nan,
        }
    bbox = np.sort(points.max(axis=0) - points.min(axis=0))[::-1]
    return {
        "linearity": float((l1 - l2) / l1),
        "planarity": float((l2 - l3) / l1),
        "scattering": float(l3 / l1),
        "curvature": float(l3 / total),
        "bbox_ratio_min": float(bbox[-1] / max(bbox[0], 1e-12)),
        "bbox_ratio_mid": float(bbox[1] / max(bbox[0], 1e-12)),
    }


def local_geometry(points, k=24, max_points=300, seed=123):
    if points.shape[0] <= k + 2:
        return {
            "local_curv_mean": np.nan,
            "local_curv_p90": np.nan,
            "normal_var": np.nan,
        }
    rng = np.random.default_rng(seed)
    sample_idx = np.arange(points.shape[0])
    if sample_idx.size > max_points:
        sample_idx = rng.choice(sample_idx, size=max_points, replace=False)
    tree = cKDTree(points)
    curvatures = []
    normals = []
    for idx in sample_idx:
        _, nn_idx = tree.query(points[idx], k=min(k, points.shape[0]))
        neigh = points[nn_idx]
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(neigh.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        total = float(eigvals.sum())
        if total > 1e-12:
            curvatures.append(float(eigvals[0] / total))
        normals.append(eigvecs[:, 0])
    normals = np.asarray(normals, dtype=np.float64)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norm, 1e-12)
    tensor = normals.T @ normals / max(normals.shape[0], 1)
    eigvals = np.linalg.eigvalsh(tensor)[::-1]
    return {
        "local_curv_mean": float(np.mean(curvatures)) if curvatures else np.nan,
        "local_curv_p90": float(np.percentile(curvatures, 90)) if curvatures else np.nan,
        "normal_var": float(1.0 - eigvals[0]),
    }


def geometry_category(row):
    if row["local_curv_p90"] >= 0.11 or row["normal_var"] >= 0.18:
        return "high_curvature_or_normal_var"
    if row["linearity"] >= 0.72 and row["bbox_ratio_mid"] <= 0.55:
        return "linear_thin"
    if row["planarity"] >= 0.48 and row["bbox_ratio_min"] <= 0.22:
        return "flat_sheet"
    if row["scattering"] >= 0.16 or row["bbox_ratio_min"] >= 0.34:
        return "volumetric_scattered"
    return "mixed_regular"


def sample_patch(rel_path, mesh_root, rng, patch_size):
    mesh_path = mesh_root / rel_path / "models/model_normalized.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    clean, _, _, _ = sample_vertex_groups(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        num_samples=32768,
        num_vertex_samples=1024,
    )
    clean = normalize_pc(clean.astype(np.float32, copy=False))
    noise_std = float(rng.uniform(0.005, 0.020))
    noisy = clean + rng.laplace(0, noise_std, size=clean.shape).astype(np.float32)
    seed_idx = int(rng.integers(0, noisy.shape[0]))
    seed_point = noisy[seed_idx]
    _, nn_idx = cKDTree(noisy).query(seed_point[None, :], k=min(patch_size, noisy.shape[0]))
    nn_idx = np.asarray(nn_idx).reshape(-1)
    return {
        "rel_path": rel_path,
        "noise_std": noise_std,
        "seed_idx": seed_idx,
        "patch_noisy": (noisy[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False),
        "patch_clean": (clean[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False),
    }


def evaluate_patch(model, patch, geom_seed):
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
    geom = pca_geometry(clean)
    local = local_geometry(clean, seed=geom_seed)
    row = {
        "rel_path": patch["rel_path"],
        "seed_idx": int(patch["seed_idx"]),
        "noise_std": float(patch["noise_std"]),
        "patch_size": int(clean.shape[0]),
        "cd_noisy": cd_noisy,
        "cd_pred": cd_pred,
        "cd_delta": cd_pred - cd_noisy,
        "cd_score": metric_to_score(cd_pred, cd_noisy),
        "paired_noisy": float(((noisy - clean) ** 2).sum(axis=1).mean()),
        "paired_pred": float(((pred - clean) ** 2).sum(axis=1).mean()),
        **geom,
        **local,
    }
    row["geometry_category"] = geometry_category(row)
    return row, pred


def quantile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def category_summary(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["geometry_category"]].append(row)
    out = []
    for category, rows in sorted(grouped.items()):
        scores = [r["cd_score"] for r in rows]
        out.append(
            {
                "geometry_category": category,
                "count": len(rows),
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "min_score": float(np.min(scores)),
                "worse_rate": float(np.mean([r["cd_delta"] > 0 for r in rows])),
            }
        )
    return out


def select_representatives(records):
    ranked = sorted(records, key=lambda r: r["cd_score"])
    mid = len(ranked) // 2
    selected = []
    for label, items in (
        ("worst", ranked[:3]),
        ("median", ranked[max(0, mid - 1) : min(len(ranked), mid + 2)]),
        ("best", ranked[-3:]),
    ):
        for row in items:
            selected.append((label, row))
    return selected


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


def render_patch(row, patch, pred, out_path):
    clean = patch["patch_clean"]
    noisy = patch["patch_noisy"]
    pts = np.concatenate([clean, noisy, pred], axis=0)
    fig = plt.figure(figsize=(12.5, 4.0))
    panels = [("clean", clean, "#222222"), ("noisy", noisy, "#d62728"), ("pred", pred, "#1f77b4")]
    for i, (title, points, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        set_axes_equal(ax, pts)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, c=color, alpha=0.72)
        ax.view_init(elev=18, azim=35)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        "score={:.2f} | {} | {}".format(
            row["cd_score"], row["geometry_category"], row["rel_path"]
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_plots(records, category_rows, out_dir):
    scores = np.asarray([r["cd_score"] for r in records], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(scores, bins=min(14, max(6, len(scores) // 3)), color="#4e79a7", alpha=0.86)
    ax.axvline(scores.mean(), color="#222222", linestyle="--", label=f"mean {scores.mean():.2f}")
    ax.axvline(np.median(scores), color="#e15759", linestyle=":", label=f"median {np.median(scores):.2f}")
    ax.set_xlabel("CD improvement score")
    ax.set_ylabel("Patch count")
    ax.set_title("Patch score distribution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [r["geometry_category"] for r in category_rows]
    med = [r["median_score"] for r in category_rows]
    counts = [r["count"] for r in category_rows]
    x = np.arange(len(labels))
    ax.bar(x, med, color="#59a14f", alpha=0.86)
    for i, count in enumerate(counts):
        ax.text(i, med[i], f"n={count}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Median CD score")
    ax.set_title("Median score by geometry category")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "score_by_geometry_category.png", dpi=180)
    plt.close(fig)


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_report(summary, out_path):
    lines = [
        "# Patch Score Distribution: outputs6.2",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "## Overall",
        "",
    ]
    for key, value in summary["score_summary"].items():
        lines.append(f"- {key}: `{value:.4f}`" if isinstance(value, float) else f"- {key}: `{value}`")
    lines += [
        f"- worse_than_noisy_rate: `{summary['worse_than_noisy_rate']:.4f}`",
        f"- worst_patch_category_counts: `{summary['worst_patch_category_counts']}`",
        "",
        "## Geometry Categories",
        "",
        "| Category | Count | Mean | Median | Min | Worse rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["category_summary"]:
        lines.append(
            "| {geometry_category} | {count} | {mean_score:.2f} | {median_score:.2f} | "
            "{min_score:.2f} | {worse_rate:.2%} |".format(**row)
        )
    lines += [
        "",
        "## Selected Patches",
        "",
    ]
    for row in summary["selected_patches"]:
        lines.append(
            "- {selection}: score=`{cd_score:.2f}`, category=`{geometry_category}`, "
            "rel_path=`{rel_path}`, image=`{image}`".format(**row)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify(records, cat_rows):
    scores = np.asarray([r["cd_score"] for r in records], dtype=np.float64)
    spread = float(scores.std())
    mean_score = float(scores.mean())
    worst_count = max(3, int(round(len(records) * 0.2)))
    worst = sorted(records, key=lambda r: r["cd_score"])[:worst_count]
    worst_counter = Counter(r["geometry_category"] for r in worst)
    dominant_count = max(worst_counter.values()) if worst_counter else 0
    dominant_rate = dominant_count / max(worst_count, 1)
    category_medians = [r["median_score"] for r in cat_rows if r["count"] >= 3]
    category_gap = max(category_medians) - min(category_medians) if len(category_medians) >= 2 else 0.0
    low_tail_gap = mean_score - float(np.percentile(scores, 10))
    if dominant_rate >= 0.6 and category_gap >= max(3.0, 0.6 * spread):
        return "CATEGORY_CONCENTRATED_LOW_TAIL"
    if low_tail_gap <= max(2.0, 0.55 * spread):
        return "NEAR_AVERAGE_OVERALL"
    return "LOW_TAIL_WEAK_CATEGORY_CONCENTRATION"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mesh-root", default=str(PROJECT_ROOT / "dataset_clean"))
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "selected_patch_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    rel_paths = read_datalist(PROJECT_ROOT / args.datalist)
    mesh_root = Path(args.mesh_root)
    usable = [
        rel for rel in rel_paths
        if (mesh_root / rel / "models/model_normalized.obj").exists()
    ]
    if not usable:
        raise FileNotFoundError(f"No mesh files found under {mesh_root}")
    chosen = [usable[int(rng.integers(0, len(usable)))] for _ in range(args.candidates)]
    model = load_model(checkpoint)

    records = []
    patches_by_key = {}
    preds_by_key = {}
    for idx, rel_path in enumerate(chosen, start=1):
        print(f"[{idx}/{len(chosen)}] {rel_path}", flush=True)
        patch = sample_patch(rel_path, mesh_root, rng, args.patch_size)
        row, pred = evaluate_patch(model, patch, args.seed + idx)
        row["index"] = idx
        records.append(row)
        patches_by_key[idx] = patch
        preds_by_key[idx] = pred

    cat_rows = category_summary(records)
    selected = []
    for n, (label, row) in enumerate(select_representatives(records), start=1):
        image_name = f"{n:02d}_{label}_score_{row['cd_score']:.2f}.png"
        render_patch(row, patches_by_key[row["index"]], preds_by_key[row["index"]], image_dir / image_name)
        selected.append(
            {
                "selection": label,
                "index": row["index"],
                "cd_score": row["cd_score"],
                "geometry_category": row["geometry_category"],
                "rel_path": row["rel_path"],
                "image": f"selected_patch_images/{image_name}",
            }
        )

    scores = np.asarray([r["cd_score"] for r in records], dtype=np.float64)
    worst_count = max(3, int(round(len(records) * 0.2)))
    worst = sorted(records, key=lambda r: r["cd_score"])[:worst_count]
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "device": "cuda" if args.use_cuda else "cpu",
        "seed": args.seed,
        "patch_size": args.patch_size,
        "candidates": args.candidates,
        "score_summary": quantile_summary(scores),
        "worse_than_noisy_rate": float(np.mean([r["cd_delta"] > 0 for r in records])),
        "category_summary": cat_rows,
        "worst_patch_count": worst_count,
        "worst_patch_category_counts": dict(Counter(r["geometry_category"] for r in worst)),
        "selected_patches": selected,
    }
    summary["conclusion"] = classify(records, cat_rows)

    write_csv(out_dir / "patch_score_records.csv", records)
    write_csv(out_dir / "category_summary.csv", cat_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_plots(records, cat_rows, out_dir)
    make_report(summary, out_dir / "report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
