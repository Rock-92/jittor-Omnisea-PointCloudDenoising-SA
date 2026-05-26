import argparse
import csv
import json
import math
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.legacy_vm import load_legacy_model
from src.data.utils import sample_vertex_groups


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


def orientation_variation(normals):
    if normals.shape[0] == 0:
        return math.nan
    tensor = np.zeros((3, 3), dtype=np.float64)
    for n in normals:
        n = n.astype(np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        tensor += np.outer(n, n)
    tensor /= max(normals.shape[0], 1)
    eigvals = np.linalg.eigvalsh(tensor)
    eigvals = np.sort(eigvals)[::-1]
    return float(1.0 - eigvals[0])


def estimate_point_sharpness(points, k=24, max_points=400, seed=123):
    if points.shape[0] <= k + 2:
        return math.nan, math.nan
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
    normal_var = orientation_variation(normals)
    surface_var = float(np.mean(surface_variations)) if surface_variations else math.nan
    return normal_var, surface_var


def pca_geometry(points):
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)[::-1]
    l1, l2, l3 = eigvals
    if l1 <= 1e-15:
        return {
            "linearity": math.nan,
            "planarity": math.nan,
            "scattering": math.nan,
            "thinness": math.nan,
        }
    return {
        "linearity": float((l1 - l2) / l1),
        "planarity": float((l2 - l3) / l1),
        "scattering": float(l3 / l1),
        "thinness": float(np.sqrt(l3 / l1)),
    }


def structure_risk(geom, point_normal_var, point_surface_var):
    thinness = geom["thinness"]
    linearity = geom["linearity"]
    thin_risk = 0.0 if not np.isfinite(thinness) else max(0.0, (0.18 - thinness) / 0.18)
    linear_risk = 0.0 if not np.isfinite(linearity) else max(0.0, (linearity - 0.45) / 0.55)
    surface_risk = 0.0 if not np.isfinite(point_surface_var) else min(point_surface_var / 0.16, 1.0)
    normal_risk = 0.0 if not np.isfinite(point_normal_var) else min(point_normal_var / 0.55, 1.0)
    return float(0.45 * thin_risk + 0.25 * linear_risk + 0.20 * surface_risk + 0.10 * normal_risk)


def load_mesh_patch(rel_path, mesh_root, rng, args, candidate_idx):
    mesh_path = mesh_root / rel_path / "models/model_normalized.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    clean, _, _, _ = sample_vertex_groups(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        num_samples=args.surface_samples,
        num_vertex_samples=1024,
    )
    clean, _, _ = normalize_pc_with_params(clean.astype(np.float32, copy=False))
    noise_std = float(rng.uniform(args.noise_std_min, args.noise_std_max))
    noisy = clean + rng.laplace(0, noise_std, size=clean.shape).astype(np.float32)

    seed_idx = int(rng.integers(0, noisy.shape[0]))
    seed_point = noisy[seed_idx].astype(np.float32, copy=False)
    _, nn_idx = cKDTree(noisy).query(seed_point[None, :], k=args.patch_size)
    nn_idx = nn_idx[0]

    patch_noisy = (noisy[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    patch_clean = (clean[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    geom = pca_geometry(patch_clean)
    point_normal_var, point_surface_var = estimate_point_sharpness(
        patch_clean,
        seed=args.seed + candidate_idx,
    )
    risk = structure_risk(geom, point_normal_var, point_surface_var)
    return {
        "rel_path": rel_path,
        "category": rel_path.split("/")[1],
        "candidate_idx": candidate_idx,
        "seed_idx": seed_idx,
        "noise_std": noise_std,
        "patch_noisy": patch_noisy,
        "patch_clean": patch_clean,
        "point_normal_var": point_normal_var,
        "point_surface_var": point_surface_var,
        "structure_risk": risk,
        **geom,
    }


def select_diverse(candidates, count):
    selected = []
    used_rel_paths = set()
    for item in candidates:
        if item["rel_path"] in used_rel_paths:
            continue
        selected.append(item)
        used_rel_paths.add(item["rel_path"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        selected_ids = {id(v) for v in selected}
        for item in candidates:
            if id(item) in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= count:
                break
    return selected


def evaluate_patch(model, item):
    pc_noisy = jt.array(item["patch_noisy"][None, :, :])
    with jt.no_grad():
        pc_pred, _ = model.denoise_langevin_dynamics(pc_noisy)
    pred = pc_pred.detach().numpy()[0].astype(np.float32, copy=False)
    clean = item["patch_clean"]
    noisy = item["patch_noisy"]
    cd_noisy = chamfer_distance(noisy, clean)
    cd_pred = chamfer_distance(pred, clean)
    return {
        **item,
        "patch_pred": pred,
        "cd_noisy": cd_noisy,
        "cd_pred": cd_pred,
        "cd_delta": cd_pred - cd_noisy,
        "cd_score": metric_to_score(cd_pred, cd_noisy),
    }


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


def scatter_panel(ax, clean, other, title, color):
    pts = np.concatenate([clean, other], axis=0)
    set_axes_equal(ax, pts)
    ax.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=3, c="#222222", alpha=0.18)
    ax.scatter(other[:, 0], other[:, 1], other[:, 2], s=4, c=color, alpha=0.70)
    ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=10)


def render_item(item, out_path):
    clean = item["patch_clean"]
    noisy = item["patch_noisy"]
    pred = item["patch_pred"]
    fig = plt.figure(figsize=(13, 4.2))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    scatter_panel(ax1, clean, noisy, "Noisy vs clean", "#e15759")
    scatter_panel(ax2, clean, pred, "Pred vs clean", "#4e79a7")
    set_axes_equal(ax3, np.concatenate([clean, noisy, pred], axis=0))
    ax3.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=3, c="#222222", alpha=0.14)
    ax3.scatter(noisy[:, 0], noisy[:, 1], noisy[:, 2], s=3, c="#e15759", alpha=0.42)
    ax3.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=3, c="#4e79a7", alpha=0.55)
    ax3.view_init(elev=18, azim=35)
    ax3.set_title("Overlay", fontsize=10)
    fig.suptitle(
        f"{item['group']} | score={item['cd_score']:.2f} | delta={item['cd_delta']:.3g} | "
        f"risk={item['structure_risk']:.3f} | thin={item['thinness']:.3f}\n{item['rel_path']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def make_contact_sheet(image_paths, out_path, title):
    import matplotlib.image as mpimg

    cols = 2
    rows = int(math.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.0, rows * 4.6))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, image_paths):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def strip_arrays(row):
    return {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}


def summarize_group(rows):
    return {
        "count": len(rows),
        "mean_score": float(np.mean([r["cd_score"] for r in rows])),
        "median_score": float(np.median([r["cd_score"] for r in rows])),
        "worse_rate": float(np.mean([r["cd_delta"] > 0 for r in rows])),
        "mean_cd_delta": float(np.mean([r["cd_delta"] for r in rows])),
        "mean_noise_std": float(np.mean([r["noise_std"] for r in rows])),
        "mean_structure_risk": float(np.mean([r["structure_risk"] for r in rows])),
        "mean_thinness": float(np.mean([r["thinness"] for r in rows])),
        "mean_point_surface_var": float(np.mean([r["point_surface_var"] for r in rows])),
    }


def write_report(summary, rows, out_path):
    risk = summary["groups"]["risk"]
    control = summary["groups"]["control"]
    lines = [
        "# Extra Patch Verification",
        "",
        f"Candidates scanned: {summary['candidates_scanned']}",
        f"Datalist: `{summary['datalist']}`",
        f"Noise range: {summary['noise_std_min']} - {summary['noise_std_max']}",
        "",
        "## Result",
        "",
        "| Group | Count | Median score | Worse rate | Mean CD delta | Mean risk | Mean thinness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Risky geometry | {risk['count']} | {risk['median_score']:.2f} | "
            f"{100.0 * risk['worse_rate']:.1f}% | {risk['mean_cd_delta']:.6g} | "
            f"{risk['mean_structure_risk']:.3f} | {risk['mean_thinness']:.3f} |"
        ),
        (
            f"| Control geometry | {control['count']} | {control['median_score']:.2f} | "
            f"{100.0 * control['worse_rate']:.1f}% | {control['mean_cd_delta']:.6g} | "
            f"{control['mean_structure_risk']:.3f} | {control['mean_thinness']:.3f} |"
        ),
        "",
        "## Worst Extra Patches",
        "",
        "| Group | Category | Score | CD delta | Risk | Thinness | Rel path |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda r: r["cd_delta"], reverse=True)[:10]:
        lines.append(
            f"| {row['group']} | {row['category']} | {row['cd_score']:.2f} | "
            f"{row['cd_delta']:.6g} | {row['structure_risk']:.3f} | "
            f"{row['thinness']:.3f} | `{row['rel_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run deliberately picked new patches by geometry rather than reusing",
            "the previous worst-case list. It is intended as a small follow-up check,",
            "not as a full benchmark.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs1.1/checkpoints/vm/checkpoint_best.pkl")
    parser.add_argument("--mesh-root", default="E:/Code/competition2_EdgeConv/dataset_clean")
    parser.add_argument("--datalist", default="datalist/train.txt")
    parser.add_argument("--out-dir", default="outputs1.1/patch_diagnostics/hypothesis_verify")
    parser.add_argument("--candidates", type=int, default=120)
    parser.add_argument("--select", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--surface-samples", type=int, default=32768)
    parser.add_argument("--noise-std-min", type=float, default=0.005)
    parser.add_argument("--noise-std-max", type=float, default=0.008)
    parser.add_argument("--seed", type=int, default=20260526)
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

    candidates = []
    for i, rel_path in enumerate(chosen, 1):
        print(f"scan [{i}/{len(chosen)}] {rel_path}", flush=True)
        try:
            candidates.append(load_mesh_patch(rel_path, mesh_root, rng, args, i))
        except Exception as exc:
            print(f"skip {rel_path}: {exc}", flush=True)

    risk_candidates = sorted(candidates, key=lambda r: r["structure_risk"], reverse=True)
    control_candidates = sorted(candidates, key=lambda r: r["structure_risk"])
    selected = []
    for item in select_diverse(risk_candidates, args.select):
        item["group"] = "risk"
        selected.append(item)
    selected_ids = {id(v) for v in selected}
    for item in select_diverse([c for c in control_candidates if id(c) not in selected_ids], args.select):
        item["group"] = "control"
        selected.append(item)

    model = load_legacy_model(PROJECT_ROOT / args.checkpoint)
    records = []
    for i, item in enumerate(selected, 1):
        print(f"eval [{i}/{len(selected)}] {item['group']} {item['rel_path']}", flush=True)
        records.append(evaluate_patch(model, item))

    fieldnames = list(strip_arrays(records[0]).keys())
    with (out_dir / "verification_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(strip_arrays(row))

    image_paths_by_group = {"risk": [], "control": []}
    for rank, row in enumerate(sorted(records, key=lambda r: (r["group"], -r["cd_delta"])), 1):
        image_path = out_dir / f"{row['group']}_{rank:02d}.png"
        render_item(row, image_path)
        image_paths_by_group[row["group"]].append(image_path)
        np.savez_compressed(
            out_dir / f"{row['group']}_{rank:02d}.npz",
            clean=row["patch_clean"],
            noisy=row["patch_noisy"],
            denoised=row["patch_pred"],
        )
    make_contact_sheet(image_paths_by_group["risk"], out_dir / "risk_contact_sheet.png", "Risky geometry patches")
    make_contact_sheet(image_paths_by_group["control"], out_dir / "control_contact_sheet.png", "Control geometry patches")

    groups = {
        "risk": summarize_group([r for r in records if r["group"] == "risk"]),
        "control": summarize_group([r for r in records if r["group"] == "control"]),
    }
    summary = {
        "checkpoint": str((PROJECT_ROOT / args.checkpoint).resolve()),
        "mesh_root": str(mesh_root.resolve()),
        "datalist": str((PROJECT_ROOT / args.datalist).resolve()),
        "candidates_scanned": len(candidates),
        "selected_per_group": args.select,
        "patch_size": args.patch_size,
        "surface_samples": args.surface_samples,
        "noise_std_min": args.noise_std_min,
        "noise_std_max": args.noise_std_max,
        "seed": args.seed,
        "groups": groups,
        "worst": [
            strip_arrays(r)
            for r in sorted(records, key=lambda r: r["cd_delta"], reverse=True)[:10]
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, records, out_dir / "verification_report.md")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
