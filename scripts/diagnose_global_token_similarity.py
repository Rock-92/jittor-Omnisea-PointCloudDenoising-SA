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

from src.data.utils import sample_vertex_groups
from src.model.feature import apply_point_linear
from src.model.vm import VelocityModule


def normalize_pc_with_params(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / scale).astype(np.float32, copy=False), center, float(scale)


def orientation_variation(normals, weights=None):
    if normals.shape[0] == 0:
        return math.nan
    if weights is None:
        weights = np.ones((normals.shape[0],), dtype=np.float64)
    weights = weights.astype(np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    tensor = np.zeros((3, 3), dtype=np.float64)
    for normal, weight in zip(normals, weights):
        n = normal.astype(np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        tensor += weight * np.outer(n, n)
    eigvals = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    return float(1.0 - eigvals[0])


def estimate_point_sharpness(points, k=24, max_points=500, seed=123):
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
    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0))[::-1]
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
        return math.nan, 0
    face_vertices = vertices[faces[selected]]
    normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    areas = np.linalg.norm(normals, axis=1) * 0.5
    valid = areas > 1e-12
    if not valid.any():
        return math.nan, int(selected.size)
    normals = normals[valid] / np.linalg.norm(normals[valid], axis=1, keepdims=True)
    return orientation_variation(normals, areas[valid]), int(selected.size)


def load_mesh(rel_path, mesh_root):
    mesh_path = mesh_root / rel_path / "models/model_normalized.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def sample_patch(rel_path, mesh_root, rng, args, candidate_idx):
    mesh = load_mesh(rel_path, mesh_root)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    clean, _, _, _ = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=args.surface_samples,
        num_vertex_samples=1024,
    )
    clean, center, scale = normalize_pc_with_params(clean.astype(np.float32, copy=False))
    vertices = ((vertices - center) / scale).astype(np.float32, copy=False)

    noise_std = float(rng.uniform(args.noise_std_min, args.noise_std_max))
    noisy = clean + rng.laplace(0, noise_std, size=clean.shape).astype(np.float32)
    seed_idx = int(rng.integers(0, noisy.shape[0]))
    seed_point = noisy[seed_idx].astype(np.float32, copy=False)
    _, nn_idx = cKDTree(noisy).query(seed_point[None, :], k=args.patch_size)
    nn_idx = nn_idx[0]

    patch_noisy = (noisy[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    patch_clean = (clean[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    mesh_local = (vertices - seed_point[None, :]).astype(np.float32, copy=False)
    patch_radius = float(np.sqrt((patch_noisy**2).sum(axis=1)).max())

    point_normal_var, point_surface_var = estimate_point_sharpness(
        patch_clean,
        seed=args.seed + candidate_idx,
    )
    mesh_normal_var, mesh_face_count = estimate_mesh_sharpness(
        mesh_local,
        faces,
        patch_radius,
    )
    geom = pca_geometry(patch_clean)
    sharp_score = float(
        np.nan_to_num(point_normal_var, nan=0.0)
        + 0.75 * np.nan_to_num(mesh_normal_var, nan=0.0)
        + 1.5 * np.nan_to_num(point_surface_var, nan=0.0)
    )
    straight_score = float(
        np.nan_to_num(geom["linearity"], nan=0.0)
        + 0.5 * np.nan_to_num(mesh_normal_var, nan=0.0)
        - 0.75 * np.nan_to_num(geom["scattering"], nan=0.0)
    )
    smooth_score = float(
        -np.nan_to_num(point_normal_var, nan=1.0)
        -np.nan_to_num(mesh_normal_var, nan=1.0)
        -2.0 * np.nan_to_num(point_surface_var, nan=1.0)
        +0.25 * np.nan_to_num(geom["planarity"], nan=0.0)
    )
    return {
        "rel_path": rel_path,
        "category": rel_path.split("/")[1] if "/" in rel_path else "",
        "candidate_idx": candidate_idx,
        "seed_idx": seed_idx,
        "noise_std": noise_std,
        "patch_radius": patch_radius,
        "patch_noisy": patch_noisy,
        "patch_clean": patch_clean,
        "point_normal_var": point_normal_var,
        "point_surface_var": point_surface_var,
        "mesh_normal_var": mesh_normal_var,
        "mesh_face_count": mesh_face_count,
        "sharp_score": sharp_score,
        "straight_score": straight_score,
        "smooth_score": smooth_score,
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
        selected_ids = {id(item) for item in selected}
        for item in candidates:
            if id(item) in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= count:
                break
    return selected


def choose_patch_groups(candidates, count):
    selected = []
    selected_keys = set()
    groups = [
        ("sharp", sorted(candidates, key=lambda r: r["sharp_score"], reverse=True)),
        ("straight", sorted(candidates, key=lambda r: r["straight_score"], reverse=True)),
        ("smooth", sorted(candidates, key=lambda r: r["smooth_score"], reverse=True)),
    ]
    for group, ranked in groups:
        pool = [
            item
            for item in ranked
            if (item["candidate_idx"], item["rel_path"], item["seed_idx"]) not in selected_keys
        ]
        for item in select_diverse(pool, count):
            key = (item["candidate_idx"], item["rel_path"], item["seed_idx"])
            item = dict(item)
            item["group"] = group
            selected.append(item)
            selected_keys.add(key)
    return selected


def load_model(checkpoint, config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_cfg = config["configs"]["model"]["config"]
    transform_cfg = config["configs"]["transform"]["config"]
    model = VelocityModule(model_cfg, transform_cfg)
    model.load(str(checkpoint))
    model.eval()
    return model


def extract_global_tokens(model, patch_noisy):
    encoder = model.encoder
    x = jt.array(patch_noisy[None, :, :])
    with jt.no_grad():
        feat = apply_point_linear(encoder.input_proj_1, x)
        feat = encoder.act(feat)
        feat = apply_point_linear(encoder.input_proj_2, feat)
        feat = encoder.act(feat)

        B, _, C = feat.shape
        generator = encoder.global_token_generator
        global_token = generator.global_token.broadcast((B, 1, C))
        tokens = jt.concat([global_token, feat], dim=1)

        outputs = {}
        for block_idx, block in enumerate(generator.blocks, start=1):
            tokens = block(tokens)
            outputs[f"block_{block_idx}"] = tokens[:, :1, :].detach().numpy()[0, 0]
    return outputs


def normalize_rows(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def pairwise_cosine(tokens):
    t = normalize_rows(tokens.astype(np.float64))
    return t @ t.T


def similarity_stats(tokens, labels):
    sim = pairwise_cosine(tokens)
    labels = np.asarray(labels)
    n = len(labels)
    same_values = []
    diff_values = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                same_values.append(float(sim[i, j]))
            else:
                diff_values.append(float(sim[i, j]))

    nn_correct = 0
    margins = []
    silhouettes = []
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        nn_correct += int(labels[int(np.argmax(row))] == labels[i])
        same_mask = labels == labels[i]
        same_mask[i] = False
        diff_mask = labels != labels[i]
        same_mean = float(np.mean(sim[i, same_mask])) if same_mask.any() else math.nan
        diff_best = float(np.max(sim[i, diff_mask])) if diff_mask.any() else math.nan
        margins.append(same_mean - diff_best)

        a = float(np.mean(1.0 - sim[i, same_mask])) if same_mask.any() else math.nan
        b_vals = []
        for label in sorted(set(labels)):
            if label == labels[i]:
                continue
            mask = labels == label
            b_vals.append(float(np.mean(1.0 - sim[i, mask])))
        b = min(b_vals) if b_vals else math.nan
        if np.isfinite(a) and np.isfinite(b) and max(a, b) > 1e-12:
            silhouettes.append((b - a) / max(a, b))

    return {
        "within_mean": float(np.mean(same_values)),
        "within_median": float(np.median(same_values)),
        "between_mean": float(np.mean(diff_values)),
        "between_median": float(np.median(diff_values)),
        "within_minus_between_mean": float(np.mean(same_values) - np.mean(diff_values)),
        "nearest_neighbor_accuracy": float(nn_correct / n),
        "mean_same_minus_nearest_other": float(np.mean(margins)),
        "silhouette_cosine_distance": float(np.mean(silhouettes)) if silhouettes else math.nan,
    }


def group_centroid_stats(tokens, labels):
    labels = np.asarray(labels)
    norm_tokens = normalize_rows(tokens.astype(np.float64))
    centroids = {}
    for label in sorted(set(labels)):
        centroid = norm_tokens[labels == label].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        centroids[label] = centroid
    rows = []
    for label_a, centroid_a in centroids.items():
        for label_b, centroid_b in centroids.items():
            rows.append({
                "label_a": label_a,
                "label_b": label_b,
                "cosine": float(np.dot(centroid_a, centroid_b)),
            })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_arrays(row):
    return {k: v for k, v in row.items() if not isinstance(v, np.ndarray)}


def render_similarity_heatmap(tokens, labels, names, title, out_path):
    sim = pairwise_cosine(tokens)
    order = sorted(range(len(labels)), key=lambda i: (labels[i], names[i]))
    sim = sim[np.ix_(order, order)]
    ordered_labels = [labels[i] for i in order]
    ordered_names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(title)
    tick_labels = [f"{label[0]}{idx + 1}" for idx, label in enumerate(ordered_labels)]
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    boundaries = []
    for i in range(1, len(ordered_labels)):
        if ordered_labels[i] != ordered_labels[i - 1]:
            boundaries.append(i - 0.5)
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=0.8)
        ax.axvline(boundary, color="white", linewidth=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_patch_contact_sheet(rows, out_path):
    cols = 4
    rows_count = int(math.ceil(len(rows) / cols))
    fig = plt.figure(figsize=(cols * 4.0, rows_count * 3.6))
    for idx, item in enumerate(rows, 1):
        ax = fig.add_subplot(rows_count, cols, idx, projection="3d")
        clean = item["patch_clean"]
        ax.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=3, c="#222222", alpha=0.75)
        mins = clean.min(axis=0)
        maxs = clean.max(axis=0)
        center = (mins + maxs) / 2
        radius = max(float((maxs - mins).max()) / 2, 1e-6)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=18, azim=35)
        ax.set_title(
            f"{item['group']} | pnv={item['point_normal_var']:.3f}\n"
            f"lin={item['linearity']:.3f} surf={item['point_surface_var']:.3f}",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs2.0/checkpoints/vm/checkpoint_best.pkl")
    parser.add_argument("--run-config", default="outputs2.0/runs/train/20260526_071257/config.json")
    parser.add_argument("--mesh-root", default="E:/Code/competition2_EdgeConv/dataset_clean")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default="outputs2.0/patch_diagnostics/global_token_similarity")
    parser.add_argument("--candidates", type=int, default=90)
    parser.add_argument("--select-per-group", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--surface-samples", type=int, default=32768)
    parser.add_argument("--noise-std-min", type=float, default=0.005)
    parser.add_argument("--noise-std-max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--use-cuda", type=int, default=1)
    args = parser.parse_args()

    jt.flags.use_cuda = int(args.use_cuda)
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
            candidates.append(sample_patch(rel_path, mesh_root, rng, args, i))
        except Exception as exc:
            print(f"skip {rel_path}: {exc}", flush=True)

    selected = choose_patch_groups(candidates, args.select_per_group)
    write_csv([strip_arrays(row) for row in selected], out_dir / "selected_patches.csv")
    render_patch_contact_sheet(selected, out_dir / "selected_patch_contact_sheet.png")

    model = load_model(PROJECT_ROOT / args.checkpoint, PROJECT_ROOT / args.run_config)
    token_rows = []
    tokens_by_block = {}
    for i, item in enumerate(selected, 1):
        print(f"token [{i}/{len(selected)}] {item['group']} {item['rel_path']}", flush=True)
        tokens = extract_global_tokens(model, item["patch_noisy"])
        for block_name, token in tokens.items():
            tokens_by_block.setdefault(block_name, []).append(token)
            token_rows.append({
                "patch_index": i - 1,
                "group": item["group"],
                "rel_path": item["rel_path"],
                "block": block_name,
                "token_norm": float(np.linalg.norm(token)),
                "token_mean": float(token.mean()),
                "token_std": float(token.std()),
            })
    write_csv(token_rows, out_dir / "token_basic_stats.csv")

    labels = [row["group"] for row in selected]
    names = [row["rel_path"] for row in selected]
    summary_by_block = {}
    centroid_rows = []
    for block_name, token_list in sorted(tokens_by_block.items()):
        token_arr = np.stack(token_list, axis=0)
        np.save(out_dir / f"{block_name}_tokens.npy", token_arr)
        stats = similarity_stats(token_arr, labels)
        summary_by_block[block_name] = stats
        centroid_rows.extend(
            {"block": block_name, **row}
            for row in group_centroid_stats(token_arr, labels)
        )
        render_similarity_heatmap(
            token_arr,
            labels,
            names,
            f"{block_name} global token cosine similarity",
            out_dir / f"{block_name}_similarity_heatmap.png",
        )
    write_csv(centroid_rows, out_dir / "group_centroid_similarity.csv")

    group_counts = {
        group: int(sum(1 for label in labels if label == group))
        for group in sorted(set(labels))
    }
    summary = {
        "checkpoint": str((PROJECT_ROOT / args.checkpoint).resolve()),
        "run_config": str((PROJECT_ROOT / args.run_config).resolve()),
        "mesh_root": str(mesh_root.resolve()),
        "datalist": str((PROJECT_ROOT / args.datalist).resolve()),
        "candidates_scanned": len(candidates),
        "selected_count": len(selected),
        "group_counts": group_counts,
        "patch_size": args.patch_size,
        "surface_samples": args.surface_samples,
        "seed": args.seed,
        "interpretation": (
            "Good class-like global tokens should show within_mean higher than between_mean, "
            "positive silhouette_cosine_distance, and high nearest_neighbor_accuracy."
        ),
        "blocks": summary_by_block,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
