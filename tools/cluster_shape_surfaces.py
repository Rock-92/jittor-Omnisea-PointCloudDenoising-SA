import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.hard_patch_common import read_datalist  # noqa: E402
from tools.train_full_cloud_fusion_probe import load_shape, usable_paths  # noqa: E402


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


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros((size,), dtype=np.int8)

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(self, a, b):
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def estimate_normals(points, k):
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(int(k), points.shape[0]))
    normals = np.zeros_like(points, dtype=np.float32)
    curvature = np.zeros((points.shape[0],), dtype=np.float32)
    for i, nn_idx in enumerate(indices):
        local = points[np.asarray(nn_idx, dtype=np.int64)]
        centered = local - local.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(local.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        normal = eigvecs[:, 0]
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        normals[i] = normal.astype(np.float32)
        curvature[i] = float(eigvals[0] / max(float(eigvals.sum()), 1e-12))
    return normals, curvature, distances.astype(np.float32), indices.astype(np.int32)


def build_surface_edges(points, normals, neighbor_indices, args):
    uf = UnionFind(points.shape[0])
    cos_threshold = float(args.normal_cos)
    plane_threshold = float(args.plane_threshold)
    if plane_threshold <= 0.0:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=min(2, points.shape[0]))
        median_nn = float(np.median(distances[:, -1]))
        plane_threshold = max(
            float(args.min_plane_threshold),
            median_nn * float(args.plane_threshold_scale),
        )

    for i in range(points.shape[0]):
        pi = points[i]
        ni = normals[i]
        for j in neighbor_indices[i, 1:]:
            j = int(j)
            if j <= i:
                continue
            pj = points[j]
            nj = normals[j]
            normal_sim = abs(float(np.dot(ni, nj)))
            if normal_sim < cos_threshold:
                continue
            delta = pj - pi
            # Same local surface: both local tangent planes accept the neighbor.
            plane_i = abs(float(np.dot(delta, ni)))
            plane_j = abs(float(np.dot(delta, nj)))
            if max(plane_i, plane_j) > plane_threshold:
                continue
            uf.union(i, j)
    roots = np.asarray([uf.find(i) for i in range(points.shape[0])], dtype=np.int32)
    unique = {root: idx for idx, root in enumerate(sorted(set(roots.tolist())))}
    labels = np.asarray([unique[root] for root in roots], dtype=np.int32)
    return labels, plane_threshold


def cluster_shape_stats(points, normals, curvature, mask):
    local = points[mask]
    local_normals = normals[mask]
    if local.shape[0] == 0:
        return None
    centered = local - local.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(local.shape[0] - 1, 1)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0.0)[::-1]
    l1, l2, l3 = eigvals
    bbox = np.sort(local.max(axis=0) - local.min(axis=0))[::-1]
    normal_tensor = local_normals.T @ local_normals / max(local_normals.shape[0], 1)
    normal_var = float(1.0 - np.linalg.eigvalsh(normal_tensor)[-1])
    mean_normal = local_normals.mean(axis=0)
    mean_normal /= max(float(np.linalg.norm(mean_normal)), 1e-12)
    return {
        "count": int(mask.sum()),
        "curvature_mean": float(curvature[mask].mean()),
        "normal_var": normal_var,
        "linearity": float((l1 - l2) / max(float(l1), 1e-12)),
        "planarity": float((l2 - l3) / max(float(l1), 1e-12)),
        "scattering": float(l3 / max(float(l1), 1e-12)),
        "bbox_ratio_min": float(bbox[-1] / max(float(bbox[0]), 1e-12)),
        "bbox_ratio_mid": float(bbox[1] / max(float(bbox[0]), 1e-12)),
        "normal": mean_normal,
        "center": local.mean(axis=0),
    }


def is_valid_small_branch(stats, args):
    if stats["count"] < int(args.min_small_cluster_size):
        return False
    if stats["normal_var"] > float(args.small_branch_normal_var_max):
        return False
    if stats["scattering"] > float(args.small_branch_scattering_max):
        return False
    planar = stats["planarity"] >= float(args.small_branch_planarity_min)
    linear_thin = (
        stats["linearity"] >= float(args.small_branch_linearity_min)
        and stats["bbox_ratio_mid"] <= float(args.small_branch_mid_ratio_max)
    )
    return bool(planar or linear_thin)


def relabel_and_filter(points, normals, curvature, labels, args):
    counts = Counter(labels.tolist())
    valid = []
    small_valid = []
    stats_by_label = {}
    for label, count in sorted(counts.items()):
        mask = labels == label
        stats = cluster_shape_stats(points, normals, curvature, mask)
        stats_by_label[label] = stats
        if count >= int(args.min_cluster_size):
            valid.append(label)
        elif stats is not None and is_valid_small_branch(stats, args):
            valid.append(label)
            small_valid.append(label)

    new_labels = np.full_like(labels, -1)
    for new_label, old_label in enumerate(valid):
        new_labels[labels == old_label] = new_label
    noise_label = len(valid)
    new_labels[new_labels < 0] = noise_label

    rows = []
    for label in range(noise_label + 1):
        mask = new_labels == label
        stats = cluster_shape_stats(points, normals, curvature, mask)
        if stats is None:
            stats = {
                "count": 0,
                "curvature_mean": 0.0,
                "normal_var": 0.0,
                "linearity": 0.0,
                "planarity": 0.0,
                "scattering": 0.0,
                "bbox_ratio_min": 0.0,
                "bbox_ratio_mid": 0.0,
                "normal": np.zeros((3,), dtype=np.float32),
                "center": np.zeros((3,), dtype=np.float32),
            }
        old_label = valid[label] if label < len(valid) else -1
        rows.append(
            {
                "branch_id": int(label),
                "is_noise": int(label == noise_label),
                "is_small_valid": int(old_label in small_valid),
                "old_label": int(old_label),
                "count": int(stats["count"]),
                "fraction": float(mask.mean()),
                "curvature_mean": float(stats["curvature_mean"]),
                "normal_var": float(stats["normal_var"]),
                "linearity": float(stats["linearity"]),
                "planarity": float(stats["planarity"]),
                "scattering": float(stats["scattering"]),
                "bbox_ratio_min": float(stats["bbox_ratio_min"]),
                "bbox_ratio_mid": float(stats["bbox_ratio_mid"]),
                "normal_x": float(stats["normal"][0]),
                "normal_y": float(stats["normal"][1]),
                "normal_z": float(stats["normal"][2]),
                "center_x": float(stats["center"][0]),
                "center_y": float(stats["center"][1]),
                "center_z": float(stats["center"][2]),
            }
        )
    return new_labels, rows


def candidate_split_normals(points, normals, indices):
    local = points[indices]
    local_normals = normals[indices]
    candidates = []

    centered = local - local.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(local.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pca_normal = eigvecs[:, int(np.argmin(eigvals))]
    candidates.append(pca_normal)

    normal_tensor = local_normals.T @ local_normals / max(local_normals.shape[0], 1)
    _, normal_vecs = np.linalg.eigh(normal_tensor)
    candidates.append(normal_vecs[:, -1])

    mean_normal = local_normals.mean(axis=0)
    if np.linalg.norm(mean_normal) > 1e-6:
        candidates.append(mean_normal)

    unique = []
    for normal in candidates:
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            continue
        normal = normal / norm
        duplicate = False
        for existing in unique:
            if abs(float(np.dot(existing, normal))) > 0.98:
                duplicate = True
                break
        if not duplicate:
            unique.append(normal.astype(np.float32, copy=False))
    return unique


def best_height_gap_split(points, indices, normal, args):
    heights = points[indices] @ normal
    order = np.argsort(heights)
    sorted_heights = heights[order]
    gaps = np.diff(sorted_heights)
    if gaps.size == 0:
        return None

    min_side = max(
        int(args.min_cluster_size),
        int(round(indices.size * float(args.thin_split_min_fraction))),
    )
    if indices.size < 2 * min_side:
        return None

    lo = min_side - 1
    hi = gaps.size - min_side + 1
    if hi <= lo:
        return None
    valid_gaps = gaps[lo:hi]
    if valid_gaps.size == 0:
        return None

    local_best = int(np.argmax(valid_gaps))
    best = lo + local_best
    best_gap = float(gaps[best])
    if best_gap < float(args.thin_split_min_gap):
        return None

    left_count = best + 1
    right_count = indices.size - left_count
    if left_count < min_side or right_count < min_side:
        return None

    # Reject splits that only find a normal sampling gap inside one thick cloud.
    left_heights = sorted_heights[:left_count]
    right_heights = sorted_heights[left_count:]
    left_iqr = float(np.percentile(left_heights, 75) - np.percentile(left_heights, 25))
    right_iqr = float(np.percentile(right_heights, 75) - np.percentile(right_heights, 25))
    width = max(left_iqr + right_iqr, 1e-12)
    score = best_gap / width
    if score < float(args.thin_split_min_gap_score):
        return None

    return {
        "score": score,
        "gap": best_gap,
        "left": indices[order[:left_count]],
        "right": indices[order[left_count:]],
    }


def best_patch_height_gap_split(points, indices, args, reference_normal=None):
    if indices.size < max(2 * int(args.branch_refine_min_side), int(args.branch_refine_min_points)):
        return None
    local = points[indices]
    centered = local - local.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(local.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return None
    normal = normal / norm
    if reference_normal is not None and float(np.dot(normal, reference_normal)) < 0.0:
        normal = -normal

    heights = local @ normal
    order = np.argsort(heights)
    sorted_heights = heights[order]
    gaps = np.diff(sorted_heights)
    if gaps.size == 0:
        return None

    min_side = max(
        int(args.branch_refine_min_side),
        int(round(indices.size * float(args.branch_refine_min_fraction))),
    )
    if indices.size < 2 * min_side:
        return None
    lo = min_side - 1
    hi = gaps.size - min_side + 1
    if hi <= lo:
        return None

    best = lo + int(np.argmax(gaps[lo:hi]))
    best_gap = float(gaps[best])
    if best_gap < float(args.branch_refine_min_gap):
        return None

    left_heights = sorted_heights[: best + 1]
    right_heights = sorted_heights[best + 1:]
    left_iqr = float(np.percentile(left_heights, 75) - np.percentile(left_heights, 25))
    right_iqr = float(np.percentile(right_heights, 75) - np.percentile(right_heights, 25))
    score = best_gap / max(left_iqr + right_iqr, 1e-12)
    if score < float(args.branch_refine_min_gap_score):
        return None

    return {
        "score": score,
        "gap": best_gap,
        "normal": normal.astype(np.float32, copy=False),
        "left": indices[order[: best + 1]],
        "right": indices[order[best + 1:]],
    }


def refine_labels_by_local_patch_splits(points, normals, labels, valid_mask, args):
    if not bool(args.enable_cache_local_refine):
        return labels

    noise_label = int(labels.max())
    refined = labels.copy()
    next_label = noise_label + 1
    full_tree = cKDTree(points)
    min_points = max(
        int(args.branch_refine_min_points),
        2 * int(args.branch_refine_min_side),
    )

    for label in range(noise_label):
        branch_indices = np.flatnonzero((labels == label) & valid_mask).astype(np.int64)
        if branch_indices.size < max(min_points, int(args.cache_refine_patch_size)):
            continue

        reference_normal = normals[branch_indices].mean(axis=0)
        ref_norm = float(np.linalg.norm(reference_normal))
        if ref_norm > 1e-12:
            reference_normal = reference_normal / ref_norm
        else:
            reference_normal = None

        sample_count = min(int(args.cache_refine_samples), branch_indices.size)
        sample_positions = np.linspace(
            0,
            branch_indices.size - 1,
            num=sample_count,
            dtype=np.int64,
        )
        vote = np.zeros((branch_indices.size,), dtype=np.float32)
        local_pos = {int(point_index): pos for pos, point_index in enumerate(branch_indices)}
        accepted = 0
        patch_size = min(int(args.cache_refine_patch_size), points.shape[0])
        for position in sample_positions:
            center_index = int(branch_indices[position])
            _, nn_idx = full_tree.query(points[center_index], k=patch_size)
            nn_idx = np.asarray(nn_idx, dtype=np.int64).reshape(-1)
            same_branch = nn_idx[(labels[nn_idx] == label) & valid_mask[nn_idx]]
            split = best_patch_height_gap_split(
                points,
                same_branch.astype(np.int64, copy=False),
                args,
                reference_normal=reference_normal,
            )
            if split is None:
                continue
            accepted += 1
            for point_index in split["left"]:
                pos = local_pos.get(int(point_index))
                if pos is not None:
                    vote[pos] -= float(split["score"])
            for point_index in split["right"]:
                pos = local_pos.get(int(point_index))
                if pos is not None:
                    vote[pos] += float(split["score"])

        vote_rate = accepted / max(sample_count, 1)
        if vote_rate < float(args.cache_refine_vote_fraction):
            continue
        voted = np.abs(vote) >= float(args.cache_refine_min_vote)
        if voted.sum() < 2 * int(args.branch_refine_min_side):
            continue

        side = np.zeros((branch_indices.size,), dtype=np.int8)
        side[vote > 0.0] = 1
        side[vote < 0.0] = -1

        if np.any(side == 0):
            voted_points = branch_indices[side != 0]
            voted_side = side[side != 0]
            tree = cKDTree(points[voted_points])
            _, nearest = tree.query(points[branch_indices[side == 0]], k=1)
            side[side == 0] = voted_side[np.asarray(nearest, dtype=np.int64)]

        left = branch_indices[side < 0]
        right = branch_indices[side > 0]
        min_side = max(
            int(args.branch_refine_min_side),
            int(round(branch_indices.size * float(args.branch_refine_min_fraction))),
        )
        if left.size < min_side or right.size < min_side:
            continue

        refined[left] = next_label
        next_label += 1
        refined[right] = next_label
        next_label += 1

    if next_label == noise_label + 1:
        return labels

    relabeled = np.full_like(refined, next_label)
    out_label = 0
    for old_label in sorted(np.unique(refined).tolist()):
        if old_label == noise_label:
            continue
        mask = refined == old_label
        if not np.any(mask):
            continue
        relabeled[mask] = out_label
        out_label += 1
    relabeled[labels == noise_label] = out_label
    return relabeled


def split_cluster_by_height(points, normals, indices, args):
    if indices.size < max(args.min_cluster_size * 2, 2):
        return [indices]
    candidates = candidate_split_normals(points, normals, indices)
    splits = [
        split
        for normal in candidates
        for split in [best_height_gap_split(points, indices, normal, args)]
        if split is not None
    ]
    if not splits:
        return [indices]
    split = max(splits, key=lambda item: (item["score"], item["gap"]))
    return [split["left"], split["right"]]


def split_cluster_by_local_layers(points, normals, indices, args):
    if not args.enable_local_thin_split:
        return [indices]
    if indices.size < max(args.min_cluster_size * 2, 2):
        return [indices]

    local = points[indices]
    candidates = candidate_split_normals(points, normals, indices)
    best = None
    for normal in candidates:
        heights = local @ normal
        tangent = local - heights[:, None] * normal[None, :]
        tree = cKDTree(tangent)
        sample_count = min(
            int(args.local_thin_split_samples),
            indices.size,
        )
        if sample_count <= 0:
            continue
        if sample_count < indices.size:
            sample_idx = np.linspace(
                0,
                indices.size - 1,
                num=sample_count,
                dtype=np.int64,
            )
        else:
            sample_idx = np.arange(indices.size, dtype=np.int64)

        votes = []
        local_k = min(int(args.local_thin_split_k), indices.size)
        for center_idx in sample_idx:
            _, nn_idx = tree.query(tangent[center_idx], k=local_k)
            nn_idx = np.asarray(nn_idx, dtype=np.int64).reshape(-1)
            if nn_idx.size < max(2 * int(args.min_small_cluster_size), 8):
                continue
            split = best_height_gap_split(
                points,
                indices[nn_idx],
                normal,
                args,
            )
            if split is None:
                continue
            votes.append(split["gap"])
        vote_rate = len(votes) / max(sample_count, 1)
        if vote_rate < float(args.local_thin_split_vote_fraction):
            continue

        global_split = best_height_gap_split(points, indices, normal, args)
        if global_split is None:
            continue
        score = global_split["score"] * vote_rate
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "left": global_split["left"],
                "right": global_split["right"],
            }
    if best is None:
        return [indices]
    return [best["left"], best["right"]]


def split_one_part(points, normals, part, args):
    local_split = split_cluster_by_local_layers(points, normals, part, args)
    if len(local_split) > 1:
        return local_split
    return split_cluster_by_height(points, normals, part, args)


def split_thin_parallel_layers(points, normals, labels, args):
    if not args.enable_thin_split:
        return labels

    noise_label = int(labels.max())
    next_label = 0
    new_labels = np.full_like(labels, noise_label)
    for old_label in range(noise_label + 1):
        indices = np.flatnonzero(labels == old_label).astype(np.int64)
        if indices.size == 0:
            continue
        if old_label == noise_label:
            continue

        parts = [indices]
        for _ in range(max(0, int(args.thin_split_max_depth))):
            changed = False
            updated = []
            for part in parts:
                split = split_one_part(points, normals, part, args)
                if len(split) > 1:
                    changed = True
                updated.extend(split)
            parts = updated
            if not changed or len(parts) >= int(args.thin_split_max_parts):
                break

        for part in parts[: int(args.thin_split_max_parts)]:
            new_labels[part] = next_label
            next_label += 1

    new_noise_label = next_label
    new_labels[labels == noise_label] = new_noise_label
    return new_labels


def project(points, view):
    if view == "xy":
        return points[:, [0, 1]]
    if view == "xz":
        return points[:, [0, 2]]
    if view == "yz":
        return points[:, [1, 2]]
    raise ValueError(view)


def equal_axes_2d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius * 1.06, center[0] + radius * 1.06)
    ax.set_ylim(center[1] - radius * 1.06, center[1] + radius * 1.06)
    ax.set_aspect("equal", adjustable="box")


def equal_axes_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius * 1.06, center[0] + radius * 1.06)
    ax.set_ylim(center[1] - radius * 1.06, center[1] + radius * 1.06)
    ax.set_zlim(center[2] - radius * 1.06, center[2] + radius * 1.06)


def draw_clusters(path, points, labels, title, max_clusters):
    cmap = plt.get_cmap("tab20")
    colors = np.asarray([cmap(i % 20) for i in range(max(labels.max() + 1, 1))])
    point_colors = colors[np.clip(labels, 0, colors.shape[0] - 1)]
    noise_label = int(labels.max())
    point_colors[labels == noise_label] = (0.45, 0.45, 0.45, 0.25)

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=2,
        c=point_colors,
        alpha=0.75,
        depthshade=False,
        linewidths=0,
    )
    equal_axes_3d(ax3d, points)
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_title("3D clusters")

    for slot, view in zip([grid[0, 1], grid[1, 0], grid[1, 1]], ["xy", "xz", "yz"]):
        ax = fig.add_subplot(slot)
        pts2 = project(points, view)
        ax.scatter(
            pts2[:, 0],
            pts2[:, 1],
            s=2,
            c=point_colors,
            alpha=0.75,
            linewidths=0,
        )
        equal_axes_2d(ax, pts2)
        ax.grid(True, color="#d5d8de", linewidth=0.5, alpha=0.55)
        ax.set_title(view.upper())
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def shape_slug(rel_path):
    return "_".join(Path(rel_path).parts)


def cluster_one_shape(rel_path, args, rng, out_dir):
    shape = load_shape(
        rel_path,
        args.clean_root,
        args.mesh_root,
        args.num_points,
        rng,
        args.sample_missing_clean,
    )
    points = shape["clean"].astype(np.float32, copy=False)
    normals, curvature, _, nn_idx = estimate_normals(points, args.normal_k)
    _, _, _, graph_idx = estimate_normals(points, args.graph_k)
    labels, used_plane_threshold = build_surface_edges(
        points,
        normals,
        graph_idx,
        args,
    )
    labels, branch_rows = relabel_and_filter(
        points,
        normals,
        curvature,
        labels,
        args,
    )
    labels = split_thin_parallel_layers(points, normals, labels, args)
    labels, branch_rows = relabel_and_filter(
        points,
        normals,
        curvature,
        labels,
        args,
    )
    slug = shape_slug(rel_path)
    npz_path = out_dir / f"{slug}_surface_clusters.npz"
    image_path = out_dir / f"{slug}_surface_clusters.png"
    csv_path = out_dir / f"{slug}_surface_clusters.csv"
    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32, copy=False),
        normals=normals.astype(np.float32, copy=False),
        curvature=curvature.astype(np.float32, copy=False),
        labels=labels.astype(np.int32, copy=False),
        rel_path=np.asarray([rel_path]),
        plane_threshold=np.asarray([used_plane_threshold], dtype=np.float32),
    )
    write_csv(csv_path, branch_rows)
    draw_clusters(
        image_path,
        points,
        labels,
        (
            f"{rel_path}\n"
            f"branches={int(labels.max()) + 1}, "
            f"plane_thr={used_plane_threshold:.6f}, "
            f"normal_cos={args.normal_cos}"
        ),
        args.max_draw_clusters,
    )
    top_counts = Counter(labels.tolist()).most_common(12)
    return {
        "rel_path": rel_path,
        "point_count": int(points.shape[0]),
        "branch_count_including_noise": int(labels.max()) + 1,
        "noise_label": int(labels.max()),
        "noise_fraction": float((labels == labels.max()).mean()),
        "small_valid_branch_count": int(
            sum(row["is_small_valid"] for row in branch_rows)
        ),
        "mean_branch_curvature": float(
            np.mean(
                [
                    row["curvature_mean"]
                    for row in branch_rows
                    if not row["is_noise"]
                ]
            )
        ),
        "mean_branch_normal_var": float(
            np.mean(
                [
                    row["normal_var"]
                    for row in branch_rows
                    if not row["is_noise"]
                ]
            )
        ),
        "plane_threshold": float(used_plane_threshold),
        "npz": str(npz_path),
        "image": str(image_path),
        "csv": str(csv_path),
        "top_counts": [
            {"branch_id": int(label), "count": int(count)}
            for label, count in top_counts
        ],
    }


def cache_one_shape(rel_path, args, rng):
    output_path = Path(args.cache_root) / rel_path / args.cache_name
    if output_path.exists() and not args.overwrite:
        cache = np.load(output_path)
        labels = cache["labels"].astype(np.int32, copy=False)
        noise_label = int(cache["noise_label"][0])
        valid_mask = cache["valid_mask"].astype(np.bool_, copy=False)
        return {
            "rel_path": rel_path,
            "status": "skip",
            "cache": str(output_path),
            "branch_count_including_noise": int(cache["branch_count"][0]),
            "noise_fraction": float(cache["noise_fraction"][0]),
            "valid_fraction": float(valid_mask.mean()),
            "point_count": int(labels.shape[0]),
            "noise_label": noise_label,
        }
    shape = load_shape(
        rel_path,
        args.clean_root,
        args.mesh_root,
        args.num_points,
        rng,
        args.sample_missing_clean,
    )
    points = shape["clean"].astype(np.float32, copy=False)
    normals, curvature, _, _ = estimate_normals(points, args.normal_k)
    _, _, _, graph_idx = estimate_normals(points, args.graph_k)
    labels, used_plane_threshold = build_surface_edges(
        points,
        normals,
        graph_idx,
        args,
    )
    labels, branch_rows = relabel_and_filter(
        points,
        normals,
        curvature,
        labels,
        args,
    )
    labels = split_thin_parallel_layers(points, normals, labels, args)
    labels = refine_labels_by_local_patch_splits(
        points,
        normals,
        labels,
        labels != int(labels.max()),
        args,
    )
    labels, branch_rows = relabel_and_filter(
        points,
        normals,
        curvature,
        labels,
        args,
    )
    noise_label = int(labels.max())
    noise_fraction = float((labels == noise_label).mean())
    valid_mask = labels != noise_label
    if noise_fraction > float(args.max_noise_fraction):
        valid_mask[:] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        labels=labels.astype(np.int32, copy=False),
        normals=normals.astype(np.float32, copy=False),
        curvature=curvature.astype(np.float32, copy=False),
        valid_mask=valid_mask.astype(np.bool_, copy=False),
        noise_label=np.asarray([noise_label], dtype=np.int32),
        noise_fraction=np.asarray([noise_fraction], dtype=np.float32),
        branch_count=np.asarray([noise_label + 1], dtype=np.int32),
        plane_threshold=np.asarray([used_plane_threshold], dtype=np.float32),
    )
    return {
        "rel_path": rel_path,
        "status": "write",
        "cache": str(output_path),
        "branch_count_including_noise": noise_label + 1,
        "noise_fraction": noise_fraction,
        "valid_fraction": float(valid_mask.mean()),
        "small_valid_branch_count": int(
            sum(row["is_small_valid"] for row in branch_rows)
        ),
    }


def refine_one_cache(task):
    index, total, rel_path, args = task
    print(f"[{index + 1}/{total}] refine {rel_path}", flush=True)
    try:
        return refine_cache_file(rel_path, args)
    except Exception as exc:
        return {
            "rel_path": rel_path,
            "status": "error",
            "error": repr(exc),
        }


def refine_cache_file(rel_path, args):
    output_path = Path(args.cache_root) / rel_path / args.cache_name
    if not output_path.exists():
        return {
            "rel_path": rel_path,
            "status": "missing_cache",
            "cache": str(output_path),
        }
    clean_path = Path(args.clean_root) / rel_path / "clean.npy"
    if not clean_path.exists():
        return {
            "rel_path": rel_path,
            "status": "missing_clean",
            "cache": str(output_path),
            "clean": str(clean_path),
        }

    cache = np.load(output_path)
    labels = cache["labels"].astype(np.int32, copy=False)
    normals = cache["normals"].astype(np.float32, copy=False)
    curvature = cache["curvature"].astype(np.float32, copy=False)
    valid_mask = cache["valid_mask"].astype(np.bool_, copy=False)
    points = np.load(clean_path).astype(np.float32, copy=False)
    if points.shape[0] != labels.shape[0]:
        return {
            "rel_path": rel_path,
            "status": "shape_mismatch",
            "cache": str(output_path),
            "clean": str(clean_path),
            "point_count": int(points.shape[0]),
            "label_count": int(labels.shape[0]),
        }

    old_branch_count = int(cache["branch_count"][0])
    old_noise_fraction = float(cache["noise_fraction"][0])
    refined = refine_labels_by_local_patch_splits(
        points,
        normals,
        labels,
        valid_mask,
        args,
    )
    if np.array_equal(refined, labels) and not args.overwrite:
        return {
            "rel_path": rel_path,
            "status": "unchanged",
            "cache": str(output_path),
            "old_branch_count": old_branch_count,
            "new_branch_count": old_branch_count,
            "noise_fraction": old_noise_fraction,
            "valid_fraction": float(valid_mask.mean()),
        }

    noise_label = int(refined.max())
    noise_fraction = float((refined == noise_label).mean())
    valid_mask = refined != noise_label
    if noise_fraction > float(args.max_noise_fraction):
        valid_mask[:] = False

    np.savez_compressed(
        output_path,
        labels=refined.astype(np.int32, copy=False),
        normals=normals.astype(np.float32, copy=False),
        curvature=curvature.astype(np.float32, copy=False),
        valid_mask=valid_mask.astype(np.bool_, copy=False),
        noise_label=np.asarray([noise_label], dtype=np.int32),
        noise_fraction=np.asarray([noise_fraction], dtype=np.float32),
        branch_count=np.asarray([noise_label + 1], dtype=np.int32),
        plane_threshold=cache["plane_threshold"],
    )
    return {
        "rel_path": rel_path,
        "status": "write",
        "cache": str(output_path),
        "old_branch_count": old_branch_count,
        "new_branch_count": noise_label + 1,
        "noise_fraction": noise_fraction,
        "valid_fraction": float(valid_mask.mean()),
        "changed_points": int(np.sum(refined != labels)),
    }


def choose_paths(args, rng):
    if args.paths:
        return args.paths
    paths = usable_paths(
        read_datalists(args.datalist),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    laptop = [p for p in paths if args.laptop_category in Path(p).parts]
    non_laptop = [p for p in paths if p not in set(laptop)]
    rng.shuffle(laptop)
    rng.shuffle(non_laptop)
    selected = laptop[: args.max_laptop_shapes]
    selected += non_laptop[: max(0, args.max_shapes - len(selected))]
    return selected[: args.max_shapes]


def read_datalists(paths):
    rel_paths = []
    seen = set()
    for path in paths:
        for rel_path in read_datalist(path):
            if rel_path in seen:
                continue
            seen.add(rel_path)
            rel_paths.append(rel_path)
    return rel_paths


def process_one_cache(task):
    index, total, rel_path, args = task
    print(f"[{index + 1}/{total}] {rel_path}", flush=True)
    rng = np.random.default_rng(int(args.seed) + index)
    try:
        return cache_one_shape(rel_path, args, rng)
    except Exception as exc:
        return {
            "rel_path": rel_path,
            "status": "error",
            "error": repr(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist", nargs="+", default=["datalist/validate.txt"])
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--out-dir", default="outputs/surface_branch_clusters")
    parser.add_argument("--cache-root", default="cache_surface_branches")
    parser.add_argument("--cache-name", default="surface_branches.npz")
    parser.add_argument(
        "--mode",
        choices=["diagnose", "cache", "refine-cache"],
        default="diagnose",
    )
    parser.add_argument("--max-noise-fraction", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--paths", nargs="*", default=None)
    parser.add_argument("--max-shapes", type=int, default=5)
    parser.add_argument("--max-laptop-shapes", type=int, default=999999)
    parser.add_argument("--laptop-category", default="03642806")
    parser.add_argument("--num-points", type=int, default=12000)
    parser.add_argument("--normal-k", type=int, default=24)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--normal-cos", type=float, default=0.92)
    parser.add_argument("--plane-threshold", type=float, default=0.0)
    parser.add_argument("--plane-threshold-scale", type=float, default=0.55)
    parser.add_argument("--min-plane-threshold", type=float, default=0.0012)
    parser.add_argument("--min-cluster-size", type=int, default=80)
    parser.add_argument("--min-small-cluster-size", type=int, default=20)
    parser.add_argument("--small-branch-normal-var-max", type=float, default=0.22)
    parser.add_argument("--small-branch-scattering-max", type=float, default=0.08)
    parser.add_argument("--small-branch-planarity-min", type=float, default=0.18)
    parser.add_argument("--small-branch-linearity-min", type=float, default=0.45)
    parser.add_argument("--small-branch-mid-ratio-max", type=float, default=0.45)
    parser.add_argument("--enable-thin-split", action="store_true")
    parser.add_argument("--thin-split-min-gap", type=float, default=0.006)
    parser.add_argument("--thin-split-min-gap-score", type=float, default=0.35)
    parser.add_argument("--thin-split-min-fraction", type=float, default=0.08)
    parser.add_argument("--thin-split-max-depth", type=int, default=2)
    parser.add_argument("--thin-split-max-parts", type=int, default=8)
    parser.add_argument("--disable-local-thin-split", action="store_true")
    parser.add_argument("--local-thin-split-k", type=int, default=384)
    parser.add_argument("--local-thin-split-samples", type=int, default=96)
    parser.add_argument("--local-thin-split-vote-fraction", type=float, default=0.15)
    parser.add_argument("--enable-cache-local-refine", action="store_true")
    parser.add_argument("--cache-refine-patch-size", type=int, default=1000)
    parser.add_argument("--cache-refine-samples", type=int, default=256)
    parser.add_argument("--cache-refine-vote-fraction", type=float, default=0.15)
    parser.add_argument("--cache-refine-min-vote", type=float, default=0.35)
    parser.add_argument("--branch-refine-min-points", type=int, default=80)
    parser.add_argument("--branch-refine-min-side", type=int, default=20)
    parser.add_argument("--branch-refine-min-fraction", type=float, default=0.08)
    parser.add_argument("--branch-refine-min-gap", type=float, default=0.001)
    parser.add_argument("--branch-refine-min-gap-score", type=float, default=0.35)
    parser.add_argument("--max-draw-clusters", type=int, default=20)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()
    args.enable_local_thin_split = (
        bool(args.enable_thin_split) and not bool(args.disable_local_thin_split)
    )
    if args.mode == "refine-cache":
        args.enable_cache_local_refine = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    paths = choose_paths(args, rng)
    rows = []
    if args.mode == "refine-cache":
        tasks = [
            (index, len(paths), rel_path, args)
            for index, rel_path in enumerate(paths)
        ]
        if int(args.workers) > 1:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = [executor.submit(refine_one_cache, task) for task in tasks]
                for future in as_completed(futures):
                    rows.append(future.result())
        else:
            for task in tasks:
                rows.append(refine_one_cache(task))
    elif args.mode == "cache" and int(args.workers) > 1:
        tasks = [
            (index, len(paths), rel_path, args)
            for index, rel_path in enumerate(paths)
        ]
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(process_one_cache, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    else:
        for index, rel_path in enumerate(paths):
            print(f"[{index + 1}/{len(paths)}] {rel_path}", flush=True)
            if args.mode == "cache":
                rows.append(cache_one_shape(rel_path, args, rng))
                continue
            rows.append(cluster_one_shape(rel_path, args, rng, out_dir))
    summary_dir = out_dir if args.mode == "diagnose" else Path(args.cache_root)
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "summary.csv", rows)
    write_json(summary_dir / "summary.json", {"args": vars(args), "shapes": rows})
    print(json.dumps({"args": vars(args), "shapes": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
