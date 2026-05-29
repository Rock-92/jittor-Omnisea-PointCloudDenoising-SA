import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups


def read_datalist(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip().replace("\\", "/")
            if rel and not rel.startswith("#"):
                rows.append(rel)
    return rows


def normalize_pc(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2.0
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / (scale + 1e-12)).astype(np.float32, copy=False)


def load_mesh_points(mesh_path, num_samples, num_vertex_samples):
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    points, _, _, _ = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=num_samples,
        num_vertex_samples=num_vertex_samples,
    )
    return normalize_pc(points.astype(np.float32, copy=False))


def pca_features(points):
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
    l1, l2, l3 = eig
    denom = l3 + 1e-12
    trace = eig.sum() + 1e-12
    bbox = points.max(axis=0) - points.min(axis=0)
    bbox_sorted = np.sort(bbox)
    return {
        "linearity": float((l3 - l2) / denom),
        "planarity": float((l2 - l1) / denom),
        "scattering": float(l1 / denom),
        "curvature": float(l1 / trace),
        "anisotropy": float((l3 - l1) / denom),
        "bbox_small": float(bbox_sorted[0]),
        "bbox_mid": float(bbox_sorted[1]),
        "bbox_large": float(bbox_sorted[2]),
        "bbox_ratio_small": float(bbox_sorted[0] / (bbox_sorted[2] + 1e-12)),
        "bbox_ratio_mid": float(bbox_sorted[1] / (bbox_sorted[2] + 1e-12)),
    }


def local_curvature_features(points, sample_count=128, k=32):
    n = points.shape[0]
    sample_count = min(sample_count, n)
    k = min(k, n)
    idx = np.linspace(0, n - 1, sample_count, dtype=np.int64)
    query = points[idx]
    tree = cKDTree(points)
    _, nn_idx = tree.query(query, k=k)
    curv = []
    normals = []
    for inds in nn_idx:
        neigh = points[inds]
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        curv.append(float(vals[0] / (vals.sum() + 1e-12)))
        normals.append(vecs[:, 0])
    curv = np.asarray(curv, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    normal_mean = normals.mean(axis=0)
    normal_mean = normal_mean / (np.linalg.norm(normal_mean) + 1e-12)
    normal_var = 1.0 - np.abs(normals @ normal_mean)
    return {
        "local_curv_mean": float(curv.mean()),
        "local_curv_std": float(curv.std()),
        "local_curv_p90": float(np.percentile(curv, 90)),
        "local_curv_max": float(curv.max()),
        "normal_var_mean": float(normal_var.mean()),
        "normal_var_p90": float(np.percentile(normal_var, 90)),
    }


def geometry_feature(points):
    d = pca_features(points)
    d.update(local_curvature_features(points))
    names = list(d.keys())
    return np.asarray([d[name] for name in names], dtype=np.float32), names


def kmeans(x, k, seed, iters=100):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = x[mask].mean(axis=0)
            else:
                centers[i] = x[rng.integers(0, n)]
    return labels.astype(np.int64), centers.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_dir", default="../competition2_EdgeConv/dataset_clean")
    parser.add_argument("--datalist", nargs="+", default=["datalist/train.txt"])
    parser.add_argument("--output_dir", default="geometry_ssl_cache")
    parser.add_argument("--num_shapes", type=int, default=5000)
    parser.add_argument("--patches_per_shape", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=32768)
    parser.add_argument("--num_vertex_samples", type=int, default=1024)
    parser.add_argument("--num_geom_classes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    patches_path = out_dir / "patches_clean.npy"
    labels_path = out_dir / "labels.npy"
    features_path = out_dir / "geom_features.npy"
    if patches_path.exists() and labels_path.exists() and features_path.exists() and not args.overwrite:
        print(f"cache exists: {out_dir}")
        return

    rng = np.random.default_rng(args.seed)
    rels = []
    for path in args.datalist:
        rels.extend(read_datalist(path))
    rels = list(dict.fromkeys(rels))
    rng.shuffle(rels)
    if args.num_shapes > 0:
        rels = rels[: args.num_shapes]

    patches = []
    features = []
    sources = []
    feature_names = None
    for shape_idx, rel in enumerate(rels):
        mesh_path = Path(args.input_dataset_dir) / rel / "models" / "model_normalized.obj"
        if not mesh_path.exists():
            print(f"missing: {mesh_path}")
            continue
        try:
            pc = load_mesh_points(mesh_path, args.num_samples, args.num_vertex_samples)
        except Exception as exc:
            print(f"error: {mesh_path}: {exc!r}")
            continue
        tree = cKDTree(pc)
        seed_idx = rng.permutation(pc.shape[0])[: args.patches_per_shape]
        _, nn_idx = tree.query(pc[seed_idx], k=args.patch_size)
        for local_i, inds in enumerate(nn_idx):
            seed = pc[seed_idx[local_i]][None, :]
            patch = (pc[inds] - seed).astype(np.float32, copy=False)
            feat, names = geometry_feature(patch)
            if feature_names is None:
                feature_names = names
            patches.append(patch)
            features.append(feat)
            sources.append(
                {
                    "source": rel,
                    "seed_index": int(seed_idx[local_i]),
                    "shape_index": int(shape_idx),
                }
            )
        if (shape_idx + 1) % 50 == 0:
            print(f"processed {shape_idx + 1}/{len(rels)} shapes, patches={len(patches)}")

    if len(patches) < args.num_geom_classes:
        raise RuntimeError("not enough patches to cluster")

    patches_arr = np.stack(patches, axis=0).astype(np.float32, copy=False)
    features_arr = np.stack(features, axis=0).astype(np.float32, copy=False)
    mean = features_arr.mean(axis=0, keepdims=True)
    std = features_arr.std(axis=0, keepdims=True) + 1e-12
    features_z = (features_arr - mean) / std
    labels, centers = kmeans(features_z, args.num_geom_classes, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(patches_path, patches_arr)
    np.save(labels_path, labels)
    np.save(features_path, features_arr)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    datalist_rows = []
    for i in range(patches_arr.shape[0]):
        name = f"{i:08d}.npz"
        np.savez_compressed(sample_dir / name, patch=patches_arr[i], label=np.asarray(labels[i], dtype=np.int64))
        datalist_rows.append(f"samples/{name}\n")
    with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
        f.writelines(datalist_rows)
    with open(out_dir / "sources.json", "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)
    with open(out_dir / "kmeans.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_geom_classes": args.num_geom_classes,
                "feature_names": feature_names,
                "feature_mean": mean.reshape(-1).tolist(),
                "feature_std": std.reshape(-1).tolist(),
                "centers": centers.tolist(),
                "counts": np.bincount(labels, minlength=args.num_geom_classes).tolist(),
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved {len(patches)} patches to {out_dir}")
    print("cluster counts:", np.bincount(labels, minlength=args.num_geom_classes).tolist())


if __name__ == "__main__":
    main()
