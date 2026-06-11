import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups  # noqa: E402
from tools.hard_patch_common import normalize_pc, read_datalist  # noqa: E402


def category_of(rel_path):
    parts = Path(str(rel_path)).parts
    return parts[1] if len(parts) > 1 else parts[0]


def choose_paths(rel_paths, num_shapes, rng, category_reference=None):
    rel_paths = list(rel_paths)
    sample_count = min(int(num_shapes), len(rel_paths))
    if sample_count == 0:
        return []
    if not category_reference:
        return rng.choice(
            np.asarray(rel_paths),
            size=sample_count,
            replace=False,
        ).tolist()

    available_counts = Counter(category_of(path) for path in rel_paths)
    reference_counts = Counter(
        category_of(path) for path in category_reference
    )
    category_mass = {
        category: reference_counts[category]
        for category in available_counts
        if reference_counts[category] > 0
    }
    if not category_mass:
        raise ValueError(
            "category reference has no categories available in the split"
        )
    weights = np.asarray(
        [
            category_mass.get(category_of(path), 0.0)
            / available_counts[category_of(path)]
            for path in rel_paths
        ],
        dtype=np.float64,
    )
    positive = np.flatnonzero(weights > 0.0)
    if positive.size < sample_count:
        sample_count = int(positive.size)
    weights /= weights.sum()
    indices = rng.choice(
        np.arange(len(rel_paths)),
        size=sample_count,
        replace=False,
        p=weights,
    )
    return [rel_paths[int(index)] for index in indices]


def sample_shape(mesh_path):
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    clean, _, _, _ = sample_vertex_groups(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        num_samples=32768,
        num_vertex_samples=1024,
    )
    clean = clean.astype(np.float32, copy=False)
    p_max = clean.max(axis=0)
    p_min = clean.min(axis=0)
    center = (p_max + p_min) / 2
    centered = clean - center
    scale = np.sqrt((centered ** 2.0).sum(axis=1).max()).max()
    clean = (centered / max(float(scale), 1e-12)).astype(
        np.float32,
        copy=False,
    )
    return clean, center.astype(np.float32), np.float32(scale)


def build_split(
    rel_paths,
    mesh_root,
    num_shapes,
    patches_per_shape,
    patch_size,
    noise_std,
    noise_std_min,
    noise_std_max,
    rng,
    category_reference=None,
):
    chosen = choose_paths(
        rel_paths,
        num_shapes,
        rng,
        category_reference=category_reference,
    )
    noisy_patches = []
    clean_patches = []
    patch_paths = []
    seed_indices = []
    patch_centers = []
    normalize_centers = []
    normalize_scales = []
    patch_sigmas = []
    for shape_index, rel_path in enumerate(chosen):
        print(
            f"[{shape_index + 1}/{len(chosen)}] {rel_path}",
            flush=True,
        )
        clean, normalize_center, normalize_scale = sample_shape(
            Path(mesh_root) / rel_path / "models/model_normalized.obj"
        )
        if noise_std_min is not None or noise_std_max is not None:
            lower = (
                float(noise_std)
                if noise_std_min is None
                else float(noise_std_min)
            )
            upper = (
                float(noise_std)
                if noise_std_max is None
                else float(noise_std_max)
            )
            if lower > upper:
                raise ValueError("noise-std-min must be <= noise-std-max")
            shape_noise_std = float(rng.uniform(lower, upper))
        else:
            shape_noise_std = float(noise_std)
        noisy = clean + (
            rng.standard_normal(clean.shape) * shape_noise_std
        ).astype(np.float32)
        seeds = rng.choice(
            np.arange(noisy.shape[0]),
            size=int(patches_per_shape),
            replace=False,
        )
        tree = cKDTree(noisy)
        _, neighbor_indices = tree.query(
            noisy[seeds],
            k=min(int(patch_size), noisy.shape[0]),
        )
        for seed, neighbor_index in zip(seeds, neighbor_indices):
            center = noisy[int(seed)]
            noisy_patches.append(
                (noisy[neighbor_index] - center).astype(
                    np.float32,
                    copy=False,
                )
            )
            clean_patches.append(
                (clean[neighbor_index] - center).astype(
                    np.float32,
                    copy=False,
                )
            )
            patch_paths.append(str(rel_path))
            seed_indices.append(int(seed))
            patch_centers.append(center.astype(np.float32, copy=False))
            normalize_centers.append(normalize_center)
            normalize_scales.append(normalize_scale)
            patch_sigmas.append(shape_noise_std)
    return {
        "pc_noisy": np.stack(noisy_patches),
        "pc_clean": np.stack(clean_patches),
        "score_sigma": np.asarray(
            patch_sigmas,
            dtype=np.float32,
        ).reshape(-1, 1),
        "rel_path": np.asarray(patch_paths),
        "seed_idx": np.asarray(seed_indices, dtype=np.int64),
        "patch_center": np.stack(patch_centers).astype(
            np.float32,
            copy=False,
        ),
        "normalize_center": np.stack(normalize_centers).astype(
            np.float32,
            copy=False,
        ),
        "normalize_scale": np.asarray(
            normalize_scales,
            dtype=np.float32,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--train-datalist", default="datalist/train.txt")
    parser.add_argument("--val-datalist", default="datalist/validate.txt")
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs_result/outputs_analysis/"
            "multistage_refinement_probe_dataset"
        ),
    )
    parser.add_argument("--train-shapes", type=int, default=80)
    parser.add_argument("--train-patches-per-shape", type=int, default=8)
    parser.add_argument("--val-shapes", type=int, default=40)
    parser.add_argument("--val-patches-per-shape", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.020)
    parser.add_argument("--noise-std-min", type=float, default=None)
    parser.add_argument("--noise-std-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--only-val", action="store_true")
    parser.add_argument(
        "--train-category-reference-datalist",
        default=None,
    )
    parser.add_argument(
        "--val-category-reference-datalist",
        default=None,
    )
    parser.add_argument(
        "--exclude-train-dataset",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--exclude-val-dataset",
        action="append",
        default=[],
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mesh_root = Path(args.mesh_root)
    train_paths = [
        rel for rel in read_datalist(args.train_datalist)
        if (mesh_root / rel / "models/model_normalized.obj").exists()
    ]
    val_paths = [
        rel for rel in read_datalist(args.val_datalist)
        if (mesh_root / rel / "models/model_normalized.obj").exists()
    ]
    train_category_reference = (
        read_datalist(args.train_category_reference_datalist)
        if args.train_category_reference_datalist
        else None
    )
    val_category_reference = (
        read_datalist(args.val_category_reference_datalist)
        if args.val_category_reference_datalist
        else None
    )
    if args.exclude_train_dataset:
        excluded_paths = set()
        for excluded_dataset in args.exclude_train_dataset:
            excluded = np.load(excluded_dataset, allow_pickle=True)
            excluded_paths.update(excluded["rel_path"].tolist())
        train_paths = [rel for rel in train_paths if rel not in excluded_paths]
    if args.exclude_val_dataset:
        excluded_paths = set()
        for excluded_dataset in args.exclude_val_dataset:
            excluded = np.load(excluded_dataset, allow_pickle=True)
            excluded_paths.update(excluded["rel_path"].tolist())
        val_paths = [rel for rel in val_paths if rel not in excluded_paths]
    overlap = set(train_paths) & set(val_paths)
    if overlap and not args.only_val:
        raise ValueError(f"train/val shape overlap: {len(overlap)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = None
    if not args.only_val:
        train = build_split(
            train_paths,
            mesh_root,
            args.train_shapes,
            args.train_patches_per_shape,
            args.patch_size,
            args.noise_std,
            args.noise_std_min,
            args.noise_std_max,
            rng,
            category_reference=train_category_reference,
        )
    val = build_split(
        val_paths,
        mesh_root,
        args.val_shapes,
        args.val_patches_per_shape,
        args.patch_size,
        args.noise_std,
        args.noise_std_min,
        args.noise_std_max,
        rng,
        category_reference=val_category_reference,
    )
    if train is not None:
        np.savez_compressed(out_dir / "train_patches.npz", **train)
    np.savez_compressed(out_dir / "val_patches.npz", **val)
    summary = {
        "train_patches": (
            int(train["pc_noisy"].shape[0]) if train is not None else 0
        ),
        "train_shapes": (
            int(len(set(train["rel_path"].tolist())))
            if train is not None else 0
        ),
        "val_patches": int(val["pc_noisy"].shape[0]),
        "val_shapes": int(len(set(val["rel_path"].tolist()))),
        "shape_overlap": 0,
        "patch_size": int(args.patch_size),
        "noise_std": float(args.noise_std),
        "noise_std_min": args.noise_std_min,
        "noise_std_max": args.noise_std_max,
        "train_category_counts": (
            dict(Counter(
                category_of(path)
                for path in set(train["rel_path"].tolist())
            ))
            if train is not None else {}
        ),
        "val_category_counts": dict(
            Counter(
                category_of(path)
                for path in set(val["rel_path"].tolist())
            )
        ),
        "seed": int(args.seed),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
