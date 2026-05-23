#!/usr/bin/env python
"""Point cloud denoising evaluation.

Metrics:
    CD:  Chamfer Distance between denoised and clean point clouds.
    P2S: Point-to-Surface distance from denoised points to the mesh surface.
"""

import argparse
import glob
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=RuntimeWarning, module="point_cloud_utils")

try:
    import point_cloud_utils as pcu

    HAS_PCU = True
except ImportError:
    HAS_PCU = False


def load_pointcloud(path):
    """Load a point cloud from .npy or .xyz, returning float64 array (N, 3)."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)

    points = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points, dtype=np.float64)


def load_mesh_vf(path):
    """Load mesh vertices and faces from an obj file."""
    if not os.path.exists(path):
        return None, None
    if HAS_PCU:
        vertices, faces = pcu.load_mesh_vf(path)
        return vertices.astype(np.float64), faces.astype(np.int32)

    try:
        import trimesh
    except ImportError:
        return None, None

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)


def normalize_to_unit_sphere(pc):
    """Normalize point cloud to a unit sphere and return pc, center, scale."""
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    pc_centered = pc - center
    scale = np.sqrt((pc_centered**2).sum(axis=1)).max()
    if scale < 1e-12:
        return pc_centered, center, scale
    return pc_centered / scale, center, scale


def chamfer_distance(pc_a, pc_b, normalize=True):
    """Compute bidirectional squared Chamfer Distance."""
    if normalize:
        pc_b, center, scale = normalize_to_unit_sphere(pc_b)
        if scale < 1e-12:
            return 0.0
        pc_a = (pc_a - center) / scale

    tree_b = cKDTree(pc_b)
    dist_a2b, _ = tree_b.query(pc_a, k=1)

    tree_a = cKDTree(pc_a)
    dist_b2a, _ = tree_a.query(pc_b, k=1)

    return float((dist_a2b**2).mean() + (dist_b2a**2).mean())


def point_to_surface_distance(pc, mesh_v, mesh_f, normalize_ref_pc=None):
    """Compute mean squared point-to-surface distance."""
    if mesh_v is None or mesh_f is None:
        return None

    vertices = mesh_v.copy()
    if normalize_ref_pc is not None:
        center = (normalize_ref_pc.max(axis=0) + normalize_ref_pc.min(axis=0)) / 2.0
        centered = normalize_ref_pc - center
        scale = np.sqrt((centered**2).sum(axis=1)).max()
        if scale < 1e-12:
            return 0.0
        pc = (pc - center) / scale
        vertices = (vertices - center) / scale

    if HAS_PCU:
        dists, _, _ = pcu.closest_points_on_mesh(
            pc.astype(np.float32),
            vertices.astype(np.float32),
            mesh_f,
        )
        return float((dists**2).mean())

    tree = cKDTree(vertices)
    dists, _ = tree.query(pc, k=1)
    return float((dists**2).mean())


def metric_to_score(val_pred, val_noisy):
    """Map a metric improvement to [0, 100]."""
    if val_noisy < 1e-15:
        return 100.0 if val_pred < 1e-15 else 0.0
    score = 100.0 * (1.0 - val_pred / val_noisy)
    return max(0.0, min(100.0, score))


def find_samples(base_dir, filename):
    """Return {relative_sample_key: file_path} for matching files."""
    samples = {}
    pattern = os.path.join(base_dir, "**", filename)
    for path in sorted(glob.glob(pattern, recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), base_dir)
        samples[rel] = path
    return samples


def find_meshes(mesh_dir, data_name="models/model_normalized.obj"):
    """Return {relative_sample_key: mesh_path} for mesh files."""
    meshes = {}
    pattern = os.path.join(mesh_dir, "**", data_name)
    for path in sorted(glob.glob(pattern, recursive=True)):
        parent = path
        for _ in data_name.split("/"):
            parent = os.path.dirname(parent)
        rel = os.path.relpath(parent, mesh_dir)
        meshes[rel] = path
    return meshes


def evaluate_single(args_tuple):
    """Evaluate one sample and return metrics and scores."""
    key, pred_path, gt_path, noisy_path, mesh_path = args_tuple

    pc_pred = load_pointcloud(pred_path)
    pc_gt = load_pointcloud(gt_path)
    pc_noisy = load_pointcloud(noisy_path)

    cd_pred = chamfer_distance(pc_pred, pc_gt, normalize=True)
    cd_noisy = chamfer_distance(pc_noisy, pc_gt, normalize=True)
    cd_score = metric_to_score(cd_pred, cd_noisy)

    p2s_pred = None
    p2s_noisy = None
    p2s_score = None
    if mesh_path is not None:
        mesh_v, mesh_f = load_mesh_vf(mesh_path)
        if mesh_v is not None:
            p2s_pred = point_to_surface_distance(
                pc_pred,
                mesh_v,
                mesh_f,
                normalize_ref_pc=pc_gt,
            )
            p2s_noisy = point_to_surface_distance(
                pc_noisy,
                mesh_v,
                mesh_f,
                normalize_ref_pc=pc_gt,
            )
            if p2s_pred is not None and p2s_noisy is not None:
                p2s_score = metric_to_score(p2s_pred, p2s_noisy)

    return key, cd_pred, cd_noisy, cd_score, p2s_pred, p2s_noisy, p2s_score


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate point cloud denoising results.")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--noisy_dir", type=str, required=True)
    parser.add_argument("--mesh_dir", type=str, default="")
    parser.add_argument("--mesh_data_name", type=str, default="models/model_normalized.obj")
    parser.add_argument("--pred_filename", type=str, default="denoised.npy")
    parser.add_argument("--gt_filename", type=str, default="clean.npy")
    parser.add_argument("--noisy_filename", type=str, default="noisy.npy")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    use_p2s = bool(args.mesh_dir)
    if use_p2s and not HAS_PCU:
        print("point-cloud-utils is not installed; P2S will use vertex approximation.")

    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 16)

    pred_samples = find_samples(args.pred_dir, args.pred_filename)
    gt_samples = find_samples(args.gt_dir, args.gt_filename)
    noisy_samples = find_samples(args.noisy_dir, args.noisy_filename)
    mesh_samples = find_meshes(args.mesh_dir, args.mesh_data_name) if use_p2s else {}

    common_keys = sorted(set(pred_samples) & set(gt_samples) & set(noisy_samples))
    if not common_keys:
        print("No matched samples found.")
        print(f"pred_dir samples: {len(pred_samples)}")
        print(f"gt_dir samples: {len(gt_samples)}")
        print(f"noisy_dir samples: {len(noisy_samples)}")
        sys.exit(1)

    missing_pred = set(gt_samples) - set(pred_samples)
    if missing_pred:
        print(f"Warning: {len(missing_pred)} samples are missing predictions.")

    tasks = []
    for key in common_keys:
        mesh_path = mesh_samples.get(key) if use_p2s else None
        tasks.append((key, pred_samples[key], gt_samples[key], noisy_samples[key], mesh_path))

    print(
        f"Backend: CD=scipy.cKDTree, "
        f"P2S={'pcu' if HAS_PCU else 'cKDTree vertex approx'}, workers={n_workers}"
    )
    print(f"Evaluating {len(tasks)} samples...")
    start = time.time()

    if n_workers > 1 and len(tasks) > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(evaluate_single, tasks)
    else:
        results = [evaluate_single(task) for task in tasks]

    cd_scores = []
    p2s_scores = []
    cd_preds = []
    cd_noisys = []
    p2s_preds = []
    p2s_noisys = []

    for key, cd_pred, cd_noisy, cd_score, p2s_pred, p2s_noisy, p2s_score in results:
        cd_scores.append(cd_score)
        cd_preds.append(cd_pred)
        cd_noisys.append(cd_noisy)
        if p2s_score is not None:
            p2s_scores.append(p2s_score)
            p2s_preds.append(p2s_pred)
            p2s_noisys.append(p2s_noisy)
        if args.verbose:
            msg = f"{key}: CD_score={cd_score:.2f}"
            if p2s_score is not None:
                msg += f", P2S_score={p2s_score:.2f}"
            print(msg)

    for _ in missing_pred:
        cd_scores.append(0.0)
        if use_p2s:
            p2s_scores.append(0.0)

    total_samples = len(common_keys) + len(missing_pred)
    mean_cd_score = float(np.mean(cd_scores)) if cd_scores else 0.0
    has_p2s = len(p2s_scores) > 0
    mean_p2s_score = float(np.mean(p2s_scores)) if has_p2s else 0.0
    final_score = (
        0.5 * mean_cd_score + 0.5 * mean_p2s_score if has_p2s else mean_cd_score
    )

    elapsed = time.time() - start
    print("=" * 65)
    print("Point Cloud Denoising Evaluation")
    print("=" * 65)
    print(f"Total samples:       {total_samples}")
    print(f"Valid predictions:   {len(common_keys)}")
    print(f"Missing predictions: {len(missing_pred)}")
    print(f"Workers:             {n_workers}")
    print(f"Elapsed:             {elapsed:.1f}s")
    print("-" * 65)
    if cd_preds:
        print(f"Mean CD_pred:        {np.mean(cd_preds):.8f}")
        print(f"Mean CD_noisy:       {np.mean(cd_noisys):.8f}")
    print(f"CD score:            {mean_cd_score:.2f} / 100.00")
    if has_p2s:
        print(f"Mean P2S_pred:       {np.mean(p2s_preds):.8f}")
        print(f"Mean P2S_noisy:      {np.mean(p2s_noisys):.8f}")
        print(f"P2S score:           {mean_p2s_score:.2f} / 100.00")
        print("-" * 65)
        print(f"Final score:         {final_score:.2f} / 100.00")
    else:
        print("-" * 65)
        print(f"Final score (CD):    {final_score:.2f} / 100.00")
    print("=" * 65)
    return final_score


if __name__ == "__main__":
    main()
