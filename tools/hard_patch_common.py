import csv
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np
import trimesh
from omegaconf import OmegaConf
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import metric_to_score  # noqa: E402
from src.data.utils import sample_vertex_groups  # noqa: E402
from src.model.parse import get_model  # noqa: E402


def read_datalist(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def normalize_pc(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / max(float(scale), 1e-12)).astype(np.float32, copy=False)


def chamfer(pc_a, pc_b):
    tree_b = cKDTree(pc_b)
    dist_a2b, _ = tree_b.query(pc_a, k=1)
    tree_a = cKDTree(pc_a)
    dist_b2a, _ = tree_a.query(pc_b, k=1)
    return float((dist_a2b**2).mean() + (dist_b2a**2).mean())


def load_model(checkpoint, model_config=None, transform_config=None):
    if model_config is None:
        model_config = PROJECT_ROOT / "configs/model/vm.yaml"
    if transform_config is None:
        transform_config = PROJECT_ROOT / "configs/transform/vm.yaml"
    model_cfg = OmegaConf.to_container(OmegaConf.load(model_config), resolve=True)
    transform_cfg = OmegaConf.to_container(OmegaConf.load(transform_config), resolve=True)
    model = get_model(model_config=model_cfg, transform_config=transform_cfg)
    model.load(str(checkpoint))
    return model


def sample_clean_from_mesh(rel_path, mesh_root):
    mesh_path = Path(mesh_root) / rel_path / "models/model_normalized.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    clean, _, _, _ = sample_vertex_groups(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        num_samples=32768,
        num_vertex_samples=1024,
    )
    return normalize_pc(clean.astype(np.float32, copy=False))


def sample_patch(rel_path, mesh_root, rng, patch_size=1000, noise_std=0.020):
    clean = sample_clean_from_mesh(rel_path, mesh_root)
    noisy = clean + (rng.standard_normal(clean.shape) * noise_std).astype(np.float32)
    seed_idx = int(rng.integers(0, noisy.shape[0]))
    seed_point = noisy[seed_idx]
    _, nn_idx = cKDTree(noisy).query(
        seed_point[None, :],
        k=min(int(patch_size), noisy.shape[0]),
    )
    nn_idx = np.asarray(nn_idx).reshape(-1)
    patch_noisy = (noisy[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    patch_clean = (clean[nn_idx] - seed_point[None, :]).astype(np.float32, copy=False)
    return {
        "rel_path": rel_path,
        "seed_idx": seed_idx,
        "noise_std": float(noise_std),
        "patch_noisy": patch_noisy,
        "patch_clean": patch_clean,
    }


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
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
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


def evaluate_patch(model, patch, mode="heun", sigma=0.020):
    noisy = patch["patch_noisy"]
    clean = patch["patch_clean"]
    x = jt.array(noisy[None, :, :])
    with jt.no_grad():
        if mode == "heun":
            pc_pred, _ = model.denoise_langevin_dynamics(x)
        elif mode == "fixed":
            pc_pred = model.predict_clean(x, sigma=float(sigma))
        else:
            raise ValueError(f"unsupported evaluate mode: {mode}")
    pred = pc_pred.detach().numpy()[0].astype(np.float32, copy=False)
    return score_prediction(noisy, clean, pred), pred


def score_prediction(noisy, clean, pred):
    cd_noisy = chamfer(noisy, clean)
    cd_pred = chamfer(pred, clean)
    return {
        "cd_noisy": cd_noisy,
        "cd_pred": cd_pred,
        "cd_score": float(metric_to_score(cd_pred, cd_noisy)),
        "paired_noisy": float(((noisy - clean) ** 2).sum(axis=1).mean()),
        "paired_pred": float(((pred - clean) ** 2).sum(axis=1).mean()),
    }


def displacement_metrics(noisy, clean, pred, eps=1e-8):
    target = clean - noisy
    pred_disp = pred - noisy
    target_len = np.linalg.norm(target, axis=1)
    pred_len = np.linalg.norm(pred_disp, axis=1)
    valid = target_len > eps
    if not np.any(valid):
        valid = np.ones_like(target_len, dtype=bool)
    dot = (target * pred_disp).sum(axis=1)
    cosine = dot / np.maximum(target_len * pred_len, eps)
    ratio = pred_len / np.maximum(target_len, eps)
    under = np.maximum(target_len - pred_len, 0.0)
    over = np.maximum(pred_len - target_len, 0.0)
    return {
        "target_len_mean": float(target_len[valid].mean()),
        "pred_len_mean": float(pred_len[valid].mean()),
        "length_ratio_mean": float(ratio[valid].mean()),
        "length_ratio_median": float(np.median(ratio[valid])),
        "under_length_mean": float(under[valid].mean()),
        "over_length_mean": float(over[valid].mean()),
        "under_length_rate": float(np.mean(pred_len[valid] < target_len[valid])),
        "cosine_mean": float(cosine[valid].mean()),
        "cosine_median": float(np.median(cosine[valid])),
        "negative_cosine_rate": float(np.mean(cosine[valid] < 0.0)),
        "pred_disp_mean": float(pred_len[valid].mean()),
    }


def quantile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "p10": float(np.nanpercentile(values, 10)),
        "p25": float(np.nanpercentile(values, 25)),
        "median": float(np.nanmedian(values)),
        "p75": float(np.nanpercentile(values, 75)),
        "p90": float(np.nanpercentile(values, 90)),
        "max": float(np.nanmax(values)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_hard_patch_npz(path, selected_rows, patches):
    indices = [row["candidate_index"] for row in selected_rows]
    noisy = np.stack([patches[i]["patch_noisy"] for i in indices], axis=0)
    clean = np.stack([patches[i]["patch_clean"] for i in indices], axis=0)
    score_sigma = np.full((len(indices), 1), selected_rows[0]["noise_std"], dtype=np.float32)
    np.savez_compressed(
        path,
        pc_noisy=noisy.astype(np.float32, copy=False),
        pc_clean=clean.astype(np.float32, copy=False),
        score_sigma=score_sigma,
        rel_path=np.asarray([patches[i]["rel_path"] for i in indices]),
        seed_idx=np.asarray([patches[i]["seed_idx"] for i in indices], dtype=np.int64),
        cd_score=np.asarray([row["cd_score"] for row in selected_rows], dtype=np.float32),
        patch_scale=np.asarray([row["patch_scale"] for row in selected_rows], dtype=np.float32),
        geometry_category=np.asarray([row["geometry_category"] for row in selected_rows]),
        candidate_index=np.asarray(indices, dtype=np.int64),
    )


def load_hard_patch_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "pc_noisy": data["pc_noisy"].astype(np.float32, copy=False),
        "pc_clean": data["pc_clean"].astype(np.float32, copy=False),
        "score_sigma": data["score_sigma"].astype(np.float32, copy=False),
        "rel_path": data["rel_path"],
        "seed_idx": data["seed_idx"],
        "cd_score": data["cd_score"],
        "patch_scale": data["patch_scale"],
        "geometry_category": data["geometry_category"],
    }
