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


def load_model(checkpoint):
    return load_legacy_model(checkpoint)


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
    radius = float(np.sqrt((patch_noisy**2).sum(axis=1)).max())

    return {
        "rel_path": rel_path,
        "noise_std": noise_std,
        "seed_idx": seed_idx,
        "patch_noisy": patch_noisy,
        "patch_clean": patch_clean,
        "mesh_vertices": mesh_local,
        "mesh_faces": mesh_faces,
        "patch_radius": radius,
    }


def evaluate_patch(model, patch):
    pc_noisy = jt.array(patch["patch_noisy"][None, :, :])
    with jt.no_grad():
        pc_pred, _ = model.denoise_langevin_dynamics(pc_noisy)
    pred = pc_pred.detach().numpy()[0].astype(np.float32, copy=False)
    clean = patch["patch_clean"]
    noisy = patch["patch_noisy"]
    cd_noisy = chamfer_distance(noisy, clean)
    cd_pred = chamfer_distance(pred, clean)
    return {
        **patch,
        "patch_pred": pred,
        "cd_noisy": cd_noisy,
        "cd_pred": cd_pred,
        "cd_score": metric_to_score(cd_pred, cd_noisy),
        "cd_delta": cd_pred - cd_noisy,
    }


def crop_mesh(vertices, faces, radius, max_faces=5000):
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
        relaxed = (
            (center_radius <= context_radius)
            & (face_radius_max <= context_radius * 1.3)
        )
        selected = np.flatnonzero(relaxed)
    if selected.size == 0:
        selected = np.argsort(center_radius)[: min(max_faces, len(faces))]
    if selected.size > max_faces:
        selected = selected[np.argsort(center_radius[selected])[:max_faces]]
    selected_faces = faces[selected]
    unique_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    cropped_vertices = vertices[unique_vertices]
    cropped_faces = inverse.reshape(selected_faces.shape)
    return cropped_vertices, cropped_faces


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
    ax.grid(False)


def render_triptych(item, rank, out_path):
    clean = item["patch_clean"]
    pred = item["patch_pred"]
    mesh_v, mesh_f = crop_mesh(
        item["mesh_vertices"],
        item["mesh_faces"],
        item["patch_radius"],
    )
    axis_pts = np.concatenate([clean, pred], axis=0)

    fig = plt.figure(figsize=(14.5, 4.6))
    ax_clean = fig.add_subplot(1, 3, 1, projection="3d")
    ax_mesh = fig.add_subplot(1, 3, 2, projection="3d")
    ax_pred = fig.add_subplot(1, 3, 3, projection="3d")

    for ax in [ax_clean, ax_mesh, ax_pred]:
        set_axes_equal(ax, axis_pts)
        ax.view_init(elev=18, azim=35)

    ax_clean.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=5, c="#222222", alpha=0.9)
    ax_clean.set_title("Clean points", fontsize=11)

    triangles = mesh_v[mesh_f]
    mesh_collection = Poly3DCollection(
        triangles,
        facecolor="#9ecae1",
        edgecolor="#6f93a8",
        linewidth=0.05,
        alpha=0.78,
    )
    ax_mesh.add_collection3d(mesh_collection)
    ax_mesh.set_title("True mesh surface", fontsize=11)

    ax_pred.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=5, c="#4e79a7", alpha=0.9)
    ax_pred.set_title("Pred points", fontsize=11)

    title = (
        f"Rank {rank} | score={item['cd_score']:.2f} | "
        f"CD noisy={item['cd_noisy']:.6g}, pred={item['cd_pred']:.6g} | "
        f"mesh faces={len(mesh_f)}\n{item['rel_path']}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs1.1/checkpoints/vm/checkpoint_best.pkl")
    parser.add_argument("--mesh-root", default="E:/Code/competition2_EdgeConv/dataset_clean")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default="outputs1.1/patch_diagnostics/best_old_decoder/true_mesh_triptychs")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    mesh_root = Path(args.mesh_root)
    rel_paths = [
        line.strip()
        for line in (PROJECT_ROOT / args.datalist).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    chosen = [rel_paths[int(rng.integers(0, len(rel_paths)))] for _ in range(args.candidates)]
    model = load_model(PROJECT_ROOT / args.checkpoint)

    records = []
    for i, rel_path in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {rel_path}", flush=True)
        patch = sample_patch(rel_path, mesh_root, rng, args.patch_size)
        records.append(evaluate_patch(model, patch))
    records.sort(key=lambda r: (r["cd_score"], -r["cd_delta"]))
    worst = records[:10]

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "rel_path", "cd_score", "cd_noisy", "cd_pred", "cd_delta", "patch_radius"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(worst, 1):
            writer.writerow({
                "rank": rank,
                "rel_path": item["rel_path"],
                "cd_score": item["cd_score"],
                "cd_noisy": item["cd_noisy"],
                "cd_pred": item["cd_pred"],
                "cd_delta": item["cd_delta"],
                "patch_radius": item["patch_radius"],
            })
            render_triptych(item, rank, out_dir / f"worst_{rank:02d}_triptych.png")
    summary = {
        "checkpoint": str((PROJECT_ROOT / args.checkpoint).resolve()),
        "mesh_root": str(mesh_root.resolve()),
        "candidates": args.candidates,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "images": [
            str((out_dir / f"worst_{rank:02d}_triptych.png").resolve())
            for rank in range(1, len(worst) + 1)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
