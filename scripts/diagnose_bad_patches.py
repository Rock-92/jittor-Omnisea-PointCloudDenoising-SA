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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups
from scripts.legacy_vm import load_legacy_model


def normalize_pc(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / scale).astype(np.float32, copy=False)


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
    _, nn_idx = cKDTree(noisy).query(seed_point[None, :], k=patch_size)
    nn_idx = nn_idx[0]
    patch_noisy_abs = noisy[nn_idx].astype(np.float32, copy=False)
    patch_clean_abs = clean[nn_idx].astype(np.float32, copy=False)
    patch_noisy = patch_noisy_abs - seed_point[None, :]
    patch_clean = patch_clean_abs - seed_point[None, :]
    return {
        "rel_path": rel_path,
        "noise_std": noise_std,
        "seed_idx": seed_idx,
        "patch_noisy": patch_noisy.astype(np.float32, copy=False),
        "patch_clean": patch_clean.astype(np.float32, copy=False),
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
    paired_noisy = float(((noisy - clean) ** 2).sum(axis=1).mean())
    paired_pred = float(((pred - clean) ** 2).sum(axis=1).mean())
    return {
        **patch,
        "patch_pred": pred,
        "cd_noisy": cd_noisy,
        "cd_pred": cd_pred,
        "cd_score": metric_to_score(cd_pred, cd_noisy),
        "cd_delta": cd_pred - cd_noisy,
        "paired_noisy": paired_noisy,
        "paired_pred": paired_pred,
        "paired_delta": paired_pred - paired_noisy,
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


def scatter_panel(ax, clean, other, other_label, title, other_color):
    pts = np.concatenate([clean, other], axis=0)
    set_axes_equal(ax, pts)
    ax.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=3, c="#222222", alpha=0.22, label="clean")
    ax.scatter(other[:, 0], other[:, 1], other[:, 2], s=4, c=other_color, alpha=0.72, label=other_label)
    ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, frameon=False)


def plot_patch(item, rank, out_path):
    clean = item["patch_clean"]
    noisy = item["patch_noisy"]
    pred = item["patch_pred"]
    fig = plt.figure(figsize=(13, 4.2))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    scatter_panel(ax1, clean, noisy, "noisy", "Noisy vs clean", "#e15759")
    scatter_panel(ax2, clean, pred, "denoised", "Denoised vs clean", "#4e79a7")
    set_axes_equal(ax3, np.concatenate([clean, noisy, pred], axis=0))
    ax3.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=3, c="#222222", alpha=0.16, label="clean")
    ax3.scatter(noisy[:, 0], noisy[:, 1], noisy[:, 2], s=3, c="#e15759", alpha=0.45, label="noisy")
    ax3.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=3, c="#4e79a7", alpha=0.55, label="denoised")
    ax3.view_init(elev=18, azim=35)
    ax3.set_title("Overlay", fontsize=10)
    ax3.legend(loc="upper right", fontsize=7, frameon=False)
    title = (
        f"Rank {rank} | score={item['cd_score']:.2f} | "
        f"CD noisy={item['cd_noisy']:.6g}, pred={item['cd_pred']:.6g} | "
        f"noise_std={item['noise_std']:.4f}\n{item['rel_path']}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_contact_sheet(image_paths, out_path):
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(len(image_paths), 1, figsize=(13, 4.2 * len(image_paths)))
    if len(image_paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, image_paths):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs1.1/checkpoints/vm/checkpoint_best.pkl")
    parser.add_argument("--mesh-root", default="E:/Code/competition2_EdgeConv/dataset_clean")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default="outputs1.1/patch_diagnostics/best_old_decoder")
    parser.add_argument("--candidates", type=int, default=80)
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
    model = load_model(PROJECT_ROOT / args.checkpoint)

    records = []
    for i, rel_path in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {rel_path}", flush=True)
        patch = sample_patch(rel_path, mesh_root, rng, args.patch_size)
        records.append(evaluate_patch(model, patch))

    records.sort(key=lambda r: (r["cd_score"], -r["cd_delta"]))
    worst = records[:10]

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "rel_path",
            "noise_std",
            "seed_idx",
            "cd_score",
            "cd_noisy",
            "cd_pred",
            "cd_delta",
            "paired_noisy",
            "paired_pred",
            "paired_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, item in enumerate(records, 1):
            writer.writerow({k: item.get(k) for k in fieldnames if k != "rank"} | {"rank": rank})

    image_paths = []
    for rank, item in enumerate(worst, 1):
        image_path = out_dir / f"worst_{rank:02d}.png"
        plot_patch(item, rank, image_path)
        image_paths.append(image_path)
        np.savez_compressed(
            out_dir / f"worst_{rank:02d}.npz",
            noisy=item["patch_noisy"],
            denoised=item["patch_pred"],
            clean=item["patch_clean"],
        )

    contact_path = out_dir / "worst_10_contact_sheet.png"
    make_contact_sheet(image_paths, contact_path)
    summary = {
        "checkpoint": str((PROJECT_ROOT / args.checkpoint).resolve()),
        "mesh_root": str(mesh_root.resolve()),
        "candidates": args.candidates,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "mean_cd_score": float(np.mean([r["cd_score"] for r in records])),
        "median_cd_score": float(np.median([r["cd_score"] for r in records])),
        "num_worse_than_noisy": int(sum(r["cd_delta"] > 0 for r in records)),
        "worst": [
            {
                "rank": i + 1,
                "rel_path": r["rel_path"],
                "cd_score": r["cd_score"],
                "cd_noisy": r["cd_noisy"],
                "cd_pred": r["cd_pred"],
                "cd_delta": r["cd_delta"],
                "image": str(image_paths[i]),
            }
            for i, r in enumerate(worst)
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"contact_sheet={contact_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
