import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


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


def pca_basis(points):
    center = points.mean(axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return center, vt


def subsample(points, max_points=900, seed=123):
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def render_points(points, out_path, title, color):
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=color, alpha=0.9)
    ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def render_clean_surface(points, out_path, title):
    plot_points = subsample(points, max_points=850)
    center, basis = pca_basis(plot_points)
    local = (plot_points - center) @ basis.T
    triangulation = mtri.Triangulation(local[:, 0], local[:, 1])

    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, plot_points)
    ax.plot_trisurf(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        triangles=triangulation.triangles,
        color="#9ecae1",
        alpha=0.86,
        linewidth=0.08,
        edgecolor="#5f7f95",
        antialiased=True,
        shade=True,
    )
    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        s=2,
        c="#1f1f1f",
        alpha=0.22,
    )
    ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def make_sheet(paths, out_path, title):
    import matplotlib.image as mpimg

    cols = 5
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.6))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, paths):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def read_metrics(metrics_path):
    rows = {}
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["rank"])] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diag-dir",
        default="outputs1.1/patch_diagnostics/best_old_decoder",
    )
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    diag_dir = Path(args.diag_dir)
    out_dir = diag_dir / "clean_pred_views"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(diag_dir / "metrics.csv")

    clean_point_paths = []
    clean_surface_paths = []
    pred_paths = []
    for rank in range(1, args.count + 1):
        data = np.load(diag_dir / f"worst_{rank:02d}.npz")
        clean = data["clean"]
        pred = data["denoised"]
        row = metrics.get(rank, {})
        score = row.get("cd_score", "")
        score_text = f" score={float(score):.2f}" if score != "" else ""

        clean_point_path = out_dir / f"worst_{rank:02d}_clean_points.png"
        clean_surface_path = out_dir / f"worst_{rank:02d}_clean_surface.png"
        pred_path = out_dir / f"worst_{rank:02d}_pred_points.png"

        render_points(
            clean,
            clean_point_path,
            f"Rank {rank} clean points{score_text}",
            "#222222",
        )
        render_clean_surface(
            clean,
            clean_surface_path,
            f"Rank {rank} clean surface{score_text}",
        )
        render_points(
            pred,
            pred_path,
            f"Rank {rank} pred points{score_text}",
            "#4e79a7",
        )
        clean_point_paths.append(clean_point_path)
        clean_surface_paths.append(clean_surface_path)
        pred_paths.append(pred_path)

    make_sheet(
        clean_point_paths,
        out_dir / "sheet_clean_points.png",
        "Worst 10 patches: clean points only",
    )
    make_sheet(
        clean_surface_paths,
        out_dir / "sheet_clean_surface.png",
        "Worst 10 patches: clean surface from clean points",
    )
    make_sheet(
        pred_paths,
        out_dir / "sheet_pred_points.png",
        "Worst 10 patches: pred points only",
    )
    print(out_dir.resolve())


if __name__ == "__main__":
    main()
