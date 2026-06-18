import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

import jittor as jt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import chamfer_distance, metric_to_score, point_to_surface_distance  # noqa: E402
from src.model.parse import get_model  # noqa: E402
from tools.evaluate_shape_context_mismatch import (  # noqa: E402
    build_patch_batch,
    make_noisy_shape,
    predict_condition,
)
from tools.hard_patch_common import (  # noqa: E402
    displacement_metrics,
    geometry_category,
    local_geometry,
    pca_geometry,
    read_datalist,
)
from tools.train_full_cloud_fusion_probe import load_shape, usable_paths  # noqa: E402


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_shape_context_model(args):
    model_config = OmegaConf.to_container(
        OmegaConf.load(args.model_config),
        resolve=True,
    )
    model_config["shape_pretrained_ckpt"] = args.shape_pretrained_checkpoint
    transform_config = OmegaConf.to_container(
        OmegaConf.load(args.transform_config),
        resolve=True,
    )
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load(args.checkpoint)
    model.eval()
    return model


def score_patch(noisy, clean, pred, mesh_vertices, mesh_faces):
    cd_noisy = chamfer_distance(noisy, clean, normalize=False)
    cd_pred = chamfer_distance(pred, clean, normalize=False)
    p2s_noisy = point_to_surface_distance(
        noisy,
        mesh_vertices,
        mesh_faces,
        normalize_ref_pc=None,
    )
    p2s_pred = point_to_surface_distance(
        pred,
        mesh_vertices,
        mesh_faces,
        normalize_ref_pc=None,
    )
    cd_score = metric_to_score(cd_pred, cd_noisy)
    p2s_score = metric_to_score(p2s_pred, p2s_noisy)
    return {
        "cd_score": float(cd_score),
        "p2s_score": float(p2s_score),
        "final_score": float(0.5 * (cd_score + p2s_score)),
        "cd_noisy": float(cd_noisy),
        "cd_pred": float(cd_pred),
        "p2s_noisy": float(p2s_noisy),
        "p2s_pred": float(p2s_pred),
    }


def pca_frame(points):
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def two_cluster_1d(values, steps=16):
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray([np.percentile(values, 25), np.percentile(values, 75)])
    for _ in range(steps):
        dist = np.abs(values[:, None] - centers[None, :])
        labels = dist.argmin(axis=1)
        for k in range(2):
            if np.any(labels == k):
                centers[k] = values[labels == k].mean()
    centers = np.sort(centers)
    labels = (np.abs(values[:, None] - centers[None, :])).argmin(axis=1)
    counts = [int(np.sum(labels == k)) for k in range(2)]
    return centers, counts


def plane_thickness_diagnostics(clean, noisy, pred):
    _, basis = pca_frame(clean)
    normal = basis[:, -1]
    origin = clean.mean(axis=0)

    def project(points):
        return (points - origin[None, :]) @ normal

    clean_h = project(clean)
    noisy_h = project(noisy)
    pred_h = project(pred)
    clean_centers, clean_counts = two_cluster_1d(clean_h)
    pred_centers, pred_counts = two_cluster_1d(pred_h)
    clean_gap = float(clean_centers[1] - clean_centers[0])
    pred_gap = float(pred_centers[1] - pred_centers[0])
    clean_mid = float(clean_centers.mean())
    pred_mid_abs = float(np.mean(np.abs(pred_h - clean_mid)))
    clean_mid_abs = float(np.mean(np.abs(clean_h - clean_mid)))
    noisy_mid_abs = float(np.mean(np.abs(noisy_h - clean_mid)))
    return {
        "clean_normal_thickness_p95": float(np.percentile(clean_h, 97.5) - np.percentile(clean_h, 2.5)),
        "noisy_normal_thickness_p95": float(np.percentile(noisy_h, 97.5) - np.percentile(noisy_h, 2.5)),
        "pred_normal_thickness_p95": float(np.percentile(pred_h, 97.5) - np.percentile(pred_h, 2.5)),
        "clean_two_plane_gap": clean_gap,
        "pred_two_plane_gap": pred_gap,
        "pred_gap_ratio": float(pred_gap / max(clean_gap, 1e-12)),
        "clean_mid_abs_mean": clean_mid_abs,
        "noisy_mid_abs_mean": noisy_mid_abs,
        "pred_mid_abs_mean": pred_mid_abs,
        "pred_mid_abs_ratio": float(pred_mid_abs / max(clean_mid_abs, 1e-12)),
        "clean_plane_count_low": clean_counts[0],
        "clean_plane_count_high": clean_counts[1],
        "pred_plane_count_low": pred_counts[0],
        "pred_plane_count_high": pred_counts[1],
    }


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
    ax.set_xlim(center[0] - radius * 1.08, center[0] + radius * 1.08)
    ax.set_ylim(center[1] - radius * 1.08, center[1] + radius * 1.08)
    ax.set_aspect("equal", adjustable="box")


def equal_axes_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius * 1.08, center[0] + radius * 1.08)
    ax.set_ylim(center[1] - radius * 1.08, center[1] + radius * 1.08)
    ax.set_zlim(center[2] - radius * 1.08, center[2] + radius * 1.08)


def draw_overlay(path, base_points, pred, title, base_label, base_color):
    all_points = np.concatenate([base_points, pred], axis=0)
    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax3d.scatter(
        base_points[:, 0],
        base_points[:, 1],
        base_points[:, 2],
        s=3,
        c=base_color,
        alpha=0.28,
        depthshade=False,
        label=base_label,
    )
    ax3d.scatter(
        pred[:, 0],
        pred[:, 1],
        pred[:, 2],
        s=4,
        c="#1f77b4",
        alpha=0.72,
        depthshade=False,
        label="pred",
    )
    equal_axes_3d(ax3d, all_points)
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_title("3D")
    ax3d.legend(loc="upper right", fontsize=8, frameon=False)

    for slot, view in zip([grid[0, 1], grid[1, 0], grid[1, 1]], ["xy", "xz", "yz"]):
        ax = fig.add_subplot(slot)
        base2 = project(base_points, view)
        pred2 = project(pred, view)
        all2 = np.concatenate([base2, pred2], axis=0)
        ax.scatter(base2[:, 0], base2[:, 1], s=4, c=base_color, alpha=0.25, linewidths=0, label=base_label)
        ax.scatter(pred2[:, 0], pred2[:, 1], s=4, c="#1f77b4", alpha=0.72, linewidths=0, label="pred")
        equal_axes_2d(ax, all2)
        ax.grid(True, color="#d5d8de", linewidth=0.5, alpha=0.55)
        ax.set_title(view.upper())
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def choose_paths(args, rng):
    paths = usable_paths(
        read_datalist(args.datalist),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    laptop = [p for p in paths if args.laptop_category in Path(p).parts]
    non_laptop = [p for p in paths if p not in set(laptop)]
    rng.shuffle(laptop)
    rng.shuffle(non_laptop)
    chosen = laptop[: args.max_laptop_shapes]
    chosen += non_laptop[: max(0, args.max_shapes - len(chosen))]
    return chosen[: args.max_shapes], len(laptop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs_result/outputs0618_3/shape_context_vm/checkpoints/checkpoint_best.pkl")
    parser.add_argument("--shape-pretrained-checkpoint", default="outputs_result/outputs0618_3/shape_pretrain/checkpoints/processor_best.pkl")
    parser.add_argument("--model-config", default="configs/model/shape_context_vm.yaml")
    parser.add_argument("--transform-config", default="configs/transform/shape_context_vm_laplace.yaml")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--out-dir", default="outputs/outputs0618_3_low_patch_planes")
    parser.add_argument("--max-shapes", type=int, default=10)
    parser.add_argument("--max-laptop-shapes", type=int, default=4)
    parser.add_argument("--laptop-category", default="03642806")
    parser.add_argument("--patches-per-shape", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    if args.use_cuda:
        jt.flags.use_cuda = 1

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    model = load_shape_context_model(args)
    paths, laptop_count = choose_paths(args, rng)

    rows = []
    records = []
    with jt.no_grad():
        for shape_index, rel_path in enumerate(paths):
            shape = load_shape(
                rel_path,
                args.clean_root,
                args.mesh_root,
                args.num_points,
                rng,
                args.sample_missing_clean,
            )
            sigma = float(rng.uniform(args.sigma_min, args.sigma_max))
            instance = make_noisy_shape(
                shape,
                sigma,
                rng,
                model.region_count,
                model.points_per_region,
            )
            patch_batch = build_patch_batch(
                instance,
                args.patches_per_shape,
                args.patch_size,
            )
            encoded = model.encode_shape(
                jt.array(instance["region_points"][None, ...]),
                jt.array(instance["region_centers"][None, ...]),
            )
            prediction = predict_condition(
                model,
                patch_batch,
                instance,
                encoded,
                args.batch_size,
            )
            for patch_index in range(prediction.shape[0]):
                seed = patch_batch["seeds"][patch_index]
                noisy_abs = patch_batch["patches"][patch_index] + seed[None, :]
                clean_abs = patch_batch["clean_absolute"][patch_index]
                pred_abs = prediction[patch_index]
                geom = {}
                geom.update(pca_geometry(clean_abs))
                geom.update(local_geometry(clean_abs, seed=args.seed + patch_index))
                row = {
                    "shape_index": shape_index,
                    "patch_index": patch_index,
                    "rel_path": rel_path,
                    "is_laptop": int(args.laptop_category in Path(rel_path).parts),
                    "sigma": sigma,
                    "seed_x": float(seed[0]),
                    "seed_y": float(seed[1]),
                    "seed_z": float(seed[2]),
                    **geom,
                }
                row["geometry_category"] = geometry_category(row)
                row.update(score_patch(noisy_abs, clean_abs, pred_abs, instance["mesh_vertices"], instance["mesh_faces"]))
                row.update(displacement_metrics(noisy_abs, clean_abs, pred_abs))
                row.update(plane_thickness_diagnostics(clean_abs, noisy_abs, pred_abs))
                rows.append(row)
                records.append({
                    "row": row,
                    "clean": clean_abs,
                    "noisy": noisy_abs,
                    "pred": pred_abs,
                })
            print(f"[{shape_index + 1}/{len(paths)}] {rel_path}", flush=True)
            jt.gc()

    ranked = sorted(records, key=lambda item: item["row"]["final_score"])
    for rank, record in enumerate(ranked[: args.top_k], start=1):
        row = record["row"]
        row["rank"] = rank
        stem = (
            f"rank{rank:02d}_shape{row['shape_index']:02d}_"
            f"patch{row['patch_index']:02d}_final{row['final_score']:.2f}"
        )
        title = (
            f"{stem} | CD={row['cd_score']:.2f} P2S={row['p2s_score']:.2f} "
            f"ratio={row['length_ratio_mean']:.3f} cos={row['cosine_mean']:.3f}\n"
            f"pred_gap/clean_gap={row['pred_gap_ratio']:.3f} "
            f"pred_mid/clean_mid={row['pred_mid_abs_ratio']:.3f} | "
            f"{row['geometry_category']} | {row['rel_path']}"
        )
        clean_path = image_dir / f"{stem}_pred_clean.png"
        noisy_path = image_dir / f"{stem}_pred_noisy.png"
        draw_overlay(clean_path, record["clean"], record["pred"], title, "clean", "#222222")
        draw_overlay(noisy_path, record["noisy"], record["pred"], title, "noisy", "#d95f02")
        row["pred_clean_image"] = str(clean_path)
        row["pred_noisy_image"] = str(noisy_path)

    rows = sorted(rows, key=lambda row: (row["final_score"], row["shape_index"], row["patch_index"]))
    write_csv(out_dir / "rows.csv", rows)
    summary = {
        "args": vars(args),
        "scanned_patch_count": len(rows),
        "available_laptop_shape_count": laptop_count,
        "scanned_laptop_patch_count": int(sum(row["is_laptop"] for row in rows)),
        "lowest": [
            {
                key: row.get(key)
                for key in [
                    "rank",
                    "shape_index",
                    "patch_index",
                    "rel_path",
                    "is_laptop",
                    "final_score",
                    "cd_score",
                    "p2s_score",
                    "length_ratio_mean",
                    "cosine_mean",
                    "negative_cosine_rate",
                    "clean_two_plane_gap",
                    "pred_two_plane_gap",
                    "pred_gap_ratio",
                    "pred_mid_abs_ratio",
                    "geometry_category",
                    "pred_clean_image",
                    "pred_noisy_image",
                ]
            }
            for row in sorted(rows, key=lambda row: row["final_score"])[: args.top_k]
        ],
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
