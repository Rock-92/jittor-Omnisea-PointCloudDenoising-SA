import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

import jittor as jt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.parse import get_model  # noqa: E402
from tools.diagnose_outputs0618_3_low_patch_planes import (  # noqa: E402
    plane_thickness_diagnostics,
)
from tools.evaluate_shape_context_mismatch import (  # noqa: E402
    build_patch_batch,
    make_noisy_shape,
    predict_condition,
)
from tools.hard_patch_common import read_datalist  # noqa: E402
from tools.train_full_cloud_fusion_probe import load_shape, usable_paths  # noqa: E402


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
    return chosen[: args.max_shapes]


def collect_ranked_patches(args, model):
    rng = np.random.default_rng(args.seed)
    paths = choose_paths(args, rng)
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
                plane = plane_thickness_diagnostics(clean_abs, noisy_abs, pred_abs)
                records.append(
                    {
                        "shape_index": shape_index,
                        "patch_index": patch_index,
                        "rel_path": rel_path,
                        "sigma": sigma,
                        "noisy": noisy_abs.astype(np.float32, copy=False),
                        "clean": clean_abs.astype(np.float32, copy=False),
                        "pred": pred_abs.astype(np.float32, copy=False),
                        "pred_gap_ratio": plane["pred_gap_ratio"],
                        "plane": plane,
                    }
                )
    return sorted(records, key=lambda item: item["pred_gap_ratio"])


def nearest_clean_stats(points, clean):
    dist = ((points[:, None, :] - clean[None, :, :]) ** 2.0).sum(axis=-1)
    nearest = dist.argmin(axis=1)
    d = np.sqrt(dist[np.arange(points.shape[0]), nearest])
    return nearest, {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p90": float(np.quantile(d, 0.90)),
        "p99": float(np.quantile(d, 0.99)),
    }


def numpy_clean_normals_like_model(clean, k):
    tree = cKDTree(clean)
    _, idx = tree.query(clean, k=min(max(int(k), 3), clean.shape[0]))
    p0 = clean
    p1 = clean[idx[:, 1]]
    p2 = clean[idx[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return normal, idx


def model_surface_terms(model, noisy, clean, pred):
    noisy_jt = jt.array(noisy[None, ...])
    clean_jt = jt.array(clean[None, ...])
    pred_jt = jt.array(pred[None, ...])
    with jt.no_grad():
        losses = model.get_surface_aligned_losses(
            pc_pred=pred_jt,
            pc_noisy=noisy_jt,
            pc_clean=clean_jt,
            sigma=None,
        )
        loss_values = {k: float(v.item()) for k, v in losses.items()}
    return loss_values


def coherence_mask_stats(model, noisy, pred):
    k = min(max(model.surface_coherence_k, 1) + 1, pred.shape[0])
    tree = cKDTree(noisy)
    _, idx = tree.query(noisy, k=k)
    idx = idx[:, 1:]
    disp = pred - noisy
    disp_len = np.linalg.norm(disp, axis=1, keepdims=True)
    direction = disp / np.maximum(disp_len, 1e-12)
    neigh_dir = direction[idx]
    cos = (direction[:, None, :] * neigh_dir).sum(axis=-1)
    same_forward = cos > model.surface_coherence_cos
    pair_vec = noisy[idx] - noisy[:, None, :]
    pair_dir = pair_vec / np.maximum(np.linalg.norm(pair_vec, axis=-1, keepdims=True), 1e-12)
    inward_i = (direction[:, None, :] * pair_dir).sum(axis=-1)
    inward_j = (neigh_dir * (-pair_dir)).sum(axis=-1)
    same_reverse = (
        (cos < -model.surface_coherence_cos)
        & (inward_i > model.surface_coherence_center_cos)
        & (inward_j > model.surface_coherence_center_cos)
    )
    same = same_forward | same_reverse
    grouped = same.sum(axis=1) > 0
    return {
        "neighbor_pairs": int(same.size),
        "same_forward_rate": float(same_forward.mean()),
        "same_reverse_rate": float(same_reverse.mean()),
        "same_surface_pair_rate": float(same.mean()),
        "grouped_point_rate": float(grouped.mean()),
        "ungrouped_point_rate": float(1.0 - grouped.mean()),
        "cos_mean": float(cos.mean()),
        "cos_p10": float(np.quantile(cos, 0.10)),
        "cos_p90": float(np.quantile(cos, 0.90)),
    }


def plane_distance_with_model_normals(model, clean, pred):
    normals, nn_idx = numpy_clean_normals_like_model(clean, model.surface_normal_k)
    nearest, nearest_stats = nearest_clean_stats(pred, clean)
    signed = ((pred - clean[nearest]) * normals[nearest]).sum(axis=1)
    abs_signed = np.abs(signed)
    neighbor_cross_layer = []
    for i, row in enumerate(nn_idx):
        # Measure whether the nearest-neighbor normal was built from points
        # separated along the patch's thinnest PCA axis.
        local = clean[row]
        eigvals, eigvecs = np.linalg.eigh(np.cov(clean.T))
        normal_axis = eigvecs[:, np.argmin(eigvals)]
        heights = local @ normal_axis
        neighbor_cross_layer.append(float(heights.max() - heights.min()))
    return {
        "nearest_euclidean": nearest_stats,
        "model_plane_abs_dist_mean": float(abs_signed.mean()),
        "model_plane_abs_dist_median": float(np.median(abs_signed)),
        "model_plane_abs_dist_p90": float(np.quantile(abs_signed, 0.90)),
        "normal_neighbor_height_span_mean": float(np.mean(neighbor_cross_layer)),
        "normal_neighbor_height_span_p90": float(np.quantile(neighbor_cross_layer, 0.90)),
        "normal_norm_zero_rate": float(np.mean(np.linalg.norm(normals, axis=1) < 1e-6)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs_result/outputs0618_3/shape_context_vm/checkpoints/checkpoint_best.pkl")
    parser.add_argument("--shape-pretrained-checkpoint", default="outputs_result/outputs0618_3/shape_pretrain/checkpoints/processor_best.pkl")
    parser.add_argument("--model-config", default="configs/model/shape_context_vm.yaml")
    parser.add_argument("--transform-config", default="configs/transform/shape_context_vm_laplace.yaml")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--out", default="outputs/outputs0618_3_low_patch_planes/loss_probe_rank1.json")
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--max-laptop-shapes", type=int, default=4)
    parser.add_argument("--laptop-category", default="03642806")
    parser.add_argument("--patches-per-shape", type=int, default=6)
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

    model = load_shape_context_model(args)
    records = collect_ranked_patches(args, model)
    record = records[0]
    noisy = record["noisy"]
    clean = record["clean"]
    pred = record["pred"]
    result = {
        "selected": {
            "shape_index": record["shape_index"],
            "patch_index": record["patch_index"],
            "rel_path": record["rel_path"],
            "sigma": record["sigma"],
            "pred_gap_ratio": record["pred_gap_ratio"],
        },
        "surface_loss_values": model_surface_terms(model, noisy, clean, pred),
        "plane_thickness": record["plane"],
        "coherence_mask": coherence_mask_stats(model, noisy, pred),
        "model_plane_distance": plane_distance_with_model_normals(model, clean, pred),
        "config": {
            "surface_normal_k": model.surface_normal_k,
            "surface_coherence_k": model.surface_coherence_k,
            "surface_coherence_cos": model.surface_coherence_cos,
            "surface_coherence_center_cos": model.surface_coherence_center_cos,
            "surface_outlier_margin": model.surface_outlier_margin,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
