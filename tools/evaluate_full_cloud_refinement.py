import argparse
import csv
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (  # noqa: E402
    chamfer_distance,
    metric_to_score,
    point_to_surface_distance,
)
from src.model.refinement import MultiStageGeometryRefiner  # noqa: E402
from src.model.vm import (  # noqa: E402
    farthest_point_sampling,
    knn_points,
)
from tools.hard_patch_common import (  # noqa: E402
    load_model,
    quantile_summary,
    read_datalist,
)


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_clean(clean):
    center = (clean.max(axis=0) + clean.min(axis=0)) / 2.0
    centered = clean - center
    scale = np.sqrt((centered ** 2.0).sum(axis=1)).max()
    return (
        (centered / max(float(scale), 1e-12)).astype(
            np.float32,
            copy=False,
        ),
        center.astype(np.float32, copy=False),
        float(scale),
    )


def build_patches(noisy, patch_size, seed_k):
    noisy_var = jt.array(noisy[None, :, :])
    point_count = noisy.shape[0]
    patch_size = min(int(patch_size), point_count)
    patch_count = min(
        point_count,
        max(1, int(float(seed_k) * point_count / patch_size)),
    )
    seeds, _ = farthest_point_sampling(noisy_var, patch_count)
    patch_distances, point_indices, patches = knn_points(
        seeds,
        noisy_var,
        patch_size,
    )

    covered = np.zeros((point_count,), dtype=np.bool_)
    covered[point_indices[0].numpy().reshape(-1)] = True
    missing_indices = np.flatnonzero(~covered).astype(np.int32)
    if missing_indices.size:
        extra_indices = jt.array(missing_indices).int32()
        extra_seeds = noisy_var[:, extra_indices, :]
        extra_distances, extra_point_indices, extra_patches = knn_points(
            extra_seeds,
            noisy_var,
            patch_size,
        )
        seeds = jt.concat([seeds, extra_seeds], dim=1)
        patch_distances = jt.concat(
            [patch_distances, extra_distances],
            dim=1,
        )
        point_indices = jt.concat(
            [point_indices, extra_point_indices],
            dim=1,
        )
        patches = jt.concat([patches, extra_patches], dim=1)

    patches = patches[0]
    seeds = seeds[0]
    point_indices = point_indices[0]
    patch_distances = patch_distances[0]
    seed_expanded = seeds.unsqueeze(1).broadcast(patches.shape)
    centered_patches = patches - seed_expanded
    normalized_distances = patch_distances / (
        patch_distances[:, -1:].broadcast(patch_distances.shape) + 1e-8
    )
    return (
        centered_patches,
        seeds,
        point_indices.numpy().astype(np.int64, copy=False),
        normalized_distances.numpy().astype(np.float32, copy=False),
    )


def predict_patch_batches(
    vm,
    refiner,
    patches,
    batch_size,
    coarse_mode,
    sigma,
):
    coarse_batches = []
    refined_batches = []
    vm.eval()
    refiner.eval()
    with jt.no_grad():
        for start in range(0, patches.shape[0], batch_size):
            end = min(start + batch_size, patches.shape[0])
            noisy_patch = patches[start:end]
            if coarse_mode == "fixed":
                sigma_batch = jt.ones((end - start, 1)) * float(sigma)
                coarse_patch = vm.predict_clean(
                    noisy_patch,
                    sigma=sigma_batch,
                )
            elif coarse_mode == "heun":
                coarse_patch, _ = vm.denoise_langevin_dynamics(noisy_patch)
            else:
                raise ValueError(f"unsupported coarse mode: {coarse_mode}")
            refined_patch, _ = refiner(coarse_patch, noisy_patch)
            coarse_batches.append(
                coarse_patch.numpy().astype(np.float32, copy=False)
            )
            refined_batches.append(
                refined_patch.numpy().astype(np.float32, copy=False)
            )
    return (
        np.concatenate(coarse_batches, axis=0),
        np.concatenate(refined_batches, axis=0),
    )


def fuse_patches(
    noisy,
    patch_prediction,
    seeds,
    point_indices,
    normalized_distances,
    fusion_tau,
):
    point_count = noisy.shape[0]
    weighted_sum = np.zeros_like(noisy, dtype=np.float64)
    weight_sum = np.zeros((point_count,), dtype=np.float64)
    absolute_prediction = patch_prediction + seeds[:, None, :]
    weights = np.exp(
        -float(fusion_tau) * normalized_distances
    ).astype(np.float64)
    for patch_index in range(point_indices.shape[0]):
        indices = point_indices[patch_index]
        patch_weights = weights[patch_index]
        np.add.at(
            weighted_sum,
            indices,
            absolute_prediction[patch_index] * patch_weights[:, None],
        )
        np.add.at(weight_sum, indices, patch_weights)
    output = noisy.copy()
    covered = weight_sum > 0.0
    output[covered] = (
        weighted_sum[covered] / weight_sum[covered, None]
    ).astype(np.float32)
    return output


def score_prediction(noisy, clean, prediction, mesh_vertices, mesh_faces):
    cd_noisy = chamfer_distance(noisy, clean, normalize=True)
    p2s_noisy = point_to_surface_distance(
        noisy,
        mesh_vertices,
        mesh_faces,
        normalize_ref_pc=clean,
    )
    cd_score = metric_to_score(
        chamfer_distance(prediction, clean, normalize=True),
        cd_noisy,
    )
    p2s_score = metric_to_score(
        point_to_surface_distance(
            prediction,
            mesh_vertices,
            mesh_faces,
            normalize_ref_pc=clean,
        ),
        p2s_noisy,
    )
    return {
        "cd_score": cd_score,
        "p2s_score": p2s_score,
        "final_score": 0.5 * (cd_score + p2s_score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/vm_ssl/checkpoint_best.pkl",
    )
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument(
        "--out-dir",
        default="outputs/full_cloud_refinement_eval",
    )
    parser.add_argument("--max-shapes", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="fixed")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-max-residual", type=float, default=0.012)
    parser.add_argument("--stage2-max-residual", type=float, default=0.008)
    parser.add_argument("--adaptive-v2", action="store_true")
    parser.add_argument("--min-residual-ratio", type=float, default=0.2)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    coarse_dir = out_dir / "vm"
    refined_dir = out_dir / "vm_refined"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    refined_dir.mkdir(parents=True, exist_ok=True)

    rel_paths = read_datalist(args.datalist)
    usable = [
        rel_path
        for rel_path in rel_paths
        if (
            Path(args.clean_root) / rel_path / "clean.npy"
        ).exists()
        and (
            Path(args.mesh_root)
            / rel_path
            / "models/model_normalized.obj"
        ).exists()
    ]
    if args.max_shapes > 0:
        usable = usable[:args.max_shapes]
    if not usable:
        raise FileNotFoundError("no complete validation shapes found")

    vm = load_model(args.checkpoint)
    for parameter in vm.parameters():
        parameter.stop_grad()
    refiner = MultiStageGeometryRefiner(
        num_stages=args.stages,
        stage_max_residuals=(
            args.stage1_max_residual,
            args.stage2_max_residual,
        ),
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        adaptive_v2=args.adaptive_v2,
        min_residual_ratio=args.min_residual_ratio,
    )
    refiner.load(args.refiner_checkpoint)

    rows = []
    for shape_index, rel_path in enumerate(usable):
        print(
            f"[{shape_index + 1}/{len(usable)}] {rel_path}",
            flush=True,
        )
        clean_raw = np.load(
            Path(args.clean_root) / rel_path / "clean.npy"
        ).astype(np.float32, copy=False)
        clean, normalize_center, normalize_scale = normalize_clean(clean_raw)
        noisy = (
            clean
            + rng.standard_normal(clean.shape).astype(np.float32)
            * float(args.noise_std)
        ).astype(np.float32, copy=False)

        mesh = trimesh.load(
            str(
                Path(args.mesh_root)
                / rel_path
                / "models/model_normalized.obj"
            ),
            process=False,
        )
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh_vertices = (
            np.asarray(mesh.vertices, dtype=np.float32) - normalize_center
        ) / max(normalize_scale, 1e-12)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int32)

        patches, seeds, point_indices, patch_distances = build_patches(
            noisy,
            args.patch_size,
            args.seed_k,
        )
        coarse_patches, refined_patches = predict_patch_batches(
            vm,
            refiner,
            patches,
            args.batch_size,
            args.coarse_mode,
            args.noise_std,
        )
        seeds_np = seeds.numpy().astype(np.float32, copy=False)
        coarse = fuse_patches(
            noisy,
            coarse_patches,
            seeds_np,
            point_indices,
            patch_distances,
            args.fusion_tau,
        )
        refined = fuse_patches(
            noisy,
            refined_patches,
            seeds_np,
            point_indices,
            patch_distances,
            args.fusion_tau,
        )

        coarse_score = score_prediction(
            noisy,
            clean,
            coarse,
            mesh_vertices,
            mesh_faces,
        )
        refined_score = score_prediction(
            noisy,
            clean,
            refined,
            mesh_vertices,
            mesh_faces,
        )
        row = {
            "rel_path": rel_path,
            "num_points": int(clean.shape[0]),
            "num_patches": int(patches.shape[0]),
            **{
                f"coarse_{key}": value
                for key, value in coarse_score.items()
            },
            **{
                f"refined_{key}": value
                for key, value in refined_score.items()
            },
        }
        row["cd_gain"] = (
            row["refined_cd_score"] - row["coarse_cd_score"]
        )
        row["p2s_gain"] = (
            row["refined_p2s_score"] - row["coarse_p2s_score"]
        )
        row["final_gain"] = (
            row["refined_final_score"] - row["coarse_final_score"]
        )
        rows.append(row)
        write_csv(out_dir / "shape_eval.csv", rows)

        coarse_path = coarse_dir / rel_path
        refined_path = refined_dir / rel_path
        coarse_path.mkdir(parents=True, exist_ok=True)
        refined_path.mkdir(parents=True, exist_ok=True)
        np.save(coarse_path / "denoised.npy", coarse)
        np.save(refined_path / "denoised.npy", refined)
        print(row, flush=True)

    summary = {
        "shape_count": len(rows),
        "coarse": {
            key: float(np.mean([row[f"coarse_{key}"] for row in rows]))
            for key in ["cd_score", "p2s_score", "final_score"]
        },
        "refined": {
            key: float(np.mean([row[f"refined_{key}"] for row in rows]))
            for key in ["cd_score", "p2s_score", "final_score"]
        },
        "gain": {
            key: quantile_summary([row[f"{key}_gain"] for row in rows])
            for key in ["cd", "p2s", "final"]
        },
        "improved_rate": float(
            np.mean([row["final_gain"] > 0.0 for row in rows])
        ),
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
