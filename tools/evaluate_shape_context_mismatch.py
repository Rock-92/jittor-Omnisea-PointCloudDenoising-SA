import argparse
import csv
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (  # noqa: E402
    chamfer_distance,
    metric_to_score,
    point_to_surface_distance,
)
from src.model.shape_ssl import (  # noqa: E402
    ShapeContextVelocityModule,
    build_region_layout,
    region_arrays,
)
from tools.hard_patch_common import read_datalist  # noqa: E402
from tools.train_full_cloud_fusion_probe import (  # noqa: E402
    farthest_point_indices,
    load_shape,
    usable_paths,
)


CONDITIONS = (
    "matched",
    "donor_remapped",
    "permuted",
    "zero",
)


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


def make_noisy_shape(shape, sigma, rng, region_count, points_per_region):
    noise = rng.laplace(
        0.0,
        float(sigma),
        size=shape["clean"].shape,
    ).astype(np.float32)
    noisy = (shape["clean"] + noise).astype(np.float32, copy=False)
    center_indices, neighbor_indices = build_region_layout(
        noisy,
        region_count,
        points_per_region,
    )
    region_points, region_centers = region_arrays(
        noisy,
        center_indices,
        neighbor_indices,
    )
    return {
        **shape,
        "noisy": noisy,
        "sigma": float(sigma),
        "region_points": region_points,
        "region_centers": region_centers,
    }


def build_patch_batch(instance, patch_count, patch_size):
    noisy = instance["noisy"]
    clean = instance["clean"]
    tree = cKDTree(noisy)
    seed_indices = farthest_point_indices(noisy, int(patch_count))
    patches = []
    clean_patches = []
    seeds = []
    for seed_index in seed_indices:
        seed = noisy[int(seed_index)]
        _, point_indices = tree.query(
            seed,
            k=min(int(patch_size), noisy.shape[0]),
        )
        point_indices = np.asarray(
            point_indices,
            dtype=np.int64,
        ).reshape(-1)
        patches.append(noisy[point_indices] - seed[None, :])
        clean_patches.append(clean[point_indices])
        seeds.append(seed)
    return {
        "patches": np.asarray(patches, dtype=np.float32),
        "clean_absolute": np.asarray(clean_patches, dtype=np.float32),
        "seeds": np.asarray(seeds, dtype=np.float32),
    }


def encode_regions(model, instance):
    region_points = jt.array(instance["region_points"][None, ...])
    region_centers = jt.array(instance["region_centers"][None, ...])
    return model.encode_shape(region_points, region_centers)


def remap_donor_tokens(donor_tokens, donor_centers, target_centers):
    _, donor_indices = cKDTree(donor_centers).query(
        target_centers,
        k=1,
    )
    donor_indices = np.asarray(donor_indices, dtype=np.int32)
    return donor_tokens[:, donor_indices, :]


def condition_tokens(
    condition,
    target_tokens,
    donor_tokens,
    target_instance,
    donor_instance,
    permutation,
):
    if condition == "matched":
        return target_tokens
    if condition == "donor_remapped":
        return remap_donor_tokens(
            donor_tokens,
            donor_instance["region_centers"],
            target_instance["region_centers"],
        )
    if condition == "permuted":
        return target_tokens[:, permutation, :]
    if condition == "zero":
        return jt.zeros_like(target_tokens)
    raise ValueError(f"unsupported condition: {condition}")


def predict_condition(
    model,
    patch_batch,
    target_instance,
    encoded_tokens,
    batch_size,
):
    outputs = []
    region_points = jt.array(target_instance["region_points"][None, ...])
    region_centers = jt.array(target_instance["region_centers"][None, ...])
    patches = patch_batch["patches"]
    seeds = patch_batch["seeds"]
    for start in range(0, patches.shape[0], int(batch_size)):
        end = min(start + int(batch_size), patches.shape[0])
        count = end - start
        tokens = encoded_tokens.broadcast(
            (
                count,
                encoded_tokens.shape[1],
                encoded_tokens.shape[2],
            )
        )
        centers = region_centers.broadcast(
            (count, region_centers.shape[1], 3)
        )
        points = region_points.broadcast(
            (
                count,
                region_points.shape[1],
                region_points.shape[2],
                3,
            )
        )
        prediction, _ = model.predict_displacement_context(
            jt.array(patches[start:end]),
            jt.array(seeds[start:end, None, :]),
            points,
            centers,
            encoded_shape=tokens,
        )
        absolute = (
            prediction
            + jt.array(patches[start:end])
            + jt.array(seeds[start:end, None, :])
        )
        outputs.append(absolute.numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def score_patch(noisy, clean, prediction, mesh_vertices, mesh_faces):
    cd_noisy = chamfer_distance(noisy, clean, normalize=False)
    cd_pred = chamfer_distance(prediction, clean, normalize=False)
    p2s_noisy = point_to_surface_distance(
        noisy,
        mesh_vertices,
        mesh_faces,
        normalize_ref_pc=None,
    )
    p2s_pred = point_to_surface_distance(
        prediction,
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
        "cd_pred": float(cd_pred),
        "p2s_pred": float(p2s_pred),
    }


def shape_gap_summary(rows, condition, seed):
    patch_scores = {}
    for row in rows:
        key = (row["shape_index"], row["patch_index"])
        patch_scores.setdefault(key, {})[row["condition"]] = row[
            "final_score"
        ]
    gaps_by_shape = {}
    for (shape_index, _), scores in patch_scores.items():
        gaps_by_shape.setdefault(shape_index, []).append(
            scores["matched"] - scores[condition]
        )
    shape_gaps = np.asarray(
        [
            np.mean(gaps_by_shape[index])
            for index in sorted(gaps_by_shape)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            rng.choice(
                shape_gaps,
                size=shape_gaps.size,
                replace=True,
            ).mean()
            for _ in range(20000)
        ],
        dtype=np.float64,
    )
    return {
        "shape_count": int(shape_gaps.size),
        "mean": float(shape_gaps.mean()),
        "std": float(shape_gaps.std(ddof=1)),
        "median": float(np.median(shape_gaps)),
        "positive_shape_rate": float(np.mean(shape_gaps > 0.0)),
        "bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def summarize(rows, include_shape_stats=False, seed=123):
    conditions = {}
    for condition in CONDITIONS:
        condition_rows = [
            row for row in rows if row["condition"] == condition
        ]
        conditions[condition] = {
            metric: float(
                np.mean([row[metric] for row in condition_rows])
            )
            for metric in ("cd_score", "p2s_score", "final_score")
        }
    matched = conditions["matched"]
    comparisons = {}
    for condition in CONDITIONS[1:]:
        other = conditions[condition]
        pair_gaps = []
        for row in rows:
            if row["condition"] != "matched":
                continue
            other_row = next(
                candidate
                for candidate in rows
                if candidate["shape_index"] == row["shape_index"]
                and candidate["patch_index"] == row["patch_index"]
                and candidate["condition"] == condition
            )
            pair_gaps.append(
                row["final_score"] - other_row["final_score"]
            )
        comparisons[f"matched_minus_{condition}"] = {
            "cd_score": float(
                matched["cd_score"] - other["cd_score"]
            ),
            "p2s_score": float(
                matched["p2s_score"] - other["p2s_score"]
            ),
            "final_score": float(
                matched["final_score"] - other["final_score"]
            ),
            "paired_win_rate": float(
                np.mean(np.asarray(pair_gaps) > 0.0)
            ),
            "paired_final_gap_std": float(np.std(pair_gaps)),
        }
        if include_shape_stats:
            comparisons[f"matched_minus_{condition}"][
                "shape_level"
            ] = shape_gap_summary(
                rows,
                condition,
                seed + CONDITIONS.index(condition),
            )
    return {
        "patch_count": int(len(rows) // len(CONDITIONS)),
        "conditions": conditions,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/shape_context_vm/checkpoints/checkpoint_best.pkl",
    )
    parser.add_argument(
        "--shape-pretrained-checkpoint",
        default="outputs/shape_pretrain/checkpoints/processor_best.pkl",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/shape_context_vm.yaml",
    )
    parser.add_argument(
        "--transform-config",
        default="configs/transform/shape_context_vm_laplace.yaml",
    )
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument(
        "--out-dir",
        default="outputs/shape_context_mismatch_eval",
    )
    parser.add_argument("--max-shapes", type=int, default=20)
    parser.add_argument("--patches-per-shape", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config = OmegaConf.to_container(
        OmegaConf.load(args.model_config),
        resolve=True,
    )
    model_config.pop("__target__", None)
    model_config["shape_pretrained_ckpt"] = (
        args.shape_pretrained_checkpoint
    )
    transform_config = OmegaConf.to_container(
        OmegaConf.load(args.transform_config),
        resolve=True,
    )
    model = ShapeContextVelocityModule(
        model_config,
        transform_config,
    )
    model.load(args.checkpoint)
    model.eval()

    candidates = usable_paths(
        read_datalist(args.datalist),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    if len(candidates) < 2:
        raise RuntimeError("mismatch evaluation requires at least two shapes")
    paths = candidates[: min(int(args.max_shapes), len(candidates))]

    instances = []
    print("Loading fixed noisy shapes and region layouts...", flush=True)
    for rel_path in tqdm(paths, unit="shape"):
        shape = load_shape(
            rel_path,
            args.clean_root,
            args.mesh_root,
            args.num_points,
            rng,
            args.sample_missing_clean,
        )
        sigma = float(rng.uniform(args.sigma_min, args.sigma_max))
        instances.append(
            make_noisy_shape(
                shape,
                sigma,
                rng,
                model.region_count,
                model.points_per_region,
            )
        )

    rows = []
    with jt.no_grad():
        encoded = [encode_regions(model, instance) for instance in instances]
        for shape_index, target in enumerate(instances):
            donor_index = (shape_index + 1) % len(instances)
            donor = instances[donor_index]
            patch_batch = build_patch_batch(
                target,
                args.patches_per_shape,
                args.patch_size,
            )
            permutation = rng.permutation(
                target["region_centers"].shape[0]
            ).astype(np.int32)
            predictions = {}
            for condition in CONDITIONS:
                print(
                    f"  shape [{shape_index + 1}/{len(instances)}] "
                    f"condition={condition}",
                    flush=True,
                )
                tokens = condition_tokens(
                    condition,
                    encoded[shape_index],
                    encoded[donor_index],
                    target,
                    donor,
                    permutation,
                )
                predictions[condition] = predict_condition(
                    model,
                    patch_batch,
                    target,
                    tokens,
                    args.batch_size,
                )

            for patch_index in range(patch_batch["patches"].shape[0]):
                noisy_absolute = (
                    patch_batch["patches"][patch_index]
                    + patch_batch["seeds"][patch_index][None, :]
                )
                clean_absolute = patch_batch["clean_absolute"][patch_index]
                for condition in CONDITIONS:
                    score = score_patch(
                        noisy_absolute,
                        clean_absolute,
                        predictions[condition][patch_index],
                        target["mesh_vertices"],
                        target["mesh_faces"],
                    )
                    rows.append(
                        {
                            "shape_index": shape_index,
                            "patch_index": patch_index,
                            "rel_path": target["rel_path"],
                            "donor_rel_path": donor["rel_path"],
                            "sigma": target["sigma"],
                            "condition": condition,
                            **score,
                        }
                    )
            partial = summarize(rows)
            gap = partial["comparisons"][
                "matched_minus_donor_remapped"
            ]["final_score"]
            print(
                f"[{shape_index + 1}/{len(instances)}] "
                f"matched={partial['conditions']['matched']['final_score']:.3f} "
                f"donor={partial['conditions']['donor_remapped']['final_score']:.3f} "
                f"gap={gap:+.3f}",
                flush=True,
            )
            jt.gc()

    summary = summarize(
        rows,
        include_shape_stats=True,
        seed=args.seed,
    )
    summary["args"] = vars(args)
    write_csv(out_dir / "rows.csv", rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
