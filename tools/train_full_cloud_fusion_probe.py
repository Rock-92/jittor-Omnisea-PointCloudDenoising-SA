import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import jittor as jt
import numpy as np
import trimesh
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import (  # noqa: E402
    chamfer_distance,
    metric_to_score,
    point_to_surface_distance,
)
from src.data.utils import sample_vertex_groups  # noqa: E402
from tools.hard_patch_common import load_model, read_datalist  # noqa: E402
from tools.train_hard_patch_overfit import (  # noqa: E402
    collect_train_parameters,
    set_train_scope,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/vm_ssl/checkpoint_best.pkl"
)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_noise_bands(value):
    bands = []
    for item in str(value).split(","):
        lower, upper = item.split(":", maxsplit=1)
        lower = float(lower)
        upper = float(upper)
        if lower <= 0.0 or lower > upper:
            raise ValueError(f"invalid noise band: {item}")
        bands.append((lower, upper))
    if not bands:
        raise ValueError("at least one noise band is required")
    return bands


def category_of(rel_path):
    parts = Path(rel_path).parts
    return parts[1] if len(parts) > 1 else parts[0]


def choose_balanced_paths(paths, count, reference_paths, rng):
    paths = list(dict.fromkeys(paths))
    if count <= 0 or not paths:
        return []
    count = min(int(count), len(paths))
    available_counts = Counter(category_of(path) for path in paths)
    reference_counts = Counter(category_of(path) for path in reference_paths)
    weights = np.asarray(
        [
            reference_counts.get(category_of(path), 0)
            / max(available_counts[category_of(path)], 1)
            for path in paths
        ],
        dtype=np.float64,
    )
    if weights.sum() <= 0.0:
        weights = np.ones((len(paths),), dtype=np.float64)
    weights /= weights.sum()
    indices = rng.choice(
        len(paths),
        size=count,
        replace=False,
        p=weights,
    )
    return [paths[int(index)] for index in indices]


def normalize_clean(clean):
    center = (clean.max(axis=0) + clean.min(axis=0)) / 2.0
    centered = clean - center
    scale = np.sqrt((centered**2.0).sum(axis=1)).max()
    normalized = centered / max(float(scale), 1e-12)
    return (
        normalized.astype(np.float32, copy=False),
        center.astype(np.float32, copy=False),
        float(scale),
    )


def load_mesh(rel_path, mesh_root):
    mesh_path = Path(mesh_root) / rel_path / "models/model_normalized.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def load_shape(
    rel_path,
    clean_root,
    mesh_root,
    num_points,
    rng,
    sample_missing_clean,
):
    clean_path = Path(clean_root) / rel_path / "clean.npy"
    mesh = load_mesh(rel_path, mesh_root)
    if clean_path.exists():
        clean_raw = np.load(clean_path).astype(np.float32, copy=False)
    elif sample_missing_clean:
        clean_raw, _, _, _ = sample_vertex_groups(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            num_samples=max(32768, int(num_points)),
            num_vertex_samples=min(1024, int(num_points)),
        )
        clean_raw = clean_raw.astype(np.float32, copy=False)
    else:
        raise FileNotFoundError(clean_path)

    clean, center, scale = normalize_clean(clean_raw)
    if num_points > 0 and clean.shape[0] > num_points:
        indices = rng.choice(
            clean.shape[0],
            size=int(num_points),
            replace=False,
        )
        clean = clean[indices]
    vertices = (
        np.asarray(mesh.vertices, dtype=np.float32) - center[None, :]
    ) / max(scale, 1e-12)
    return {
        "rel_path": rel_path,
        "clean": clean.astype(np.float32, copy=False),
        "mesh_vertices": vertices.astype(np.float32, copy=False),
        "mesh_faces": np.asarray(mesh.faces, dtype=np.int32),
    }


def usable_paths(paths, clean_root, mesh_root, sample_missing_clean):
    usable = []
    clean_root = Path(clean_root)
    mesh_root = Path(mesh_root)
    for rel_path in dict.fromkeys(paths):
        mesh_path = mesh_root / rel_path / "models/model_normalized.obj"
        clean_path = clean_root / rel_path / "clean.npy"
        if mesh_path.exists() and (clean_path.exists() or sample_missing_clean):
            usable.append(rel_path)
    return usable


def farthest_point_indices(points, count, rng):
    count = min(max(int(count), 1), points.shape[0])
    selected = np.empty((count,), dtype=np.int64)
    selected[0] = int(rng.integers(points.shape[0]))
    min_dist = ((points - points[selected[0]]) ** 2.0).sum(axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(min_dist))
        distance = ((points - points[selected[index]]) ** 2.0).sum(axis=1)
        min_dist = np.minimum(min_dist, distance)
    return selected


def build_patch_layout(noisy, patch_size, seed_k, rng):
    point_count = noisy.shape[0]
    patch_size = min(int(patch_size), point_count)
    base_count = max(
        1,
        int(np.ceil(float(seed_k) * point_count / patch_size)),
    )
    tree = cKDTree(noisy)
    seed_indices = farthest_point_indices(noisy, base_count, rng).tolist()
    all_indices = []
    all_distances = []
    covered = np.zeros((point_count,), dtype=np.bool_)

    def append_seed(seed_index):
        distances, indices = tree.query(
            noisy[int(seed_index)],
            k=patch_size,
        )
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        distances = np.asarray(distances, dtype=np.float32).reshape(-1)
        all_indices.append(indices)
        all_distances.append(distances)
        covered[indices] = True

    for seed_index in seed_indices:
        append_seed(seed_index)
    while not covered.all():
        missing = np.flatnonzero(~covered)
        append_seed(int(missing[0]))
        seed_indices.append(int(missing[0]))

    point_indices = np.stack(all_indices)
    distances = np.stack(all_distances)
    normalized_distances = distances / np.maximum(
        distances[:, -1:],
        1e-8,
    )
    seeds = noisy[np.asarray(seed_indices, dtype=np.int64)]
    patches = noisy[point_indices] - seeds[:, None, :]
    return {
        "patches": patches.astype(np.float32, copy=False),
        "seeds": seeds.astype(np.float32, copy=False),
        "point_indices": point_indices.astype(np.int32, copy=False),
        "normalized_distances": normalized_distances.astype(
            np.float32,
            copy=False,
        ),
    }


def make_instance(shape, sigma, patch_size, seed_k, rng):
    clean = shape["clean"]
    noisy = clean + (
        rng.standard_normal(clean.shape).astype(np.float32) * float(sigma)
    )
    layout = build_patch_layout(
        noisy,
        patch_size=patch_size,
        seed_k=seed_k,
        rng=rng,
    )
    return {
        **shape,
        **layout,
        "noisy": noisy.astype(np.float32, copy=False),
        "sigma": float(sigma),
    }


def predict_and_fuse(
    model,
    instance,
    patch_batch_size,
    fusion_tau,
    sampler,
):
    patches_np = instance["patches"]
    sigma = float(instance["sigma"])
    predictions = []
    for start in range(0, patches_np.shape[0], patch_batch_size):
        end = min(start + patch_batch_size, patches_np.shape[0])
        patches = jt.array(patches_np[start:end])
        if sampler == "fixed":
            sigma_batch = jt.ones((end - start, 1)) * sigma
            predictions.append(
                model.predict_clean(patches, sigma=sigma_batch)
            )
        elif sampler == "heun":
            predictions.append(model.edm_heun_sampler(patches))
        else:
            raise ValueError(f"unsupported sampler: {sampler}")
    patch_prediction = jt.concat(predictions, dim=0)

    seeds = jt.array(instance["seeds"])
    absolute = patch_prediction + seeds.unsqueeze(1)
    point_indices_np = instance["point_indices"]
    weights_np = np.exp(
        -float(fusion_tau) * instance["normalized_distances"]
    ).astype(np.float32)
    flat_indices = jt.array(point_indices_np.reshape(-1)).int32()
    flat_absolute = absolute.reshape(-1, 3)
    flat_weights = jt.array(weights_np.reshape(-1, 1))
    index_3d = flat_indices.reshape(-1, 1).broadcast(flat_absolute.shape)
    weighted_sum = jt.zeros(
        (instance["noisy"].shape[0], 3)
    ).scatter(
        0,
        index_3d,
        flat_absolute * flat_weights,
        reduce="add",
    )
    weight_sum_np = np.zeros(
        (instance["noisy"].shape[0], 1),
        dtype=np.float32,
    )
    np.add.at(
        weight_sum_np[:, 0],
        point_indices_np.reshape(-1),
        weights_np.reshape(-1),
    )
    fused = weighted_sum / (jt.array(weight_sum_np) + 1e-8)
    return fused, patch_prediction


def pairwise_sqdist(a, b):
    return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2.0).sum(dim=-1)


def sampled_indices(point_count, max_points, rng):
    if max_points <= 0 or point_count <= max_points:
        return np.arange(point_count, dtype=np.int32)
    return rng.choice(
        point_count,
        size=int(max_points),
        replace=False,
    ).astype(np.int32)


def fusion_losses(
    fused,
    patch_prediction,
    instance,
    rng,
    chamfer_points,
):
    clean = jt.array(instance["clean"])
    noisy = jt.array(instance["noisy"])
    sigma2 = max(float(instance["sigma"]) ** 2.0, 1e-8)

    pred_indices = sampled_indices(
        fused.shape[0],
        chamfer_points,
        rng,
    )
    clean_indices = sampled_indices(
        clean.shape[0],
        chamfer_points,
        rng,
    )
    pred_sample = fused[jt.array(pred_indices).int32()]
    clean_sample = clean[jt.array(clean_indices).int32()]
    pred_to_clean = pairwise_sqdist(pred_sample, clean).min(dim=1)
    clean_to_pred = pairwise_sqdist(clean_sample, fused).min(dim=1)
    chamfer = (pred_to_clean.mean() + clean_to_pred.mean()) / sigma2

    paired = ((fused - clean) ** 2.0).sum(dim=-1).mean() / sigma2

    spacing_indices = sampled_indices(
        fused.shape[0],
        chamfer_points,
        rng,
    )
    spacing_indices_var = jt.array(spacing_indices).int32()
    pred_spacing_sample = fused[spacing_indices_var]
    clean_spacing_sample = clean[spacing_indices_var]
    pred_dist = pairwise_sqdist(
        pred_spacing_sample,
        pred_spacing_sample,
    )
    clean_dist = pairwise_sqdist(
        clean_spacing_sample,
        clean_spacing_sample,
    )
    diagonal = jt.array(
        np.eye(pred_dist.shape[0], dtype=np.float32) * 1e6
    )
    pred_nn = jt.sqrt((pred_dist + diagonal).min(dim=1) + 1e-12)
    clean_nn = jt.sqrt((clean_dist + diagonal).min(dim=1) + 1e-12)
    spacing = (
        jt.maximum(clean_nn - pred_nn, 0.0) ** 2.0
    ).mean() / sigma2

    point_indices = jt.array(instance["point_indices"]).int32()
    clean_patches = clean[point_indices]
    patch_paired = (
        (patch_prediction - (clean_patches - instance_seed_var(instance))) ** 2.0
    ).sum(dim=-1).mean() / sigma2
    noisy_paired = ((noisy - clean) ** 2.0).sum(dim=-1).mean() / sigma2
    return {
        "chamfer": chamfer,
        "paired": paired,
        "spacing": spacing,
        "patch_paired": patch_paired,
        "noisy_paired": noisy_paired,
    }


def instance_seed_var(instance):
    return jt.array(instance["seeds"]).unsqueeze(1)


def score_instance(instance, prediction):
    clean = instance["clean"]
    noisy = instance["noisy"]
    cd_noisy = chamfer_distance(noisy, clean, normalize=False)
    cd_pred = chamfer_distance(prediction, clean, normalize=False)
    p2s_noisy = point_to_surface_distance(
        noisy,
        instance["mesh_vertices"],
        instance["mesh_faces"],
    )
    p2s_pred = point_to_surface_distance(
        prediction,
        instance["mesh_vertices"],
        instance["mesh_faces"],
    )
    cd_score = metric_to_score(cd_pred, cd_noisy)
    p2s_score = metric_to_score(p2s_pred, p2s_noisy)
    return {
        "rel_path": instance["rel_path"],
        "sigma": float(instance["sigma"]),
        "noise_band": (
            f"{instance['noise_band'][0]:.3f}:"
            f"{instance['noise_band'][1]:.3f}"
        ),
        "patch_count": int(instance["patches"].shape[0]),
        "cd_score": float(cd_score),
        "p2s_score": float(p2s_score),
        "final_score": float(0.5 * (cd_score + p2s_score)),
    }


def evaluate(
    model,
    instances,
    patch_batch_size,
    fusion_tau,
    sampler,
):
    model.eval()
    rows = []
    with jt.no_grad():
        for index, instance in enumerate(instances):
            fused, _ = predict_and_fuse(
                model,
                instance,
                patch_batch_size=patch_batch_size,
                fusion_tau=fusion_tau,
                sampler=sampler,
            )
            prediction = fused.numpy().astype(np.float32, copy=False)
            row = score_instance(instance, prediction)
            rows.append(row)
            print(
                f"  eval [{index + 1}/{len(instances)}] "
                f"final={row['final_score']:.3f}",
                flush=True,
            )
    summary = {
        "count": len(rows),
        "cd_score": float(np.mean([row["cd_score"] for row in rows])),
        "p2s_score": float(np.mean([row["p2s_score"] for row in rows])),
        "final_score": float(
            np.mean([row["final_score"] for row in rows])
        ),
        "by_noise_band": {},
    }
    for band in sorted({row["noise_band"] for row in rows}):
        band_rows = [row for row in rows if row["noise_band"] == band]
        summary["by_noise_band"][band] = {
            "count": len(band_rows),
            "cd_score": float(
                np.mean([row["cd_score"] for row in band_rows])
            ),
            "p2s_score": float(
                np.mean([row["p2s_score"] for row in band_rows])
            ),
            "final_score": float(
                np.mean([row["final_score"] for row in band_rows])
            ),
        }
    return rows, summary


def assign_bands(shapes, bands, rng):
    band_indices = np.arange(len(shapes)) % len(bands)
    rng.shuffle(band_indices)
    return [
        {
            **shape,
            "noise_band": bands[int(band_index)],
        }
        for shape, band_index in zip(shapes, band_indices)
    ]


def fixed_instances(shapes, patch_size, seed_k, rng):
    instances = []
    for shape in shapes:
        lower, upper = shape["noise_band"]
        sigma = float(rng.uniform(lower, upper))
        instances.append(
            make_instance(
                shape,
                sigma=sigma,
                patch_size=patch_size,
                seed_k=seed_k,
                rng=rng,
            )
        )
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--train-list", default="datalist/train.txt")
    parser.add_argument("--val-list", default="datalist/validate.txt")
    parser.add_argument(
        "--category-reference-list",
        default="datalist/test.txt",
    )
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument(
        "--out-dir",
        default="outputs/full_cloud_fusion_probe_v1",
    )
    parser.add_argument("--train-shapes", type=int, default=20)
    parser.add_argument("--train-eval-shapes", type=int, default=5)
    parser.add_argument("--val-shapes", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed-k", type=float, default=3.0)
    parser.add_argument("--patch-batch-size", type=int, default=2)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument(
        "--train-sampler",
        choices=["fixed", "heun"],
        default="fixed",
    )
    parser.add_argument(
        "--eval-sampler",
        choices=["fixed", "heun"],
        default="heun",
    )
    parser.add_argument(
        "--noise-bands",
        default="0.005:0.010,0.010:0.015,0.015:0.020",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--train-scope",
        choices=["decoder", "all"],
        default="decoder",
    )
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--paired-weight", type=float, default=0.2)
    parser.add_argument("--spacing-weight", type=float, default=0.2)
    parser.add_argument("--patch-weight", type=float, default=0.1)
    parser.add_argument("--chamfer-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    selection_rng = np.random.default_rng(args.seed)
    train_rng = np.random.default_rng(args.seed + 1)
    val_rng = np.random.default_rng(args.seed + 2)
    noise_bands = parse_noise_bands(args.noise_bands)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    train_candidates = usable_paths(
        read_datalist(args.train_list),
        clean_root=args.clean_root,
        mesh_root=args.mesh_root,
        sample_missing_clean=args.sample_missing_clean,
    )
    val_candidates = usable_paths(
        read_datalist(args.val_list),
        clean_root=args.clean_root,
        mesh_root=args.mesh_root,
        sample_missing_clean=args.sample_missing_clean,
    )
    reference_paths = read_datalist(args.category_reference_list)
    train_paths = choose_balanced_paths(
        train_candidates,
        args.train_shapes,
        reference_paths,
        selection_rng,
    )
    val_paths = choose_balanced_paths(
        [path for path in val_candidates if path not in set(train_paths)],
        args.val_shapes,
        reference_paths,
        selection_rng,
    )
    if not train_paths or not val_paths:
        raise FileNotFoundError(
            "no usable train/validation shapes; generate cache_clean_points "
            "or pass --sample-missing-clean"
        )

    print("Loading selected shapes...", flush=True)
    train_shapes = [
        load_shape(
            path,
            clean_root=args.clean_root,
            mesh_root=args.mesh_root,
            num_points=args.num_points,
            rng=selection_rng,
            sample_missing_clean=args.sample_missing_clean,
        )
        for path in train_paths
    ]
    val_shapes = [
        load_shape(
            path,
            clean_root=args.clean_root,
            mesh_root=args.mesh_root,
            num_points=args.num_points,
            rng=selection_rng,
            sample_missing_clean=args.sample_missing_clean,
        )
        for path in val_paths
    ]
    train_shapes = assign_bands(train_shapes, noise_bands, selection_rng)
    val_shapes = assign_bands(val_shapes, noise_bands, selection_rng)
    val_instances = fixed_instances(
        val_shapes,
        patch_size=args.patch_size,
        seed_k=args.seed_k,
        rng=val_rng,
    )
    train_eval_instances = fixed_instances(
        train_shapes[: min(args.train_eval_shapes, len(train_shapes))],
        patch_size=args.patch_size,
        seed_k=args.seed_k,
        rng=np.random.default_rng(args.seed + 3),
    )
    write_json(
        out_dir / "selection.json",
        {
            "train": [
                {
                    "rel_path": shape["rel_path"],
                    "noise_band": shape["noise_band"],
                }
                for shape in train_shapes
            ],
            "val": [
                {
                    "rel_path": instance["rel_path"],
                    "sigma": instance["sigma"],
                }
                for instance in val_instances
            ],
            "args": vars(args),
        },
    )

    model = load_model(checkpoint)
    set_train_scope(model, args.train_scope)
    optimizer = jt.optim.Adam(
        collect_train_parameters(model, args.train_scope),
        lr=args.lr,
    )

    print("Baseline held-out evaluation:", flush=True)
    baseline_rows, baseline_summary = evaluate(
        model,
        val_instances,
        patch_batch_size=args.patch_batch_size,
        fusion_tau=args.fusion_tau,
        sampler=args.eval_sampler,
    )
    write_csv(out_dir / "baseline_val.csv", baseline_rows)
    write_json(out_dir / "baseline_summary.json", baseline_summary)
    write_csv(out_dir / "best_val.csv", baseline_rows)
    write_json(out_dir / "best_summary.json", baseline_summary)
    print("Baseline train-shape evaluation:", flush=True)
    train_baseline_rows, train_baseline_summary = evaluate(
        model,
        train_eval_instances,
        patch_batch_size=args.patch_batch_size,
        fusion_tau=args.fusion_tau,
        sampler=args.eval_sampler,
    )
    write_csv(out_dir / "baseline_train.csv", train_baseline_rows)
    write_json(
        out_dir / "baseline_train_summary.json",
        train_baseline_summary,
    )
    model.save(str(out_dir / "checkpoint_best.pkl"))

    best_score = baseline_summary["final_score"]
    best_val_summary = baseline_summary
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = train_rng.permutation(len(train_shapes))
        epoch_metrics = []
        for shape_index in order:
            shape = train_shapes[int(shape_index)]
            lower, upper = shape["noise_band"]
            sigma = float(train_rng.uniform(lower, upper))
            instance = make_instance(
                shape,
                sigma=sigma,
                patch_size=args.patch_size,
                seed_k=args.seed_k,
                rng=train_rng,
            )
            fused, patch_prediction = predict_and_fuse(
                model,
                instance,
                patch_batch_size=args.patch_batch_size,
                fusion_tau=args.fusion_tau,
                sampler=args.train_sampler,
            )
            losses = fusion_losses(
                fused,
                patch_prediction,
                instance,
                rng=train_rng,
                chamfer_points=args.chamfer_points,
            )
            loss = (
                args.chamfer_weight * losses["chamfer"]
                + args.paired_weight * losses["paired"]
                + args.spacing_weight * losses["spacing"]
                + args.patch_weight * losses["patch_paired"]
            )
            optimizer.step(loss)
            epoch_metrics.append(
                {
                    key: float(value.item())
                    for key, value in {"loss": loss, **losses}.items()
                }
            )
            jt.gc()

        record = {
            "epoch": epoch,
            **{
                f"train_{key}": float(
                    np.mean([row[key] for row in epoch_metrics])
                )
                for key in epoch_metrics[0]
            },
        }
        should_evaluate = (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        )
        if should_evaluate:
            print(f"Epoch {epoch} held-out evaluation:", flush=True)
            val_rows, val_summary = evaluate(
                model,
                val_instances,
                patch_batch_size=args.patch_batch_size,
                fusion_tau=args.fusion_tau,
                sampler=args.eval_sampler,
            )
            record.update(
                {
                    "val_cd_score": val_summary["cd_score"],
                    "val_p2s_score": val_summary["p2s_score"],
                    "val_final_score": val_summary["final_score"],
                    "val_gain": (
                        val_summary["final_score"]
                        - baseline_summary["final_score"]
                    ),
                }
            )
            if val_summary["final_score"] > best_score:
                best_score = val_summary["final_score"]
                best_val_summary = val_summary
                best_epoch = epoch
                model.save(str(out_dir / "checkpoint_best.pkl"))
                write_csv(out_dir / "best_val.csv", val_rows)
                write_json(out_dir / "best_summary.json", val_summary)
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        print(record, flush=True)

    model.save(str(out_dir / "checkpoint_last.pkl"))
    model.load(str(out_dir / "checkpoint_best.pkl"))
    best_train_rows, best_train_summary = evaluate(
        model,
        train_eval_instances,
        patch_batch_size=args.patch_batch_size,
        fusion_tau=args.fusion_tau,
        sampler=args.eval_sampler,
    )
    write_csv(out_dir / "best_train.csv", best_train_rows)
    write_json(out_dir / "best_train_summary.json", best_train_summary)
    train_gain = (
        best_train_summary["final_score"]
        - train_baseline_summary["final_score"]
    )
    val_gain = best_score - baseline_summary["final_score"]
    cd_gain = (
        best_val_summary["cd_score"] - baseline_summary["cd_score"]
    )
    p2s_gain = (
        best_val_summary["p2s_score"] - baseline_summary["p2s_score"]
    )
    band_gains = {
        band: (
            best_val_summary["by_noise_band"][band]["final_score"]
            - baseline_summary["by_noise_band"][band]["final_score"]
        )
        for band in baseline_summary["by_noise_band"]
    }
    go_no_go = {
        "decision": (
            "expand"
            if (
                train_gain >= 8.0
                and val_gain >= 6.0
                and cd_gain >= 8.0
                and p2s_gain >= -1.0
            )
            else "stop_or_revise"
        ),
        "criteria": {
            "train_gain_ge_8": train_gain >= 8.0,
            "val_gain_ge_6": val_gain >= 6.0,
            "val_cd_gain_ge_8": cd_gain >= 8.0,
            "val_p2s_gain_ge_minus_1": p2s_gain >= -1.0,
        },
    }
    write_json(
        out_dir / "train_summary.json",
        {
            "checkpoint": str(checkpoint.resolve()),
            "baseline": baseline_summary,
            "train_baseline": train_baseline_summary,
            "train_best": best_train_summary,
            "train_gain": train_gain,
            "best_epoch": best_epoch,
            "best_final_score": best_score,
            "best_gain": val_gain,
            "best_cd_gain": cd_gain,
            "best_p2s_gain": p2s_gain,
            "best_band_gains": band_gains,
            "go_no_go": go_no_go,
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
