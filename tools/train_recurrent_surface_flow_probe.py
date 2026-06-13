import argparse
import csv
import json
import math
import sys
from pathlib import Path

import jittor as jt
import numpy as np
import point_cloud_utils as pcu
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import metric_to_score  # noqa: E402
from src.data.utils import sample_vertex_groups  # noqa: E402
from tools.build_refinement_probe_dataset import choose_paths  # noqa: E402
from tools.hard_patch_common import (  # noqa: E402
    chamfer,
    load_model,
    normalize_pc,
    read_datalist,
)
from tools.train_hard_patch_overfit import (  # noqa: E402
    collect_train_parameters,
    set_train_scope,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs_result/outputs2.1/checkpoints/vm/checkpoint_best.pkl"
)
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/vm_pure_global.yaml"
DEFAULT_TRANSFORM_CONFIG = (
    PROJECT_ROOT / "configs/transform/vm_pure_laplace.yaml"
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


def validate_checkpoint(model, checkpoint):
    checkpoint_state = jt.load(str(checkpoint))
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(checkpoint_state))
    extra = sorted(set(checkpoint_state) - set(model_state))
    mismatch = sorted(
        key
        for key in set(model_state) & set(checkpoint_state)
        if tuple(model_state[key].shape)
        != tuple(checkpoint_state[key].shape)
    )
    result = {
        "checkpoint_parameters": len(checkpoint_state),
        "model_parameters": len(model_state),
        "missing": missing,
        "extra": extra,
        "shape_mismatch": mismatch,
        "compatible": not missing and not extra and not mismatch,
    }
    if not result["compatible"]:
        raise RuntimeError(
            "pure VM checkpoint mismatch: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"shape_mismatch={len(mismatch)}"
        )
    return result


def normalize_shape(clean_raw, mesh):
    p_max = clean_raw.max(axis=0)
    p_min = clean_raw.min(axis=0)
    center = (p_max + p_min) / 2.0
    centered = clean_raw - center
    scale = np.sqrt((centered**2.0).sum(axis=1)).max()
    scale = max(float(scale), 1e-12)
    clean = (centered / scale).astype(np.float32, copy=False)
    vertices = (
        (np.asarray(mesh.vertices, dtype=np.float32) - center) / scale
    ).astype(np.float32, copy=False)
    return clean, vertices


def load_shape(
    rel_path,
    clean_root,
    mesh_root,
    sample_missing_clean=False,
):
    clean_path = Path(clean_root) / rel_path / "clean.npy"
    mesh_path = (
        Path(mesh_root)
        / rel_path
        / "models/model_normalized.obj"
    )
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if clean_path.exists():
        clean_raw = np.load(clean_path).astype(
            np.float32,
            copy=False,
        )
    elif sample_missing_clean:
        clean_raw, _, _, _ = sample_vertex_groups(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            num_samples=32768,
            num_vertex_samples=1024,
        )
        clean_raw = clean_raw.astype(np.float32, copy=False)
    else:
        raise FileNotFoundError(clean_path)
    clean, vertices = normalize_shape(clean_raw, mesh)
    return {
        "rel_path": rel_path,
        "clean": clean,
        "vertices": vertices,
        "faces": np.asarray(mesh.faces, dtype=np.int32),
    }


def usable_paths(
    paths,
    clean_root,
    mesh_root,
    sample_missing_clean=False,
):
    return [
        path
        for path in dict.fromkeys(paths)
        if (
            sample_missing_clean
            or (Path(clean_root) / path / "clean.npy").exists()
        )
        and (
            Path(mesh_root)
            / path
            / "models/model_normalized.obj"
        ).exists()
    ]


def sample_patch(shape, sigma, patch_size, rng):
    clean = shape["clean"]
    noise = rng.laplace(
        0.0,
        float(sigma),
        size=clean.shape,
    ).astype(np.float32)
    noisy = (clean + noise).astype(np.float32, copy=False)
    seed_index = int(rng.integers(noisy.shape[0]))
    center = noisy[seed_index]
    _, indices = cKDTree(noisy).query(
        center,
        k=min(int(patch_size), noisy.shape[0]),
    )
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    return {
        "rel_path": shape["rel_path"],
        "sigma": float(sigma),
        "noisy": (noisy[indices] - center).astype(
            np.float32,
            copy=False,
        ),
        "clean": (clean[indices] - center).astype(
            np.float32,
            copy=False,
        ),
        "vertices": (shape["vertices"] - center).astype(
            np.float32,
            copy=False,
        ),
        "faces": shape["faces"],
    }


def closest_surface(points, vertices, faces):
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise FloatingPointError("surface query received non-finite points")
    _, face_ids, barycentric = pcu.closest_points_on_mesh(
        points,
        vertices,
        faces,
    )
    face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    barycentric = np.asarray(
        barycentric,
        dtype=np.float32,
    ).reshape(-1, 3)
    target = pcu.interpolate_barycentric_coords(
        faces,
        face_ids,
        barycentric,
        vertices,
    ).astype(np.float32, copy=False)
    finite = np.isfinite(target).all(axis=1)
    if not finite.all():
        _, vertex_ids = cKDTree(vertices).query(points[~finite], k=1)
        target = target.copy()
        target[~finite] = vertices[
            np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
        ]
    if not np.isfinite(target).all():
        raise FloatingPointError("surface query produced non-finite targets")
    return target


def optimizer_grad_norm(optimizer):
    grads = []
    for group in optimizer.param_groups:
        for param, grad in zip(group["params"], group["grads"]):
            if param.is_stop_grad():
                continue
            grads.append(grad.flatten())
    if not grads:
        return 0.0
    return float(jt.norm(jt.concat(grads), 2).item())


def pairwise_sqdist(a, b):
    return ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2.0).sum(dim=-1)


def recurrent_forward(
    model,
    noisy,
    vertices,
    faces,
    steps,
    step_scale,
    chamfer_points,
    rng,
    with_loss,
):
    state = noisy
    trajectory = [state]
    step_losses = []
    for step in range(int(steps)):
        state_np = state.detach().numpy()[0].astype(
            np.float32,
            copy=False,
        )
        target_np = closest_surface(state_np, vertices, faces)
        target = jt.array(target_np[None, :, :])
        target_flow = target - state.detach()
        predicted_flow = model.predict_displacement(state)
        next_state = state + float(step_scale) * predicted_flow
        trajectory.append(next_state)
        if with_loss:
            current_surface_mse = (
                (state.detach() - target) ** 2.0
            ).sum(dim=-1).mean()
            flow_loss = (
                (predicted_flow - target_flow) ** 2.0
            ).sum(dim=-1).mean()
            surface_mse = (
                (next_state - target) ** 2.0
            ).sum(dim=-1).mean()
            monotonic = jt.maximum(
                surface_mse - current_surface_mse,
                0.0,
            )
            step_losses.append(
                {
                    "flow": flow_loss,
                    "surface": surface_mse,
                    "monotonic": monotonic.mean(),
                }
            )
        state = next_state

    losses = None
    if with_loss:
        point_count = state.shape[1]
        sample_count = min(int(chamfer_points), point_count)
        indices = rng.choice(
            point_count,
            size=sample_count,
            replace=False,
        ).astype(np.int32)
        final_sample = state[0, jt.array(indices).int32(), :]
        final_target_np = closest_surface(
            final_sample.detach().numpy(),
            vertices,
            faces,
        )
        final_target = jt.array(final_target_np)
        final_surface = (
            (final_sample - final_target) ** 2.0
        ).sum(dim=-1).mean()
        losses = {
            "flow": jt.stack(
                [item["flow"] for item in step_losses]
            ).mean(),
            "surface": jt.stack(
                [item["surface"] for item in step_losses]
            ).mean(),
            "monotonic": jt.stack(
                [item["monotonic"] for item in step_losses]
            ).mean(),
            "final_surface": final_surface,
        }
    return state, trajectory, losses


def surface_mse(points, vertices, faces):
    target = closest_surface(points, vertices, faces)
    return float(((points - target) ** 2.0).sum(axis=1).mean())


def score_patch(patch, prediction):
    noisy_cd = chamfer(patch["noisy"], patch["clean"])
    pred_cd = chamfer(prediction, patch["clean"])
    noisy_p2s = surface_mse(
        patch["noisy"],
        patch["vertices"],
        patch["faces"],
    )
    pred_p2s = surface_mse(
        prediction,
        patch["vertices"],
        patch["faces"],
    )
    cd_score = metric_to_score(pred_cd, noisy_cd)
    p2s_score = metric_to_score(pred_p2s, noisy_p2s)
    return {
        "cd_score": float(cd_score),
        "p2s_score": float(p2s_score),
        "final_score": float(0.5 * (cd_score + p2s_score)),
        "p2s_mse": pred_p2s,
    }


def predict_patch(model, patch, steps, step_scale):
    model.eval()
    state = jt.array(patch["noisy"][None, :, :])
    with jt.no_grad():
        prediction, trajectory, _ = recurrent_forward(
            model,
            state,
            patch["vertices"],
            patch["faces"],
            steps=steps,
            step_scale=step_scale,
            chamfer_points=0,
            rng=np.random.default_rng(0),
            with_loss=False,
        )
    return (
        prediction.numpy()[0].astype(np.float32, copy=False),
        [
            item.numpy()[0].astype(np.float32, copy=False)
            for item in trajectory
        ],
    )


def evaluate(model, patches, steps, step_scale):
    rows = []
    for index, patch in enumerate(patches):
        one_step, _ = predict_patch(
            model,
            patch,
            steps=1,
            step_scale=1.0,
        )
        recurrent, trajectory = predict_patch(
            model,
            patch,
            steps=steps,
            step_scale=step_scale,
        )
        base_score = score_patch(patch, one_step)
        recurrent_score = score_patch(patch, recurrent)
        trajectory_p2s = [
            surface_mse(
                state,
                patch["vertices"],
                patch["faces"],
            )
            for state in trajectory
        ]
        monotonic = all(
            right <= left + 1e-12
            for left, right in zip(
                trajectory_p2s[:-1],
                trajectory_p2s[1:],
            )
        )
        row = {
            "index": index,
            "rel_path": patch["rel_path"],
            "sigma": patch["sigma"],
            "baseline_final": base_score["final_score"],
            "recurrent_final": recurrent_score["final_score"],
            "final_gain": (
                recurrent_score["final_score"]
                - base_score["final_score"]
            ),
            "baseline_cd": base_score["cd_score"],
            "recurrent_cd": recurrent_score["cd_score"],
            "baseline_p2s": base_score["p2s_score"],
            "recurrent_p2s": recurrent_score["p2s_score"],
            "p2s_monotonic": bool(monotonic),
        }
        rows.append(row)
        print(
            f"  eval [{index + 1}/{len(patches)}] "
            f"base={row['baseline_final']:.3f} "
            f"recurrent={row['recurrent_final']:.3f}",
            flush=True,
        )
    return rows, {
        "count": len(rows),
        "baseline_final": float(
            np.mean([row["baseline_final"] for row in rows])
        ),
        "recurrent_final": float(
            np.mean([row["recurrent_final"] for row in rows])
        ),
        "final_gain": float(
            np.mean([row["final_gain"] for row in rows])
        ),
        "cd_gain": float(
            np.mean(
                [
                    row["recurrent_cd"] - row["baseline_cd"]
                    for row in rows
                ]
            )
        ),
        "p2s_gain": float(
            np.mean(
                [
                    row["recurrent_p2s"] - row["baseline_p2s"]
                    for row in rows
                ]
            )
        ),
        "monotonic_rate": float(
            np.mean([row["p2s_monotonic"] for row in rows])
        ),
    }


def add_original_baseline_comparison(rows, summary, original_rows):
    original_by_index = {
        int(row["index"]): row
        for row in original_rows
    }
    gains = []
    for row in rows:
        original = original_by_index[int(row["index"])]
        row["original_vm_final"] = original["baseline_final"]
        row["gain_vs_original_vm"] = (
            row["recurrent_final"] - original["baseline_final"]
        )
        gains.append(row["gain_vs_original_vm"])
    summary["gain_vs_original_vm"] = float(np.mean(gains))
    return rows, summary


def build_eval_patches(
    shapes,
    patches_per_shape,
    patch_size,
    noise_min,
    noise_max,
    rng,
):
    candidates = []
    for shape in shapes:
        for _ in range(int(patches_per_shape)):
            sigma = float(rng.uniform(noise_min, noise_max))
            candidates.append(
                sample_patch(
                    shape,
                    sigma=sigma,
                    patch_size=patch_size,
                    rng=rng,
                )
            )
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
    )
    parser.add_argument(
        "--transform-config",
        default=str(DEFAULT_TRANSFORM_CONFIG),
    )
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
        default="outputs/recurrent_surface_flow_probe_v1",
    )
    parser.add_argument("--train-shapes", type=int, default=20)
    parser.add_argument("--val-shapes", type=int, default=10)
    parser.add_argument("--train-patches-per-shape", type=int, default=4)
    parser.add_argument("--val-patches-per-shape", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--noise-min", type=float, default=0.005)
    parser.add_argument("--noise-max", type=float, default=0.020)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--step-scale", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--train-scope",
        choices=["decoder", "all"],
        default="all",
    )
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--surface-weight", type=float, default=1.0)
    parser.add_argument("--monotonic-weight", type=float, default=0.5)
    parser.add_argument("--final-surface-weight", type=float, default=1.0)
    parser.add_argument("--chamfer-points", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    if args.patch_size != 1000:
        raise ValueError("surface-flow probe must keep patch_size=1000")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    selection_rng = np.random.default_rng(args.seed)
    train_rng = np.random.default_rng(args.seed + 1)
    val_rng = np.random.default_rng(args.seed + 2)

    checkpoint = Path(args.checkpoint)
    model = load_model(
        checkpoint,
        model_config=args.model_config,
        transform_config=args.transform_config,
    )
    compatibility = validate_checkpoint(model, checkpoint)
    if model.use_edm:
        raise RuntimeError("surface-flow probe requires use_edm=false")
    print(
        "Pure VM checkpoint compatibility: exact match "
        f"({compatibility['model_parameters']} parameters)",
        flush=True,
    )
    set_train_scope(model, args.train_scope)
    optimizer = jt.optim.Adam(
        collect_train_parameters(model, args.train_scope),
        lr=args.lr,
    )

    reference = read_datalist(args.category_reference_list)
    train_paths = choose_paths(
        usable_paths(
            read_datalist(args.train_list),
            args.clean_root,
            args.mesh_root,
            sample_missing_clean=args.sample_missing_clean,
        ),
        args.train_shapes,
        selection_rng,
        category_reference=reference,
    )
    val_paths = choose_paths(
        usable_paths(
            read_datalist(args.val_list),
            args.clean_root,
            args.mesh_root,
            sample_missing_clean=args.sample_missing_clean,
        ),
        args.val_shapes,
        selection_rng,
        category_reference=reference,
    )
    train_shapes = [
        load_shape(
            path,
            args.clean_root,
            args.mesh_root,
            sample_missing_clean=args.sample_missing_clean,
        )
        for path in train_paths
    ]
    val_shapes = [
        load_shape(
            path,
            args.clean_root,
            args.mesh_root,
            sample_missing_clean=args.sample_missing_clean,
        )
        for path in val_paths
    ]
    val_patches = build_eval_patches(
        val_shapes,
        patches_per_shape=args.val_patches_per_shape,
        patch_size=args.patch_size,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        rng=val_rng,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "setup.json",
        {
            "args": vars(args),
            "checkpoint_compatibility": compatibility,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "noise_type": "laplace",
            "use_edm": False,
        },
    )

    print("Before training:", flush=True)
    before_rows, before_summary = evaluate(
        model,
        val_patches,
        steps=args.steps,
        step_scale=args.step_scale,
    )
    before_rows, before_summary = add_original_baseline_comparison(
        before_rows,
        before_summary,
        before_rows,
    )
    write_csv(out_dir / "before_eval.csv", before_rows)
    write_json(out_dir / "before_summary.json", before_summary)
    write_csv(out_dir / "best_eval.csv", before_rows)
    write_json(out_dir / "best_summary.json", before_summary)
    model.save(str(out_dir / "checkpoint_best.pkl"))

    best_gain = before_summary["gain_vs_original_vm"]
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses_epoch = []
        skipped_steps = 0
        order = train_rng.permutation(len(train_shapes))
        total_steps = len(train_shapes) * args.train_patches_per_shape
        running_loss = 0.0
        with tqdm(
            total=total_steps,
            desc=f"Epoch {epoch}",
            unit="patch",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as pbar:
            for shape_index in order:
                shape = train_shapes[int(shape_index)]
                for _ in range(args.train_patches_per_shape):
                    sigma = float(
                        train_rng.uniform(args.noise_min, args.noise_max)
                    )
                    patch = sample_patch(
                        shape,
                        sigma=sigma,
                        patch_size=args.patch_size,
                        rng=train_rng,
                    )
                    noisy = jt.array(patch["noisy"][None, :, :])
                    _, _, losses = recurrent_forward(
                        model,
                        noisy,
                        patch["vertices"],
                        patch["faces"],
                        steps=args.steps,
                        step_scale=args.step_scale,
                        chamfer_points=args.chamfer_points,
                        rng=train_rng,
                        with_loss=True,
                    )
                    sigma2 = max(sigma**2.0, 1e-8)
                    loss = (
                        args.flow_weight * losses["flow"]
                        + args.surface_weight * losses["surface"]
                        + args.monotonic_weight * losses["monotonic"]
                        + args.final_surface_weight
                        * losses["final_surface"]
                    ) / sigma2
                    loss_value = float(loss.item())
                    grad_norm = float("nan")
                    if math.isfinite(loss_value):
                        optimizer.zero_grad()
                        optimizer.backward(loss)
                        grad_norm = optimizer_grad_norm(optimizer)
                    if math.isfinite(loss_value) and math.isfinite(grad_norm):
                        optimizer.clip_grad_norm(args.max_grad_norm)
                        optimizer.step()
                    else:
                        skipped_steps += 1
                        optimizer.zero_grad()
                    loss_values = {
                        "loss": loss_value,
                        **{
                            key: float(value.item() / sigma2)
                            for key, value in losses.items()
                        },
                    }
                    if math.isfinite(loss_value):
                        losses_epoch.append(loss_values)
                        running_loss += loss_value
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=(
                            f"{running_loss / len(losses_epoch):.4f}"
                            if losses_epoch
                            else "nan"
                        ),
                        grad=(
                            f"{grad_norm:.3f}"
                            if math.isfinite(grad_norm)
                            else "nan"
                        ),
                        skipped=skipped_steps,
                        sigma=f"{sigma:.4f}",
                    )
                    jt.gc()

        if not losses_epoch:
            raise RuntimeError(
                f"epoch {epoch} had no finite optimization steps"
            )
        record = {
            "epoch": epoch,
            "train_skipped_steps": skipped_steps,
            **{
                f"train_{key}": float(
                    np.mean([item[key] for item in losses_epoch])
                )
                for key in losses_epoch[0]
            },
        }
        if (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        ):
            print(f"Epoch {epoch} validation:", flush=True)
            rows, summary = evaluate(
                model,
                val_patches,
                steps=args.steps,
                step_scale=args.step_scale,
            )
            rows, summary = add_original_baseline_comparison(
                rows,
                summary,
                before_rows,
            )
            record.update(
                {
                    f"val_{key}": value
                    for key, value in summary.items()
                    if key != "count"
                }
            )
            if summary["gain_vs_original_vm"] > best_gain:
                best_gain = summary["gain_vs_original_vm"]
                best_epoch = epoch
                model.save(str(out_dir / "checkpoint_best.pkl"))
                write_csv(out_dir / "best_eval.csv", rows)
                write_json(out_dir / "best_summary.json", summary)
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        print(record, flush=True)

    model.save(str(out_dir / "checkpoint_last.pkl"))
    write_json(
        out_dir / "train_summary.json",
        {
            "before": before_summary,
            "best_epoch": best_epoch,
            "best_gain": best_gain,
            "gain_over_untrained_recurrent": (
                best_gain - before_summary["final_gain"]
            ),
            "go_no_go": (
                "expand"
                if (
                    best_gain >= 4.0
                    and best_epoch >= 0
                )
                else "stop_or_revise"
            ),
            "args": vars(args),
        },
    )


if __name__ == "__main__":
    main()
