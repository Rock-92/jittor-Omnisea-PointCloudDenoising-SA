import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.refinement import FusionAwareResidualRefiner  # noqa: E402
from tools.hard_patch_common import load_model, read_datalist  # noqa: E402
from tools.probe_fusion_aware_residual_oracle import (  # noqa: E402
    fuse_absolute,
    make_instance,
    predict_stage1,
)
from tools.train_full_cloud_fusion_probe import (  # noqa: E402
    choose_balanced_paths,
    load_shape,
    parse_noise_bands,
    score_instance,
    usable_paths,
    validate_checkpoint_compatibility,
)
from tools.train_fusion_aware_stage2 import (  # noqa: E402
    finite,
    optimizer_grad_norm,
    refine_full_cloud,
    stage2_loss,
    summarize,
    write_csv,
    write_json,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/vm/checkpoint_best.pkl"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/vm_pure_global.yaml"
DEFAULT_TRANSFORM_CONFIG = PROJECT_ROOT / "configs/transform/vm_pure_laplace.yaml"


def nearest_patch_groups(instance, group_size):
    patch_count = instance["seeds"].shape[0]
    neighbor_count = min(int(group_size), patch_count)
    _, groups = cKDTree(instance["seeds"]).query(
        instance["seeds"],
        k=neighbor_count,
    )
    groups = np.asarray(groups, dtype=np.int32)
    if groups.ndim == 1:
        groups = groups[:, None]
    return groups


def run_refiner_with_full_consensus(
    refiner,
    instance,
    patch_indices,
):
    patch_indices = np.asarray(patch_indices, dtype=np.int32)
    point_indices = instance["point_indices"][patch_indices]
    seeds = instance["seeds"][patch_indices]
    consensus = (
        instance["baseline_prediction"][point_indices]
        - seeds[:, None, :]
    ).astype(np.float32)
    return refiner(
        jt.array(instance["stage1_patch"][patch_indices]),
        jt.array(instance["patches"][patch_indices]),
        jt.array(consensus),
        jt.array(
            instance["normalized_distances"][patch_indices, :, None]
        ),
    )


def evaluate_cached(refiner, instances, args):
    rows = []
    for index, instance in enumerate(instances):
        stage1_patch = instance["stage1_patch"]
        baseline = instance["baseline_prediction"]
        refined = refine_full_cloud(
            refiner,
            instance,
            stage1_patch,
            batch_size=args.stage2_batch_size,
            fusion_tau=args.fusion_tau,
        )
        base_score = score_instance(instance, baseline)
        refined_score = score_instance(instance, refined)
        row = {
            "rel_path": instance["rel_path"],
            "sigma": instance["sigma"],
            **{
                f"base_{key}": value
                for key, value in base_score.items()
                if key.endswith("_score")
            },
            **{
                f"refined_{key}": value
                for key, value in refined_score.items()
                if key.endswith("_score")
            },
            "final_gain": (
                refined_score["final_score"] - base_score["final_score"]
            ),
        }
        rows.append(row)
        print(
            f"  eval [{index + 1}/{len(instances)}] "
            f"base={base_score['final_score']:.3f} "
            f"refined={refined_score['final_score']:.3f} "
            f"gain={row['final_gain']:+.3f}",
            flush=True,
        )
        jt.gc()
    return rows, summarize(rows)


def build_fixed_instances(stage1, paths, bands, args, rng):
    instances = []
    progress = tqdm(paths, desc="Caching frozen VM predictions", unit="shape")
    for index, path in enumerate(progress):
        shape = load_shape(
            path,
            clean_root=args.clean_root,
            mesh_root=args.mesh_root,
            num_points=args.num_points,
            rng=rng,
            sample_missing_clean=args.sample_missing_clean,
        )
        lower, upper = bands[index % len(bands)]
        sigma = float(rng.uniform(lower, upper))
        instance = make_instance(
            shape,
            sigma=sigma,
            patch_size=args.patch_size,
            seed_k=args.seed_k,
            noise_type=args.noise_type,
            rng=rng,
        )
        instance["noise_band"] = (lower, upper)
        stage1_patch = predict_stage1(
            stage1,
            instance,
            batch_size=args.stage1_batch_size,
            mode=args.stage1_mode,
        )
        instance["stage1_patch"] = stage1_patch
        instance["baseline_prediction"] = fuse_absolute(
            instance,
            stage1_patch + instance["seeds"][:, None, :],
            fusion_tau=args.fusion_tau,
        )
        instance["patch_groups"] = nearest_patch_groups(
            instance,
            args.group_size,
        )
        instances.append(instance)
    return instances


def build_anchor_schedule(instances):
    schedule = []
    for shape_index, instance in enumerate(instances):
        for anchor_index in range(instance["patches"].shape[0]):
            schedule.append((shape_index, anchor_index))
    return np.asarray(schedule, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--transform-config", default=str(DEFAULT_TRANSFORM_CONFIG))
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--category-reference-list", default="datalist/test.txt")
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument(
        "--out-dir",
        default="outputs/fusion_aware_stage2_overfit_probe",
    )
    parser.add_argument("--shapes", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--stage1-batch-size", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument(
        "--stage1-mode",
        choices=["one_step", "heun"],
        default="one_step",
    )
    parser.add_argument(
        "--noise-bands",
        default="0.005:0.010,0.010:0.015,0.015:0.020",
    )
    parser.add_argument(
        "--noise-type",
        choices=["laplace", "gaussian"],
        default="laplace",
    )
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--max-residual", type=float, default=0.008)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--load-refiner", default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--teacher-smooth-k", type=int, default=24)
    parser.add_argument("--teacher-smooth-alpha", type=float, default=0.75)
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--fused-weight", type=float, default=1.0)
    parser.add_argument("--patch-weight", type=float, default=0.05)
    parser.add_argument("--direction-weight", type=float, default=0.5)
    parser.add_argument("--length-weight", type=float, default=0.5)
    parser.add_argument("--keep-weight", type=float, default=0.2)
    parser.add_argument("--keep-threshold", type=float, default=0.002)
    parser.add_argument("--gate-weight", type=float, default=0.2)
    parser.add_argument("--overlap-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--allow-nonstandard-protocol", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    if (
        args.num_points != 32768
        or args.patch_size != 1000
        or args.seed_k != 6.0
    ) and not args.allow_nonstandard_protocol:
        raise ValueError(
            "probe must use 32768 points / 1000-point patches / seed_k=6; "
            "use --allow-nonstandard-protocol only for smoke tests"
        )
    if args.shapes < 1 or args.steps_per_epoch < 1 or args.epochs < 1:
        raise ValueError("shapes, steps_per_epoch, and epochs must be positive")
    if not 0.0 <= args.teacher_smooth_alpha <= 1.0:
        raise ValueError("teacher_smooth_alpha must be in [0, 1]")

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    selection_rng = np.random.default_rng(args.seed)
    train_rng = np.random.default_rng(args.seed + 1)
    bands = parse_noise_bands(args.noise_bands)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    stage1 = load_model(
        checkpoint,
        model_config=args.model_config,
        transform_config=args.transform_config,
    )
    compatibility = validate_checkpoint_compatibility(stage1, checkpoint)
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.stop_grad()

    candidates = usable_paths(
        read_datalist(args.datalist),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    paths = choose_balanced_paths(
        candidates,
        args.shapes,
        read_datalist(args.category_reference_list),
        selection_rng,
    )
    if not paths:
        raise FileNotFoundError(
            "no usable shapes; generate cache_clean_points or pass "
            "--sample-missing-clean"
        )

    instances = build_fixed_instances(
        stage1,
        paths,
        bands,
        args,
        selection_rng,
    )
    anchor_schedule = build_anchor_schedule(instances)
    train_rng.shuffle(anchor_schedule)
    schedule_cursor = 0
    print(
        f"Fixed overfit set: {len(instances)} shapes, "
        f"{len(anchor_schedule)} patch anchors",
        flush=True,
    )

    refiner = FusionAwareResidualRefiner(
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        max_residual=args.max_residual,
    )
    if args.load_refiner:
        refiner.load(str(args.load_refiner))
        print(f"Loaded Stage2 checkpoint: {args.load_refiner}", flush=True)
    parameters = list(refiner.parameters())
    optimizer = jt.optim.Adam(parameters, lr=args.lr)
    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "paths": paths,
            "patch_counts": [
                int(instance["patches"].shape[0])
                for instance in instances
            ],
            "checkpoint_compatibility": compatibility,
        },
    )

    print("Before-training complete-cloud evaluation:", flush=True)
    before_rows, before_summary = evaluate_cached(refiner, instances, args)
    write_csv(out_dir / "before_rows.csv", before_rows)
    write_json(out_dir / "before_summary.json", before_summary)
    write_csv(out_dir / "best_rows.csv", before_rows)
    write_json(out_dir / "best_summary.json", before_summary)
    refiner.save(str(out_dir / "refiner_best.pkl"))
    best_score = before_summary["refined"]["final_score"]
    best_summary = before_summary
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1.0 + np.cos(np.pi * progress)
        )
        optimizer.lr = float(lr)
        refiner.train()
        epoch_metrics = []
        skipped = 0
        epoch_start = time.time()
        bar = tqdm(
            range(args.steps_per_epoch),
            desc=f"Epoch {epoch}",
            unit="group",
        )
        for _ in bar:
            if schedule_cursor == len(anchor_schedule):
                train_rng.shuffle(anchor_schedule)
                schedule_cursor = 0
            shape_index, anchor_index = anchor_schedule[schedule_cursor]
            schedule_cursor += 1
            instance = instances[int(shape_index)]
            patch_indices = instance["patch_groups"][int(anchor_index)]
            prediction, aux = run_refiner_with_full_consensus(
                refiner,
                instance,
                patch_indices,
            )
            loss, parts = stage2_loss(
                instance,
                patch_indices,
                int(anchor_index),
                prediction,
                aux,
                fusion_tau=args.fusion_tau,
                max_residual=args.max_residual,
                args=args,
            )
            if not finite(loss):
                skipped += 1
                optimizer.zero_grad()
                jt.gc()
                continue
            optimizer.zero_grad()
            optimizer.backward(loss)
            grad_norm = optimizer_grad_norm(optimizer)
            if not np.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad()
                jt.gc()
                continue
            optimizer.clip_grad_norm(args.grad_clip)
            optimizer.step()
            row = {
                "loss": float(loss.item()),
                "grad_norm": grad_norm,
                **{key: float(value.item()) for key, value in parts.items()},
            }
            epoch_metrics.append(row)
            ratio = row["residual_mean"] / max(row["teacher_mean"], 1e-8)
            bar.set_postfix(
                loss=f"{row['loss']:.3f}",
                cosine=f"{row['cosine_mean']:.3f}",
                ratio=f"{ratio:.2f}",
            )
            jt.gc()

        if not epoch_metrics:
            raise RuntimeError("all training steps were skipped")
        record = {
            "epoch": epoch,
            "lr": float(lr),
            "seconds": float(time.time() - epoch_start),
            "steps": len(epoch_metrics),
            "skipped": skipped,
            **{
                f"train_{key}": float(
                    np.mean([row[key] for row in epoch_metrics])
                )
                for key in epoch_metrics[0]
            },
        }
        record["train_residual_teacher_ratio"] = (
            record["train_residual_mean"]
            / max(record["train_teacher_mean"], 1e-8)
        )
        should_eval = (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        )
        if should_eval:
            print(f"Epoch {epoch} memorization evaluation:", flush=True)
            rows, summary = evaluate_cached(refiner, instances, args)
            score = summary["refined"]["final_score"]
            record.update(
                {
                    "eval_base": summary["base"]["final_score"],
                    "eval_refined": score,
                    "eval_gain": summary["gain"]["mean"],
                    "eval_min_gain": summary["gain"]["min"],
                    "eval_improved_rate": summary["gain"]["improved_rate"],
                }
            )
            if score > best_score:
                best_score = score
                best_summary = summary
                best_epoch = epoch
                refiner.save(str(out_dir / "refiner_best.pkl"))
                write_csv(out_dir / "best_rows.csv", rows)
                write_json(out_dir / "best_summary.json", summary)
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        refiner.save(str(out_dir / "refiner_last.pkl"))
        print(record, flush=True)

    final_cosine = history[-1]["train_cosine_mean"]
    final_ratio = history[-1]["train_residual_teacher_ratio"]
    best_gain = best_summary["gain"]["mean"]
    criteria = {
        "cosine_ge_0p7": final_cosine >= 0.7,
        "residual_teacher_ratio_ge_0p7": final_ratio >= 0.7,
        "complete_cloud_gain_ge_5": best_gain >= 5.0,
    }
    decision = (
        "architecture_can_memorize_teacher"
        if all(criteria.values())
        else "architecture_or_inputs_insufficient"
    )
    summary = {
        "decision": decision,
        "criteria": criteria,
        "best_epoch": best_epoch,
        "before": before_summary,
        "best": best_summary,
        "final_train_cosine": final_cosine,
        "final_residual_teacher_ratio": final_ratio,
        "args": vars(args),
    }
    write_json(out_dir / "probe_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
