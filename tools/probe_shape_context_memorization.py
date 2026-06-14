import argparse
import json
import sys
import time
from pathlib import Path

import jittor as jt
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.shape_context import (  # noqa: E402
    CleanShapeRegionProcessor,
    ShapeContextVMAdapter,
)
from tools.hard_patch_common import load_model, read_datalist  # noqa: E402
from tools.probe_clean_shape_context_vm import (  # noqa: E402
    adapter_loss,
    add_fixed_instance,
    cache_vm_predictions,
    evaluate,
    optimizer_grad_norm,
    refine_patch_batch,
    shape_tokens,
    write_csv,
    write_json,
)
from tools.train_full_cloud_fusion_probe import (  # noqa: E402
    choose_balanced_paths,
    load_shape,
    parse_noise_bands,
    usable_paths,
    validate_checkpoint_compatibility,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/vm/checkpoint_best.pkl"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs/model/vm_pure_global.yaml"
DEFAULT_TRANSFORM_CONFIG = PROJECT_ROOT / "configs/transform/vm_pure_laplace.yaml"


def sample_other_index(rng, count, target_index):
    offset = int(rng.integers(1, count))
    return (int(target_index) + offset) % count


def clean_patch(instance, patch_indices):
    return (
        instance["clean"][instance["point_indices"][patch_indices]]
        - instance["seeds"][patch_indices, None, :]
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--transform-config", default=str(DEFAULT_TRANSFORM_CONFIG))
    parser.add_argument("--train-list", default="datalist/train.txt")
    parser.add_argument("--category-reference-list", default="datalist/test.txt")
    parser.add_argument("--clean-root", default="cache_clean_points")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument(
        "--out-dir",
        default="outputs/shape_context_memorization_probe",
    )
    parser.add_argument("--shapes", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--patches-per-step", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--fusion-tau", type=float, default=2.0)
    parser.add_argument("--vm-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--region-count", type=int, default=256)
    parser.add_argument("--points-per-region", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--context-knn", type=int, default=4)
    parser.add_argument("--max-residual", type=float, default=0.008)
    parser.add_argument(
        "--noise-bands",
        default="0.005:0.010,0.010:0.015,0.015:0.020",
    )
    parser.add_argument("--rank-margin", type=float, default=0.35)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--separation-margin", type=float, default=0.15)
    parser.add_argument("--separation-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--sample-missing-clean", action="store_true")
    parser.add_argument("--allow-nonstandard-protocol", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    if args.shapes < 2:
        raise ValueError("memorization probe requires at least two shapes")
    if (
        args.num_points != 32768
        or args.patch_size != 1000
        or args.seed_k != 6.0
    ) and not args.allow_nonstandard_protocol:
        raise ValueError(
            "probe must use 32768 points / 1000-point patches / seed_k=6; "
            "use --allow-nonstandard-protocol only for smoke tests"
        )

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    selection_rng = np.random.default_rng(args.seed)
    train_rng = np.random.default_rng(args.seed + 1)
    bands = parse_noise_bands(args.noise_bands)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vm = load_model(
        args.checkpoint,
        model_config=args.model_config,
        transform_config=args.transform_config,
    )
    compatibility = validate_checkpoint_compatibility(vm, args.checkpoint)
    if vm.use_edm:
        raise RuntimeError("probe requires pure VM with use_edm=false")
    vm.eval()
    for parameter in vm.parameters():
        parameter.stop_grad()

    reference = read_datalist(args.category_reference_list)
    candidates = usable_paths(
        read_datalist(args.train_list),
        args.clean_root,
        args.mesh_root,
        args.sample_missing_clean,
    )
    paths = choose_balanced_paths(
        candidates,
        args.shapes,
        reference,
        selection_rng,
    )
    print("Building fixed Laplace instances and caching pure VM...", flush=True)
    instances = []
    for index, path in enumerate(tqdm(paths, unit="shape")):
        shape = load_shape(
            path,
            args.clean_root,
            args.mesh_root,
            args.num_points,
            train_rng,
            args.sample_missing_clean,
        )
        lower, upper = bands[index % len(bands)]
        instance = add_fixed_instance(
            shape,
            sigma=float(train_rng.uniform(lower, upper)),
            args=args,
            rng=train_rng,
        )
        instance["noise_band"] = (lower, upper)
        cache_vm_predictions(vm, instance, args)
        instances.append(instance)

    processor = CleanShapeRegionProcessor(token_dim=args.token_dim)
    adapter = ShapeContextVMAdapter(
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        context_knn=args.context_knn,
        max_residual=args.max_residual,
        context_only_head=True,
    )
    optimizer = jt.optim.Adam(
        list(processor.parameters()) + list(adapter.parameters()),
        lr=args.lr,
    )
    write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "paths": paths,
            "checkpoint_compatibility": compatibility,
            "protocol": {
                "vm_use_edm": False,
                "noise_type": "laplace",
                "fixed_train_equals_eval": True,
                "context_only_head": True,
                "paired_correct_shuffled_training": True,
            },
        },
    )

    history = []
    best_objective = -1e9
    best_summary = None
    best_epoch = -1
    best_train_cosine = 0.0
    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        optimizer.lr = float(
            args.min_lr
            + 0.5 * (args.lr - args.min_lr)
            * (1.0 + np.cos(np.pi * progress))
        )
        processor.train()
        adapter.train()
        metrics = []
        start_time = time.time()
        bar = tqdm(
            range(args.steps_per_epoch),
            desc=f"Epoch {epoch}",
            unit="step",
        )
        for _ in bar:
            target_index = int(train_rng.integers(len(instances)))
            wrong_index = sample_other_index(
                train_rng,
                len(instances),
                target_index,
            )
            instance = instances[target_index]
            wrong_instance = instances[wrong_index]
            patch_count = instance["patches"].shape[0]
            patch_indices = train_rng.choice(
                patch_count,
                size=min(args.patches_per_step, patch_count),
                replace=False,
            ).astype(np.int32)

            correct_tokens, correct_global = shape_tokens(processor, instance)
            wrong_tokens, wrong_global = shape_tokens(processor, wrong_instance)
            correct_prediction, correct_aux = refine_patch_batch(
                adapter,
                instance,
                patch_indices,
                correct_tokens,
                correct_global,
            )
            wrong_prediction, wrong_aux = refine_patch_batch(
                adapter,
                instance,
                patch_indices,
                wrong_tokens,
                wrong_global,
            )
            target_patch = clean_patch(instance, patch_indices)
            coarse = instance["vm_patch"][patch_indices]
            correct_loss, correct_parts = adapter_loss(
                correct_prediction,
                coarse,
                target_patch,
                correct_aux,
                args.max_residual,
            )
            _, wrong_parts = adapter_loss(
                wrong_prediction,
                coarse,
                target_patch,
                wrong_aux,
                args.max_residual,
            )
            rank = jt.maximum(
                correct_parts["vector"]
                - wrong_parts["vector"]
                + args.rank_margin,
                jt.zeros((1,)),
            ).mean()
            residual_difference = jt.sqrt(
                (
                    (correct_aux["residual"] - wrong_aux["residual"]) ** 2.0
                ).sum(dim=-1)
                + 1e-8
            ).mean() / max(args.max_residual, 1e-6)
            separation = jt.maximum(
                args.separation_margin - residual_difference,
                jt.zeros((1,)),
            ).mean()
            loss = (
                correct_loss
                + args.rank_weight * rank
                + args.separation_weight * separation
            )

            optimizer.zero_grad()
            optimizer.backward(loss)
            grad_norm = optimizer_grad_norm(optimizer)
            if not np.isfinite(grad_norm):
                optimizer.zero_grad()
                continue
            optimizer.clip_grad_norm(args.grad_clip)
            optimizer.step()
            row = {
                "loss": float(loss.item()),
                "correct_loss": float(correct_loss.item()),
                "correct_vector": float(correct_parts["vector"].item()),
                "wrong_vector": float(wrong_parts["vector"].item()),
                "rank": float(rank.item()),
                "separation": float(separation.item()),
                "residual_difference": float(residual_difference.item()),
                "cosine": float(correct_parts["cosine"].item()),
                "residual_mean": float(
                    correct_parts["residual_mean"].item()
                ),
                "target_mean": float(correct_parts["target_mean"].item()),
                "grad_norm": grad_norm,
            }
            metrics.append(row)
            bar.set_postfix(
                loss=f"{row['loss']:.3f}",
                cosine=f"{row['cosine']:.3f}",
                rank=f"{row['rank']:.3f}",
                diff=f"{row['residual_difference']:.3f}",
            )
            jt.gc()

        record = {
            "epoch": epoch,
            "lr": float(optimizer.lr),
            "seconds": float(time.time() - start_time),
            **{
                f"train_{key}": float(
                    np.mean([row[key] for row in metrics])
                )
                for key in metrics[0]
            },
        }
        if (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        ):
            print(f"Epoch {epoch} memorization evaluation:", flush=True)
            rows, summary = evaluate(
                processor,
                adapter,
                instances,
                args,
            )
            record.update(
                {
                    "eval_vm": summary["vm"]["final_score"],
                    "eval_correct": summary["correct"]["final_score"],
                    "eval_shuffled": summary["shuffled"]["final_score"],
                    "eval_correct_gain": summary["correct_gain"],
                    "eval_context_specific_gain": summary[
                        "context_specific_gain"
                    ],
                }
            )
            objective = (
                summary["correct_gain"]
                + summary["context_specific_gain"]
            )
            if objective > best_objective:
                best_objective = objective
                best_summary = summary
                best_epoch = epoch
                best_train_cosine = record["train_cosine"]
                processor.save(str(out_dir / "processor_best.pkl"))
                adapter.save(str(out_dir / "adapter_best.pkl"))
                write_csv(out_dir / "best_rows.csv", rows)
                write_json(out_dir / "best_summary.json", summary)
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        processor.save(str(out_dir / "processor_last.pkl"))
        adapter.save(str(out_dir / "adapter_last.pkl"))
        print(record, flush=True)

    decision = {
        "shape_context_can_be_memorized": (
            best_summary["correct_gain"] >= 5.0
            and best_summary["context_specific_gain"] >= 3.0
            and best_train_cosine >= 0.3
        ),
        "criteria": {
            "correct_gain_ge_5": best_summary["correct_gain"] >= 5.0,
            "correct_minus_shuffled_ge_3": (
                best_summary["context_specific_gain"] >= 3.0
            ),
            "train_cosine_ge_0_3": best_train_cosine >= 0.3,
        },
        "best_epoch": best_epoch,
        "best_train_cosine": best_train_cosine,
        "best": best_summary,
        "args": vars(args),
    }
    write_json(out_dir / "memorization_summary.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
