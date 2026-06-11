import argparse
import csv
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.refinement import GeometryResidualRefiner  # noqa: E402
from tools.hard_patch_common import (  # noqa: E402
    displacement_metrics,
    load_hard_patch_npz,
    load_model,
    quantile_summary,
    score_prediction,
    write_json,
)


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_batches(indices, batch_size, rng, shuffle):
    indices = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def generate_coarse(model, noisy_np, batch_size, mode, sigma_np):
    outputs = []
    model.eval()
    with jt.no_grad():
        for start in range(0, noisy_np.shape[0], batch_size):
            end = min(start + batch_size, noisy_np.shape[0])
            noisy = jt.array(noisy_np[start:end])
            if mode == "heun":
                coarse, _ = model.denoise_langevin_dynamics(noisy)
            elif mode == "fixed":
                sigma = jt.array(sigma_np[start:end])
                coarse = model.predict_clean(noisy, sigma=sigma)
            else:
                raise ValueError(f"unsupported coarse mode: {mode}")
            outputs.append(coarse.numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def evaluate_predictions(noisy_np, clean_np, pred_np, indices):
    rows = []
    for index in indices:
        row = {
            "index": int(index),
            **score_prediction(noisy_np[index], clean_np[index], pred_np[index]),
            **displacement_metrics(noisy_np[index], clean_np[index], pred_np[index]),
        }
        rows.append(row)
    return rows, {
        "score": quantile_summary([row["cd_score"] for row in rows]),
        "length_ratio": quantile_summary(
            [row["length_ratio_mean"] for row in rows]
        ),
        "cosine": quantile_summary([row["cosine_mean"] for row in rows]),
        "under_length_rate": quantile_summary(
            [row["under_length_rate"] for row in rows]
        ),
    }


def predict_refined(refiner, coarse_np, noisy_np, batch_size):
    outputs = []
    refiner.eval()
    with jt.no_grad():
        for start in range(0, coarse_np.shape[0], batch_size):
            end = min(start + batch_size, coarse_np.shape[0])
            pred, _ = refiner(
                jt.array(coarse_np[start:end]),
                jt.array(noisy_np[start:end]),
            )
            outputs.append(pred.numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def refinement_loss(
    pred,
    clean,
    aux,
    residual_weight,
    gate_weight,
    loss_scale,
):
    scale2 = max(float(loss_scale) ** 2.0, 1e-8)
    paired = ((pred - clean) ** 2.0).sum(dim=-1).mean() / scale2
    residual_reg = (
        (aux["residual"] ** 2.0).sum(dim=-1).mean() / scale2
    )
    gate_reg = aux["gate"].mean()
    total = (
        paired
        + float(residual_weight) * residual_reg
        + float(gate_weight) * gate_reg
    )
    return total, paired, residual_reg, gate_reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs_result/outputs_hardware/"
            "checkpoints/vm_ssl/checkpoint_best.pkl"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "outputs_result/outputs_analysis/"
            "hardware_patch_diagnosis/hard_patches.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs_result/outputs_analysis/refinement_probe",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-residual", type=float, default=0.006)
    parser.add_argument("--tangent-scale", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--gate-weight", type=float, default=0.0001)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_hard_patch_npz(args.dataset)
    noisy_np = data["pc_noisy"]
    clean_np = data["pc_clean"]
    sigma_np = data["score_sigma"]

    all_indices = np.arange(noisy_np.shape[0])
    split_rng = np.random.default_rng(args.seed + 1)
    split_rng.shuffle(all_indices)
    holdout_count = max(
        1,
        int(round(noisy_np.shape[0] * args.holdout_fraction)),
    )
    holdout_indices = np.sort(all_indices[:holdout_count])
    train_indices = np.sort(all_indices[holdout_count:])

    coarse_model = load_model(args.checkpoint)
    for parameter in coarse_model.parameters():
        parameter.stop_grad()
    coarse_np = generate_coarse(
        coarse_model,
        noisy_np,
        args.batch_size,
        args.coarse_mode,
        sigma_np,
    )
    np.savez_compressed(
        out_dir / "cached_coarse.npz",
        pc_noisy=noisy_np,
        pc_clean=clean_np,
        pc_coarse=coarse_np,
        train_indices=train_indices,
        holdout_indices=holdout_indices,
    )

    coarse_train_rows, coarse_train_summary = evaluate_predictions(
        noisy_np,
        clean_np,
        coarse_np,
        train_indices,
    )
    coarse_holdout_rows, coarse_holdout_summary = evaluate_predictions(
        noisy_np,
        clean_np,
        coarse_np,
        holdout_indices,
    )
    write_csv(out_dir / "coarse_train_eval.csv", coarse_train_rows)
    write_csv(out_dir / "coarse_holdout_eval.csv", coarse_holdout_rows)

    refiner = GeometryResidualRefiner(
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        max_residual=args.max_residual,
        tangent_scale=args.tangent_scale,
    )
    optimizer = jt.optim.Adam(refiner.parameters(), lr=args.lr)
    history = []
    best_holdout_score = coarse_holdout_summary["score"]["mean"]
    best_epoch = -1

    for epoch in range(args.epochs):
        refiner.train()
        epoch_losses = []
        for batch_indices in iter_batches(
            train_indices,
            args.batch_size,
            rng,
            shuffle=True,
        ):
            coarse = jt.array(coarse_np[batch_indices])
            noisy = jt.array(noisy_np[batch_indices])
            clean = jt.array(clean_np[batch_indices])
            pred, aux = refiner(coarse, noisy)
            loss, paired, residual_reg, gate_reg = refinement_loss(
                pred,
                clean,
                aux,
                args.residual_weight,
                args.gate_weight,
                args.max_residual,
            )
            optimizer.step(loss)
            epoch_losses.append(
                [
                    float(loss.item()),
                    float(paired.item()),
                    float(residual_reg.item()),
                    float(gate_reg.item()),
                ]
            )

        should_eval = (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        )
        if not should_eval:
            continue

        refined_np = predict_refined(
            refiner,
            coarse_np,
            noisy_np,
            args.batch_size,
        )
        train_rows, train_summary = evaluate_predictions(
            noisy_np,
            clean_np,
            refined_np,
            train_indices,
        )
        holdout_rows, holdout_summary = evaluate_predictions(
            noisy_np,
            clean_np,
            refined_np,
            holdout_indices,
        )
        loss_mean = np.asarray(epoch_losses).mean(axis=0)
        record = {
            "epoch": epoch,
            "loss": float(loss_mean[0]),
            "paired_loss": float(loss_mean[1]),
            "residual_reg": float(loss_mean[2]),
            "gate_mean": float(loss_mean[3]),
            "train_score": train_summary["score"]["mean"],
            "train_cosine": train_summary["cosine"]["mean"],
            "train_length_ratio": train_summary["length_ratio"]["mean"],
            "holdout_score": holdout_summary["score"]["mean"],
            "holdout_cosine": holdout_summary["cosine"]["mean"],
            "holdout_length_ratio": holdout_summary["length_ratio"]["mean"],
        }
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        print(record, flush=True)

        if holdout_summary["score"]["mean"] >= best_holdout_score:
            best_holdout_score = holdout_summary["score"]["mean"]
            best_epoch = epoch
            refiner.save(str(out_dir / "refiner_best.pkl"))
            write_csv(out_dir / "best_train_eval.csv", train_rows)
            write_csv(out_dir / "best_holdout_eval.csv", holdout_rows)
            write_json(out_dir / "best_train_summary.json", train_summary)
            write_json(out_dir / "best_holdout_summary.json", holdout_summary)

    refiner.save(str(out_dir / "refiner_last.pkl"))
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "train_count": int(len(train_indices)),
        "holdout_count": int(len(holdout_indices)),
        "best_epoch": best_epoch,
        "best_holdout_score": best_holdout_score,
        "coarse_train_summary": coarse_train_summary,
        "coarse_holdout_summary": coarse_holdout_summary,
        "args": vars(args),
        "interpretation": {
            "promising": (
                best_holdout_score
                >= coarse_holdout_summary["score"]["mean"] + 2.0
            ),
            "target_cosine": 0.4,
        },
    }
    write_json(out_dir / "probe_summary.json", summary)
    (out_dir / "split.json").write_text(
        json.dumps(
            {
                "train_indices": train_indices.tolist(),
                "holdout_indices": holdout_indices.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
