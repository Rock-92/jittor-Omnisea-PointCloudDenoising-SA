import argparse
import csv
import json
import sys
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.noise_classifier import PatchNoiseClassifier  # noqa: E402
from tools.train_multistage_refinement_probe import (  # noqa: E402
    cache_coarse,
    load_patch_file,
)


BAND_NAMES = ["low", "medium", "high"]


def sigma_to_label(sigma):
    sigma = np.asarray(sigma).reshape(-1)
    return np.where(sigma < 0.010, 0, np.where(sigma < 0.015, 1, 2)).astype(
        np.int32
    )


def sample_points(noisy, coarse, point_indices):
    return noisy[:, point_indices, :], coarse[:, point_indices, :]


def confusion_matrix(labels, predictions):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def classification_summary(labels, predictions, true_sigma, predicted_sigma):
    matrix = confusion_matrix(labels, predictions)
    recalls = []
    per_band = {}
    for index, name in enumerate(BAND_NAMES):
        count = int(matrix[index].sum())
        recall = float(matrix[index, index] / count) if count else 0.0
        recalls.append(recall)
        mask = labels == index
        per_band[name] = {
            "count": count,
            "recall": recall,
            "precision": float(
                matrix[index, index] / max(int(matrix[:, index].sum()), 1)
            ),
            "sigma_mae": float(np.mean(
                np.abs(predicted_sigma[mask] - true_sigma[mask])
            )) if mask.any() else 0.0,
        }
    correlation = 0.0
    if np.std(true_sigma) > 1e-12 and np.std(predicted_sigma) > 1e-12:
        correlation = float(np.corrcoef(true_sigma, predicted_sigma)[0, 1])
    return {
        "count": int(labels.size),
        "accuracy": float(np.mean(labels == predictions)),
        "macro_recall": float(np.mean(recalls)),
        "min_recall": float(np.min(recalls)),
        "sigma_mae": float(np.mean(np.abs(predicted_sigma - true_sigma))),
        "sigma_rmse": float(np.sqrt(np.mean(
            (predicted_sigma - true_sigma) ** 2.0
        ))),
        "pearson": correlation,
        "confusion_matrix": {
            BAND_NAMES[row]: {
                BAND_NAMES[column]: int(matrix[row, column])
                for column in range(3)
            }
            for row in range(3)
        },
        "per_band": per_band,
    }


def predict(
    model,
    noisy,
    coarse,
    point_indices,
    batch_size,
    sigma_min,
    sigma_max,
):
    model.eval()
    logits_all = []
    sigma_all = []
    with jt.no_grad():
        for start in range(0, noisy.shape[0], batch_size):
            end = min(start + batch_size, noisy.shape[0])
            noisy_batch, coarse_batch = sample_points(
                noisy[start:end],
                coarse[start:end],
                point_indices,
            )
            logits, output = model(
                jt.array(noisy_batch),
                jt.array(coarse_batch),
            )
            logits_all.append(logits.numpy())
            sigma_all.append(
                float(sigma_min)
                + (float(sigma_max) - float(sigma_min))
                * output["sigma_normalized"].numpy().reshape(-1)
            )
    logits = np.concatenate(logits_all, axis=0)
    predicted_sigma = np.concatenate(sigma_all, axis=0)
    return logits.argmax(axis=1).astype(np.int32), predicted_sigma


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/vm_ssl/checkpoint_best.pkl",
    )
    parser.add_argument(
        "--dataset-dir",
        default="outputs/refinement_v2_dataset",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/patch_noise_classifier_v1",
    )
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-loss-weight", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_patch_file(dataset_dir / "train_patches.npz")
    val = load_patch_file(dataset_dir / "val_patches.npz")
    train_coarse = cache_coarse(
        args.checkpoint,
        train,
        dataset_dir / "train_coarse.npz",
        args.batch_size,
        args.coarse_mode,
    )
    val_coarse = cache_coarse(
        args.checkpoint,
        val,
        dataset_dir / "val_coarse.npz",
        args.batch_size,
        args.coarse_mode,
    )
    point_count = train["pc_noisy"].shape[1]
    sample_count = min(int(args.num_points), point_count)
    point_indices = np.linspace(
        0,
        point_count - 1,
        sample_count,
        dtype=np.int32,
    )
    train_labels = sigma_to_label(train["score_sigma"])
    val_labels = sigma_to_label(val["score_sigma"])

    model = PatchNoiseClassifier(
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
    )
    optimizer = jt.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_selection_score = -1.0
    best_epoch = -1
    sigma_range = max(float(args.sigma_max - args.sigma_min), 1e-6)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_labels.size)
        batch_rows = []
        for start in range(0, order.size, args.batch_size):
            indices = order[start:start + args.batch_size]
            noisy_batch, coarse_batch = sample_points(
                train["pc_noisy"][indices],
                train_coarse[indices],
                point_indices,
            )
            labels = jt.array(train_labels[indices]).int32()
            sigma_target = jt.array(
                (
                    train["score_sigma"][indices].reshape(-1)
                    - float(args.sigma_min)
                ) / sigma_range
            ).reshape((-1, 1))
            logits, output = model(
                jt.array(noisy_batch),
                jt.array(coarse_batch),
            )
            class_loss = nn.cross_entropy_loss(logits, labels)
            sigma_loss = (
                (output["sigma_normalized"] - sigma_target) ** 2.0
            ).mean()
            loss = (
                class_loss
                + float(args.sigma_loss_weight) * sigma_loss
            )
            optimizer.step(loss)
            batch_rows.append({
                "loss": float(loss.item()),
                "class_loss": float(class_loss.item()),
                "sigma_loss": float(sigma_loss.item()),
            })

        if (
            epoch != 0
            and (epoch + 1) % args.eval_every != 0
            and epoch != args.epochs - 1
        ):
            continue
        predictions, predicted_sigma = predict(
            model,
            val["pc_noisy"],
            val_coarse,
            point_indices,
            args.batch_size,
            args.sigma_min,
            args.sigma_max,
        )
        summary = classification_summary(
            val_labels,
            predictions,
            val["score_sigma"].reshape(-1),
            predicted_sigma,
        )
        selection_score = (
            summary["macro_recall"]
            + 0.25 * summary["accuracy"]
            + 0.25 * summary["min_recall"]
        )
        record = {
            "epoch": epoch,
            **{
                key: float(np.mean([row[key] for row in batch_rows]))
                for key in batch_rows[0]
            },
            "val_accuracy": summary["accuracy"],
            "val_macro_recall": summary["macro_recall"],
            "val_min_recall": summary["min_recall"],
            "val_sigma_mae": summary["sigma_mae"],
            "val_pearson": summary["pearson"],
            "selection_score": selection_score,
        }
        history.append(record)
        write_csv(out_dir / "epoch_log.csv", history)
        print(record, flush=True)

        if selection_score >= best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch
            model.save(str(out_dir / "classifier_best.pkl"))
            (out_dir / "best_val_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

    model.save(str(out_dir / "classifier_last.pkl"))
    final = {
        "best_epoch": best_epoch,
        "best_selection_score": best_selection_score,
        "train_patches": int(train_labels.size),
        "val_patches": int(val_labels.size),
        "train_class_counts": {
            BAND_NAMES[index]: int(np.sum(train_labels == index))
            for index in range(3)
        },
        "val_class_counts": {
            BAND_NAMES[index]: int(np.sum(val_labels == index))
            for index in range(3)
        },
        "args": vars(args),
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
