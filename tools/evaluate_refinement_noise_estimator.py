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

from src.model.refinement import MultiStageGeometryRefiner  # noqa: E402
from tools.train_multistage_refinement_probe import (  # noqa: E402
    cache_coarse,
    load_patch_file,
)


BAND_NAMES = ["low", "medium", "high"]


def sigma_to_band(sigma):
    sigma = np.asarray(sigma)
    return np.where(sigma < 0.010, 0, np.where(sigma < 0.015, 1, 2))


def predict_noise_strength(model, coarse, noisy, batch_size):
    model.eval()
    stage_values = None
    with jt.no_grad():
        for start in range(0, coarse.shape[0], batch_size):
            end = min(start + batch_size, coarse.shape[0])
            _, output = model(
                jt.array(coarse[start:end]),
                jt.array(noisy[start:end]),
            )
            if stage_values is None:
                stage_values = [[] for _ in output["stages"]]
            for stage_index, stage in enumerate(output["stages"]):
                values = stage["noise_strength"].numpy().reshape(-1)
                stage_values[stage_index].append(values)
    return [
        np.concatenate(values).astype(np.float64, copy=False)
        for values in stage_values
    ]


def confusion_matrix(true_band, predicted_band):
    matrix = np.zeros((3, 3), dtype=np.int64)
    for true_value, predicted_value in zip(true_band, predicted_band):
        matrix[int(true_value), int(predicted_value)] += 1
    return matrix


def safe_correlation(first, second):
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def summarize(true_sigma, predicted_sigma):
    true_band = sigma_to_band(true_sigma)
    predicted_band = sigma_to_band(predicted_sigma)
    matrix = confusion_matrix(true_band, predicted_band)
    per_band = {}
    for band_index, band_name in enumerate(BAND_NAMES):
        mask = true_band == band_index
        per_band[band_name] = {
            "count": int(mask.sum()),
            "accuracy": float(
                np.mean(predicted_band[mask] == band_index)
            ) if mask.any() else 0.0,
            "true_sigma_mean": float(np.mean(true_sigma[mask]))
            if mask.any() else 0.0,
            "predicted_sigma_mean": float(np.mean(predicted_sigma[mask]))
            if mask.any() else 0.0,
            "mae": float(np.mean(
                np.abs(predicted_sigma[mask] - true_sigma[mask])
            )) if mask.any() else 0.0,
        }
    return {
        "count": int(true_sigma.size),
        "sigma_mae": float(np.mean(np.abs(predicted_sigma - true_sigma))),
        "sigma_rmse": float(np.sqrt(np.mean(
            (predicted_sigma - true_sigma) ** 2.0
        ))),
        "pearson": safe_correlation(true_sigma, predicted_sigma),
        "band_accuracy": float(np.mean(true_band == predicted_band)),
        "within_0.0025_rate": float(np.mean(
            np.abs(predicted_sigma - true_sigma) <= 0.0025
        )),
        "confusion_matrix": {
            BAND_NAMES[row]: {
                BAND_NAMES[column]: int(matrix[row, column])
                for column in range(3)
            }
            for row in range(3)
        },
        "per_true_band": per_band,
    }


def aggregate_shapes(paths, true_sigma, predicted_sigma):
    grouped = {}
    for path, true_value, predicted_value in zip(
        paths,
        true_sigma,
        predicted_sigma,
    ):
        grouped.setdefault(str(path), {"true": [], "predicted": []})
        grouped[str(path)]["true"].append(float(true_value))
        grouped[str(path)]["predicted"].append(float(predicted_value))
    rows = []
    for path, values in sorted(grouped.items()):
        rows.append({
            "rel_path": path,
            "true_sigma": float(np.median(values["true"])),
            "predicted_sigma": float(np.median(values["predicted"])),
            "patch_count": len(values["true"]),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/vm_ssl/checkpoint_best.pkl",
    )
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-max-residual", type=float, default=0.012)
    parser.add_argument("--stage2-max-residual", type=float, default=0.008)
    parser.add_argument("--min-residual-ratio", type=float, default=0.2)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.020)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_patch_file(args.dataset)
    coarse = cache_coarse(
        args.checkpoint,
        data,
        args.coarse_cache,
        args.batch_size,
        args.coarse_mode,
    )
    model = MultiStageGeometryRefiner(
        num_stages=args.stages,
        stage_max_residuals=(
            args.stage1_max_residual,
            args.stage2_max_residual,
        ),
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        adaptive_v2=True,
        min_residual_ratio=args.min_residual_ratio,
    )
    model.load(args.refiner_checkpoint)

    true_sigma = data["score_sigma"].reshape(-1).astype(np.float64)
    strengths = predict_noise_strength(
        model,
        coarse,
        data["pc_noisy"],
        args.batch_size,
    )
    summary = {"stages": {}, "args": vars(args)}
    sigma_range = float(args.sigma_max) - float(args.sigma_min)
    for stage_index, strength in enumerate(strengths):
        predicted_sigma = float(args.sigma_min) + sigma_range * strength
        patch_rows = [
            {
                "index": index,
                "rel_path": str(data["rel_path"][index]),
                "true_sigma": float(true_sigma[index]),
                "predicted_sigma": float(predicted_sigma[index]),
                "true_band": BAND_NAMES[int(sigma_to_band(true_sigma[index]))],
                "predicted_band": BAND_NAMES[
                    int(sigma_to_band(predicted_sigma[index]))
                ],
            }
            for index in range(true_sigma.size)
        ]
        shape_rows = aggregate_shapes(
            data["rel_path"],
            true_sigma,
            predicted_sigma,
        )
        shape_true = np.asarray(
            [row["true_sigma"] for row in shape_rows],
            dtype=np.float64,
        )
        shape_predicted = np.asarray(
            [row["predicted_sigma"] for row in shape_rows],
            dtype=np.float64,
        )
        stage_name = f"stage{stage_index + 1}"
        summary["stages"][stage_name] = {
            "patch_level": summarize(true_sigma, predicted_sigma),
            "shape_level": summarize(shape_true, shape_predicted),
        }
        write_csv(out_dir / f"{stage_name}_patch.csv", patch_rows)
        write_csv(out_dir / f"{stage_name}_shape.csv", shape_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
