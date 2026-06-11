import argparse
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
    load_hard_patch_npz,
    load_model,
    quantile_summary,
    write_csv,
    write_json,
)
from tools.train_refinement_probe import (  # noqa: E402
    evaluate_predictions,
    generate_coarse,
    predict_refined,
)


def summarize_gain(rows):
    gains = np.asarray([row["score_gain"] for row in rows], dtype=np.float64)
    return {
        "count": int(gains.size),
        "coarse_score_mean": float(
            np.mean([row["coarse_score"] for row in rows])
        ),
        "refined_score_mean": float(
            np.mean([row["refined_score"] for row in rows])
        ),
        "score_gain": quantile_summary(gains),
        "improved_rate": float(np.mean(gains > 0.0)),
        "gain_ge_1_rate": float(np.mean(gains >= 1.0)),
        "degraded_ge_1_rate": float(np.mean(gains <= -1.0)),
    }


def grouped_summary(rows, key):
    groups = {}
    values = sorted({str(row[key]) for row in rows})
    for value in values:
        selected = [row for row in rows if str(row[key]) == value]
        groups[value] = summarize_gain(selected)
    return groups


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
        "--refiner-checkpoint",
        default=(
            "outputs_result/outputs_analysis/"
            "refinement_probe_v1_full/refiner_best.pkl"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "outputs_result/outputs_analysis/"
            "hardware_patch_diagnosis/all96/hard_patches.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs_result/outputs_analysis/"
            "refinement_probe_v1_all96"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-residual", type=float, default=0.006)
    parser.add_argument("--tangent-scale", type=float, default=0.25)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_hard_patch_npz(args.dataset)
    noisy_np = data["pc_noisy"]
    clean_np = data["pc_clean"]
    sigma_np = data["score_sigma"]

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

    refiner = GeometryResidualRefiner(
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        max_residual=args.max_residual,
        tangent_scale=args.tangent_scale,
    )
    refiner.load(args.refiner_checkpoint)
    refined_np = predict_refined(
        refiner,
        coarse_np,
        noisy_np,
        args.batch_size,
    )

    indices = np.arange(noisy_np.shape[0])
    coarse_rows, _ = evaluate_predictions(
        noisy_np,
        clean_np,
        coarse_np,
        indices,
    )
    refined_rows, _ = evaluate_predictions(
        noisy_np,
        clean_np,
        refined_np,
        indices,
    )
    coarse_scores = np.asarray(
        [row["cd_score"] for row in coarse_rows],
        dtype=np.float64,
    )
    low_threshold, high_threshold = np.quantile(coarse_scores, [1 / 3, 2 / 3])

    rows = []
    for index, (coarse_row, refined_row) in enumerate(
        zip(coarse_rows, refined_rows)
    ):
        score = coarse_row["cd_score"]
        if score <= low_threshold:
            score_band = "low"
        elif score <= high_threshold:
            score_band = "mid"
        else:
            score_band = "high"
        rows.append(
            {
                "index": index,
                "rel_path": str(data["rel_path"][index]),
                "seed_idx": int(data["seed_idx"][index]),
                "geometry_category": str(data["geometry_category"][index]),
                "patch_scale": float(data["patch_scale"][index]),
                "score_band": score_band,
                "coarse_score": float(score),
                "refined_score": float(refined_row["cd_score"]),
                "score_gain": float(refined_row["cd_score"] - score),
                "coarse_cosine": float(coarse_row["cosine_mean"]),
                "refined_cosine": float(refined_row["cosine_mean"]),
                "coarse_length_ratio": float(
                    coarse_row["length_ratio_mean"]
                ),
                "refined_length_ratio": float(
                    refined_row["length_ratio_mean"]
                ),
            }
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "refiner_checkpoint": str(Path(args.refiner_checkpoint).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "score_band_thresholds": {
            "low_max": float(low_threshold),
            "mid_max": float(high_threshold),
        },
        "overall": summarize_gain(rows),
        "by_score_band": grouped_summary(rows, "score_band"),
        "by_geometry_category": grouped_summary(
            rows,
            "geometry_category",
        ),
        "args": vars(args),
    }
    write_csv(out_dir / "patch_eval.csv", rows)
    write_json(out_dir / "summary.json", summary)
    np.savez_compressed(
        out_dir / "predictions.npz",
        pc_noisy=noisy_np,
        pc_clean=clean_np,
        pc_coarse=coarse_np,
        pc_refined=refined_np,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
