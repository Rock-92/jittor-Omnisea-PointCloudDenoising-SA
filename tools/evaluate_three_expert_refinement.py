import argparse
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
from src.model.noise_classifier import PatchNoiseClassifier  # noqa: E402
from src.model.refinement import MultiStageGeometryRefiner  # noqa: E402
from tools.hard_patch_common import quantile_summary  # noqa: E402
from tools.train_multistage_refinement_probe import (  # noqa: E402
    cache_coarse,
    load_patch_file,
    predict,
)
from tools.train_patch_noise_classifier import (  # noqa: E402
    BAND_NAMES,
    classification_summary,
    sigma_to_label,
)


def softmax(logits, temperature):
    scaled = logits / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def routing_weights(
    logits,
    predicted_sigma,
    temperature,
    mode="soft",
    high_threshold=0.45,
    adjacent_boundary=0.0125,
    high_sigma_boundary=0.0145,
):
    probabilities = softmax(logits, temperature)
    if mode == "soft":
        return probabilities
    if mode != "sparse":
        raise ValueError(f"unsupported routing mode: {mode}")

    predicted_sigma = np.asarray(predicted_sigma).reshape(-1)
    if predicted_sigma.shape[0] != probabilities.shape[0]:
        raise ValueError("predicted sigma count does not match classifier logits")

    weights = np.zeros_like(probabilities)
    predicted_band = probabilities.argmax(axis=1)
    high_only = (
        (
            (predicted_band == 2)
            & (probabilities[:, 2] >= float(high_threshold))
        )
        | (predicted_sigma >= float(high_sigma_boundary))
    )
    weights[high_only, 2] = 1.0

    remaining = ~high_only
    lower = remaining & (predicted_sigma < float(adjacent_boundary))
    upper = remaining & ~lower
    weights[lower, :2] = probabilities[lower, :2]
    weights[upper, 1:] = probabilities[upper, 1:]
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights


def classifier_outputs(model, noisy, coarse, point_indices, batch_size):
    model.eval()
    logits_all = []
    sigma_all = []
    with jt.no_grad():
        for start in range(0, noisy.shape[0], batch_size):
            end = min(start + batch_size, noisy.shape[0])
            logits, output = model(
                jt.array(noisy[start:end, point_indices, :]),
                jt.array(coarse[start:end, point_indices, :]),
            )
            logits_all.append(logits.numpy())
            sigma_all.append(
                0.005
                + 0.015
                * output["sigma_normalized"].numpy().reshape(-1)
            )
    return np.concatenate(logits_all), np.concatenate(sigma_all)


def load_refiner(path, args, adaptive_v2):
    model = MultiStageGeometryRefiner(
        num_stages=args.stages,
        stage_max_residuals=(
            args.stage1_max_residual,
            args.stage2_max_residual,
        ),
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
        adaptive_v2=adaptive_v2,
        min_residual_ratio=args.min_residual_ratio,
    )
    model.load(path)
    return model


def score_prediction(
    noisy,
    clean,
    prediction,
    normalized_vertices,
    mesh_faces,
    cd_noisy,
    p2s_noisy,
):
    cd_score = metric_to_score(
        chamfer_distance(prediction, clean, normalize=True),
        cd_noisy,
    )
    p2s_score = metric_to_score(
        point_to_surface_distance(
            prediction,
            normalized_vertices,
            mesh_faces,
            normalize_ref_pc=clean,
        ),
        p2s_noisy,
    )
    return cd_score, p2s_score, 0.5 * (cd_score + p2s_score)


def summarize_rows(rows, prefix):
    gains = np.asarray([row[f"{prefix}_gain"] for row in rows])
    return {
        "count": len(rows),
        "cd_score": float(np.mean(
            [row[f"{prefix}_cd_score"] for row in rows]
        )),
        "p2s_score": float(np.mean(
            [row[f"{prefix}_p2s_score"] for row in rows]
        )),
        "final_score": float(np.mean(
            [row[f"{prefix}_final_score"] for row in rows]
        )),
        "gain": quantile_summary(gains),
        "improved_rate": float(np.mean(gains > 0.0)),
        "degraded_ge_1_rate": float(np.mean(gains <= -1.0)),
    }


def grouped_summary(rows, prefix):
    result = {"overall": summarize_rows(rows, prefix), "by_noise_band": {}}
    for band in BAND_NAMES:
        selected = [row for row in rows if row["noise_band"] == band]
        result["by_noise_band"][band] = summarize_rows(selected, prefix)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/vm_ssl/checkpoint_best.pkl",
    )
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--low-refiner", required=True)
    parser.add_argument("--medium-refiner", required=True)
    parser.add_argument("--high-refiner", required=True)
    parser.add_argument("--low-refiner-v1", action="store_true")
    parser.add_argument("--medium-refiner-v1", action="store_true")
    parser.add_argument("--high-refiner-v1", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--routing-mode",
        choices=["soft", "sparse"],
        default="soft",
    )
    parser.add_argument("--high-route-threshold", type=float, default=0.45)
    parser.add_argument("--adjacent-boundary", type=float, default=0.0125)
    parser.add_argument("--high-sigma-boundary", type=float, default=0.0145)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classifier-points", type=int, default=256)
    parser.add_argument("--classifier-k", type=int, default=24)
    parser.add_argument("--classifier-local-dim", type=int, default=96)
    parser.add_argument("--classifier-hidden-dim", type=int, default=192)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-max-residual", type=float, default=0.012)
    parser.add_argument("--stage2-max-residual", type=float, default=0.008)
    parser.add_argument("--min-residual-ratio", type=float, default=0.2)
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
    refiners = [
        load_refiner(
            args.low_refiner,
            args,
            adaptive_v2=not args.low_refiner_v1,
        ),
        load_refiner(
            args.medium_refiner,
            args,
            adaptive_v2=not args.medium_refiner_v1,
        ),
        load_refiner(
            args.high_refiner,
            args,
            adaptive_v2=not args.high_refiner_v1,
        ),
    ]
    expert_predictions = [
        predict(
            model,
            coarse,
            data["pc_noisy"],
            args.batch_size,
        )
        for model in refiners
    ]
    classifier = PatchNoiseClassifier(
        k=args.classifier_k,
        local_dim=args.classifier_local_dim,
        hidden_dim=args.classifier_hidden_dim,
    )
    classifier.load(args.classifier_checkpoint)
    point_indices = np.linspace(
        0,
        data["pc_noisy"].shape[1] - 1,
        min(args.classifier_points, data["pc_noisy"].shape[1]),
        dtype=np.int32,
    )
    logits, predicted_sigma = classifier_outputs(
        classifier,
        data["pc_noisy"],
        coarse,
        point_indices,
        args.batch_size,
    )
    probabilities = routing_weights(
        logits,
        predicted_sigma,
        args.temperature,
        mode=args.routing_mode,
        high_threshold=args.high_route_threshold,
        adjacent_boundary=args.adjacent_boundary,
        high_sigma_boundary=args.high_sigma_boundary,
    )
    soft_prediction = sum(
        expert_predictions[index] * probabilities[:, index, None, None]
        for index in range(3)
    )
    true_sigma = data["score_sigma"].reshape(-1)
    true_labels = sigma_to_label(true_sigma)
    true_oracle = np.stack(
        [
            expert_predictions[int(true_labels[index])][index]
            for index in range(true_labels.size)
        ]
    )
    classifier_summary = classification_summary(
        true_labels,
        logits.argmax(axis=1),
        true_sigma,
        predicted_sigma,
    )

    rows = []
    mesh_cache = {}
    for index in range(data["pc_noisy"].shape[0]):
        rel_path = str(data["rel_path"][index])
        if rel_path not in mesh_cache:
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
            mesh_cache[rel_path] = (
                np.asarray(mesh.vertices, dtype=np.float32),
                np.asarray(mesh.faces, dtype=np.int32),
            )
        mesh_vertices, mesh_faces = mesh_cache[rel_path]
        center = data["normalize_center"][index]
        scale = max(float(data["normalize_scale"][index]), 1e-12)
        normalized_vertices = (mesh_vertices - center) / scale
        patch_center = data["patch_center"][index]
        noisy = data["pc_noisy"][index] + patch_center
        clean = data["pc_clean"][index] + patch_center
        coarse_abs = coarse[index] + patch_center
        cd_noisy = chamfer_distance(noisy, clean, normalize=True)
        p2s_noisy = point_to_surface_distance(
            noisy,
            normalized_vertices,
            mesh_faces,
            normalize_ref_pc=clean,
        )
        coarse_scores = score_prediction(
            noisy,
            clean,
            coarse_abs,
            normalized_vertices,
            mesh_faces,
            cd_noisy,
            p2s_noisy,
        )
        row = {
            "index": index,
            "rel_path": rel_path,
            "noise_sigma": float(true_sigma[index]),
            "noise_band": BAND_NAMES[int(true_labels[index])],
            "classifier_band": BAND_NAMES[int(logits[index].argmax())],
            "classifier_low_probability": float(probabilities[index, 0]),
            "classifier_medium_probability": float(probabilities[index, 1]),
            "classifier_high_probability": float(probabilities[index, 2]),
            "coarse_cd_score": coarse_scores[0],
            "coarse_p2s_score": coarse_scores[1],
            "coarse_final_score": coarse_scores[2],
        }
        for prefix, prediction in [
            ("oracle", true_oracle[index]),
            ("soft", soft_prediction[index]),
        ]:
            scores = score_prediction(
                noisy,
                clean,
                prediction + patch_center,
                normalized_vertices,
                mesh_faces,
                cd_noisy,
                p2s_noisy,
            )
            row[f"{prefix}_cd_score"] = scores[0]
            row[f"{prefix}_p2s_score"] = scores[1]
            row[f"{prefix}_final_score"] = scores[2]
            row[f"{prefix}_gain"] = scores[2] - coarse_scores[2]
        rows.append(row)
        print(f"[{index + 1}/{len(true_labels)}] {rel_path}", flush=True)

    summary = {
        "classifier": classifier_summary,
        "oracle": grouped_summary(rows, "oracle"),
        "soft": grouped_summary(rows, "soft"),
        "routing_loss": {
            "overall": (
                grouped_summary(rows, "oracle")["overall"]["gain"]["mean"]
                - grouped_summary(rows, "soft")["overall"]["gain"]["mean"]
            ),
            "by_noise_band": {
                band: (
                    grouped_summary(rows, "oracle")["by_noise_band"][band][
                        "gain"
                    ]["mean"]
                    - grouped_summary(rows, "soft")["by_noise_band"][band][
                        "gain"
                    ]["mean"]
                )
                for band in BAND_NAMES
            },
        },
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
