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
from src.model.refinement import MultiStageGeometryRefiner  # noqa: E402
from tools.train_multistage_refinement_probe import (  # noqa: E402
    add_score_bands,
    cache_coarse,
    full_summary,
    load_patch_file,
    predict,
    score_rows,
    write_csv,
)
from tools.hard_patch_common import quantile_summary  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs_result/outputs_hardware/"
            "checkpoints/vm_ssl/checkpoint_best.pkl"
        ),
    )
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="fixed")
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-max-residual", type=float, default=0.012)
    parser.add_argument("--stage2-max-residual", type=float, default=0.008)
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
    )
    model.load(args.refiner_checkpoint)
    prediction = predict(
        model,
        coarse,
        data["pc_noisy"],
        args.batch_size,
    )
    rows = score_rows(
        data["pc_noisy"],
        data["pc_clean"],
        coarse,
        prediction,
        data["rel_path"],
    )
    thresholds = add_score_bands(rows)
    summary = full_summary(rows)
    formal_rows = []
    if all(
        key in data
        for key in ["patch_center", "normalize_center", "normalize_scale"]
    ):
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
                    mesh = trimesh.util.concatenate(
                        tuple(mesh.geometry.values())
                    )
                mesh_cache[rel_path] = (
                    np.asarray(mesh.vertices, dtype=np.float32),
                    np.asarray(mesh.faces, dtype=np.int32),
                )
            mesh_vertices, mesh_faces = mesh_cache[rel_path]
            center = data["normalize_center"][index]
            scale = max(float(data["normalize_scale"][index]), 1e-12)
            normalized_vertices = (mesh_vertices - center) / scale
            patch_center = data["patch_center"][index]
            noisy_abs = data["pc_noisy"][index] + patch_center
            clean_abs = data["pc_clean"][index] + patch_center
            coarse_abs = coarse[index] + patch_center
            refined_abs = prediction[index] + patch_center

            cd_noisy = chamfer_distance(noisy_abs, clean_abs, normalize=True)
            p2s_noisy = point_to_surface_distance(
                noisy_abs,
                normalized_vertices,
                mesh_faces,
                normalize_ref_pc=clean_abs,
            )
            formal_row = {
                "index": index,
                "rel_path": rel_path,
                "coarse_cd_score": metric_to_score(
                    chamfer_distance(
                        coarse_abs,
                        clean_abs,
                        normalize=True,
                    ),
                    cd_noisy,
                ),
                "refined_cd_score": metric_to_score(
                    chamfer_distance(
                        refined_abs,
                        clean_abs,
                        normalize=True,
                    ),
                    cd_noisy,
                ),
                "coarse_p2s_score": metric_to_score(
                    point_to_surface_distance(
                        coarse_abs,
                        normalized_vertices,
                        mesh_faces,
                        normalize_ref_pc=clean_abs,
                    ),
                    p2s_noisy,
                ),
                "refined_p2s_score": metric_to_score(
                    point_to_surface_distance(
                        refined_abs,
                        normalized_vertices,
                        mesh_faces,
                        normalize_ref_pc=clean_abs,
                    ),
                    p2s_noisy,
                ),
            }
            formal_row["coarse_final_score"] = 0.5 * (
                formal_row["coarse_cd_score"]
                + formal_row["coarse_p2s_score"]
            )
            formal_row["refined_final_score"] = 0.5 * (
                formal_row["refined_cd_score"]
                + formal_row["refined_p2s_score"]
            )
            formal_row["final_gain"] = (
                formal_row["refined_final_score"]
                - formal_row["coarse_final_score"]
            )
            formal_rows.append(formal_row)
        summary["formal"] = {
            "count": len(formal_rows),
            "coarse_cd_score": float(
                np.mean([row["coarse_cd_score"] for row in formal_rows])
            ),
            "refined_cd_score": float(
                np.mean([row["refined_cd_score"] for row in formal_rows])
            ),
            "coarse_p2s_score": float(
                np.mean([row["coarse_p2s_score"] for row in formal_rows])
            ),
            "refined_p2s_score": float(
                np.mean([row["refined_p2s_score"] for row in formal_rows])
            ),
            "coarse_final_score": float(
                np.mean([row["coarse_final_score"] for row in formal_rows])
            ),
            "refined_final_score": float(
                np.mean([row["refined_final_score"] for row in formal_rows])
            ),
            "final_gain": quantile_summary(
                [row["final_gain"] for row in formal_rows]
            ),
            "improved_rate": float(
                np.mean([row["final_gain"] > 0.0 for row in formal_rows])
            ),
        }
    summary["shape_count"] = len(set(data["rel_path"].tolist()))
    summary["score_band_thresholds"] = thresholds.tolist()
    summary["args"] = vars(args)
    write_csv(out_dir / "patch_eval.csv", rows)
    write_csv(out_dir / "formal_eval.csv", formal_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
