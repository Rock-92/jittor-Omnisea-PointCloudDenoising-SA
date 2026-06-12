import argparse
import csv
import json
import sys
from pathlib import Path

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


def read_datalist(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_clean(clean):
    center = (clean.max(axis=0) + clean.min(axis=0)) / 2.0
    centered = clean - center
    scale = np.sqrt((centered ** 2.0).sum(axis=1)).max()
    return (
        (centered / max(float(scale), 1e-12)).astype(
            np.float32,
            copy=False,
        ),
        center.astype(np.float32, copy=False),
        float(scale),
    )


def recreate_clean(mesh, seed, num_samples, num_vertex_samples):
    np.random.seed(seed)
    clean, _, _, _ = sample_vertex_groups(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        num_samples=num_samples,
        num_vertex_samples=num_vertex_samples,
    )
    return clean.astype(np.float32, copy=False)


def tangent_repulsion(points, k, strength, iterations, max_step):
    output = points.astype(np.float64, copy=True)
    for _ in range(int(iterations)):
        tree = cKDTree(output)
        distances, indices = tree.query(
            output,
            k=min(int(k) + 1, output.shape[0]),
            workers=-1,
        )
        distances = distances[:, 1:]
        neighbors = output[indices[:, 1:]]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum(
            "nki,nkj->nij",
            centered,
            centered,
        ) / max(centered.shape[1], 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]

        away = output[:, None, :] - neighbors
        normal_component = (
            away * normals[:, None, :]
        ).sum(axis=2, keepdims=True)
        tangent = away - normal_component * normals[:, None, :]
        tangent_norm = np.linalg.norm(tangent, axis=2)
        local_scale = np.maximum(
            np.median(distances, axis=1, keepdims=True),
            1e-8,
        )
        weights = np.exp(
            -(distances / (1.5 * local_scale)) ** 2.0
        ) / np.maximum(distances, 1e-8)
        direction = (
            tangent
            / np.maximum(tangent_norm[:, :, None], 1e-8)
            * weights[:, :, None]
        ).sum(axis=1)
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction /= np.maximum(direction_norm, 1e-8)

        local_step = (
            float(strength)
            * np.maximum(distances[:, :1], 1e-8)
        )
        local_step = np.minimum(local_step, float(max_step))
        displacement = direction * local_step
        displacement -= displacement.mean(axis=0, keepdims=True)
        output += displacement
    return output.astype(np.float32, copy=False)


def score_prediction(
    prediction,
    clean,
    noisy,
    mesh_vertices,
    mesh_faces,
    cd_noisy,
    p2s_noisy,
):
    cd = chamfer_distance(prediction, clean, normalize=True)
    p2s = point_to_surface_distance(
        prediction,
        mesh_vertices,
        mesh_faces,
        normalize_ref_pc=clean,
    )
    cd_score = metric_to_score(cd, cd_noisy)
    p2s_score = metric_to_score(p2s, p2s_noisy)
    return {
        "cd": cd,
        "p2s": p2s,
        "cd_score": cd_score,
        "p2s_score": p2s_score,
        "final_score": 0.5 * (cd_score + p2s_score),
    }


def summarize(rows, source, strength):
    selected = [
        row
        for row in rows
        if row["source"] == source and row["strength"] == strength
    ]
    return {
        "count": len(selected),
        "cd_score": float(np.mean([row["cd_score"] for row in selected])),
        "p2s_score": float(np.mean([row["p2s_score"] for row in selected])),
        "final_score": float(
            np.mean([row["final_score"] for row in selected])
        ),
        "cd_gain": float(np.mean([row["cd_gain"] for row in selected])),
        "p2s_gain": float(np.mean([row["p2s_gain"] for row in selected])),
        "final_gain": float(
            np.mean([row["final_gain"] for row in selected])
        ),
        "improved_rate": float(
            np.mean([row["final_gain"] > 0.0 for row in selected])
        ),
        "degraded_ge_1_rate": float(
            np.mean([row["final_gain"] <= -1.0 for row in selected])
        ),
        "mean_displacement": float(
            np.mean([row["mean_displacement"] for row in selected])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default=(
            "outputs_result/outputs_refinement/"
            "full_cloud_refinement_heun_20"
        ),
    )
    parser.add_argument("--mesh-root", default="dataset_clean")
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["vm", "vm_refined"],
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.20, 0.35, 0.50],
    )
    parser.add_argument("--max-shapes", type=int, default=20)
    parser.add_argument("--clean-seed", type=int, default=123)
    parser.add_argument("--clean-seed-offset", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=789)
    parser.add_argument("--noise-std", type=float, default=0.020)
    parser.add_argument("--num-samples", type=int, default=32768)
    parser.add_argument("--num-vertex-samples", type=int, default=1024)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-step", type=float, default=0.002)
    parser.add_argument(
        "--out-dir",
        default="outputs/coverage_restoration_probe",
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = read_datalist(args.datalist)
    usable = [
        rel_path
        for rel_path in rel_paths
        if all(
            (
                result_root
                / source
                / rel_path
                / "denoised.npy"
            ).exists()
            for source in args.sources
        )
    ]
    if args.max_shapes > 0:
        usable = usable[:args.max_shapes]
    if not usable:
        raise FileNotFoundError("no saved full-cloud predictions found")

    noise_rng = np.random.default_rng(args.noise_seed)
    rows = []
    baseline_scores = {}
    for shape_index, rel_path in enumerate(usable):
        print(f"[{shape_index + 1}/{len(usable)}] {rel_path}", flush=True)
        mesh_path = (
            Path(args.mesh_root)
            / rel_path
            / "models/model_normalized.obj"
        )
        mesh = trimesh.load(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        clean_raw = recreate_clean(
            mesh,
            (
                args.clean_seed
                + args.clean_seed_offset
                + rel_paths.index(rel_path)
            ),
            args.num_samples,
            args.num_vertex_samples,
        )
        clean, center, scale = normalize_clean(clean_raw)
        noisy = (
            clean
            + noise_rng.standard_normal(clean.shape).astype(np.float32)
            * float(args.noise_std)
        ).astype(np.float32, copy=False)
        mesh_vertices = (
            np.asarray(mesh.vertices, dtype=np.float32) - center
        ) / max(scale, 1e-12)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int32)
        cd_noisy = chamfer_distance(noisy, clean, normalize=True)
        p2s_noisy = point_to_surface_distance(
            noisy,
            mesh_vertices,
            mesh_faces,
            normalize_ref_pc=clean,
        )

        for source in args.sources:
            prediction = np.load(
                result_root / source / rel_path / "denoised.npy"
            ).astype(np.float32, copy=False)
            baseline = score_prediction(
                prediction,
                clean,
                noisy,
                mesh_vertices,
                mesh_faces,
                cd_noisy,
                p2s_noisy,
            )
            baseline_scores[(rel_path, source)] = baseline
            for strength in args.strengths:
                restored = tangent_repulsion(
                    prediction,
                    args.k,
                    strength,
                    args.iterations,
                    args.max_step,
                )
                score = score_prediction(
                    restored,
                    clean,
                    noisy,
                    mesh_vertices,
                    mesh_faces,
                    cd_noisy,
                    p2s_noisy,
                )
                displacement = np.linalg.norm(
                    restored - prediction,
                    axis=1,
                )
                row = {
                    "rel_path": rel_path,
                    "source": source,
                    "strength": float(strength),
                    "mean_displacement": float(displacement.mean()),
                    **score,
                    "cd_gain": score["cd_score"] - baseline["cd_score"],
                    "p2s_gain": (
                        score["p2s_score"] - baseline["p2s_score"]
                    ),
                    "final_gain": (
                        score["final_score"] - baseline["final_score"]
                    ),
                }
                rows.append(row)
                print(
                    f"  {source} strength={strength:.3f} "
                    f"gain={row['final_gain']:+.3f} "
                    f"cd={row['cd_gain']:+.3f} "
                    f"p2s={row['p2s_gain']:+.3f}",
                    flush=True,
                )
        write_csv(out_dir / "shape_eval.csv", rows)

    baseline_summary = {}
    probe_summary = {}
    for source in args.sources:
        source_baselines = [
            baseline_scores[(rel_path, source)]
            for rel_path in usable
        ]
        baseline_summary[source] = {
            key: float(np.mean([row[key] for row in source_baselines]))
            for key in ["cd_score", "p2s_score", "final_score"]
        }
        probe_summary[source] = {
            str(strength): summarize(rows, source, float(strength))
            for strength in args.strengths
        }
    summary = {
        "shape_count": len(usable),
        "baseline": baseline_summary,
        "probe": probe_summary,
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
