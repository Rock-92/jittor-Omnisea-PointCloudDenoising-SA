import argparse
from collections import Counter
import csv
import json
import sys
from pathlib import Path

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


def read_datalist(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def category_of(rel_path):
    parts = Path(str(rel_path)).parts
    return parts[1] if len(parts) > 1 else parts[0]


def choose_paths(rel_paths, num_shapes, rng, category_reference=None):
    rel_paths = list(rel_paths)
    sample_count = min(int(num_shapes), len(rel_paths))
    if sample_count == 0:
        return []
    if not category_reference:
        return rng.choice(
            np.asarray(rel_paths),
            size=sample_count,
            replace=False,
        ).tolist()

    available_counts = Counter(category_of(path) for path in rel_paths)
    reference_counts = Counter(
        category_of(path) for path in category_reference
    )
    weights = np.asarray(
        [
            reference_counts.get(category_of(path), 0.0)
            / available_counts[category_of(path)]
            for path in rel_paths
        ],
        dtype=np.float64,
    )
    positive = np.flatnonzero(weights > 0.0)
    sample_count = min(sample_count, int(positive.size))
    weights /= weights.sum()
    indices = rng.choice(
        np.arange(len(rel_paths)),
        size=sample_count,
        replace=False,
        p=weights,
    )
    return [rel_paths[int(index)] for index in indices]


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


def parse_noise_bands(value):
    if not value:
        return None
    bands = []
    for item in value.split(","):
        lower, upper = item.split(":", maxsplit=1)
        bands.append((float(lower), float(upper)))
    return bands


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_prediction(
    prediction,
    clean,
    mesh_vertices,
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
            mesh_vertices,
            mesh_faces,
            normalize_ref_pc=clean,
        ),
        p2s_noisy,
    )
    return cd_score, p2s_score, 0.5 * (cd_score + p2s_score)


def summarize(rows):
    gains = np.asarray([row["final_gain"] for row in rows])
    return {
        "count": len(rows),
        "cd_score": float(np.mean([row["cd_score"] for row in rows])),
        "p2s_score": float(np.mean([row["p2s_score"] for row in rows])),
        "final_score": float(
            np.mean([row["final_score"] for row in rows])
        ),
        "cd_gain": float(np.mean([row["cd_gain"] for row in rows])),
        "p2s_gain": float(np.mean([row["p2s_gain"] for row in rows])),
        "final_gain": float(gains.mean()),
        "improved_rate": float(np.mean(gains > 0.0)),
        "degraded_ge_1_rate": float(np.mean(gains <= -1.0)),
    }


def reconstruct_order_and_rng(metadata, result_root):
    run_args = metadata["args"]
    rng = np.random.default_rng(int(run_args["seed"]))
    excluded = set()
    for dataset_path in run_args.get("exclude_dataset", []):
        dataset = np.load(dataset_path, allow_pickle=True)
        excluded.update(str(path) for path in dataset["rel_path"].tolist())
    rel_paths = [
        rel_path
        for rel_path in read_datalist(run_args["datalist"])
        if rel_path not in excluded
    ]
    clean_root = Path(run_args["clean_root"])
    mesh_root = Path(run_args["mesh_root"])
    usable = [
        rel_path
        for rel_path in rel_paths
        if (clean_root / rel_path / "clean.npy").exists()
        and (
            mesh_root / rel_path / "models/model_normalized.obj"
        ).exists()
    ]
    category_reference = run_args.get("category_reference_datalist")
    if category_reference:
        usable = choose_paths(
            usable,
            run_args["max_shapes"],
            rng,
            category_reference=read_datalist(category_reference),
        )
    elif int(run_args.get("max_shapes", 0)) > 0:
        usable = usable[:int(run_args["max_shapes"])]

    csv_rows = read_csv(result_root / "shape_eval.csv")
    saved_order = [row["rel_path"] for row in csv_rows]
    if usable != saved_order:
        raise ValueError(
            "reconstructed shape order does not match saved shape_eval.csv"
        )
    return usable, csv_rows, rng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="outputs/full_cloud_continuous_spacing_30",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument("--sigma-tolerance", type=float, default=1e-6)
    parser.add_argument("--score-tolerance", type=float, default=1e-3)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else result_root / "alpha_scan"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (result_root / "summary.json").read_text(encoding="utf-8")
    )
    run_args = metadata["args"]
    noise_bands = parse_noise_bands(
        run_args.get("balanced_noise_bands")
    )
    usable, saved_rows, rng = reconstruct_order_and_rng(
        metadata,
        result_root,
    )

    rows = []
    for shape_index, (rel_path, saved_row) in enumerate(
        zip(usable, saved_rows)
    ):
        if noise_bands:
            lower, upper = noise_bands[shape_index % len(noise_bands)]
            noise_std = float(rng.uniform(lower, upper))
            noise_band = f"{lower:.4f}:{upper:.4f}"
        else:
            noise_std = float(run_args["noise_std"])
            noise_band = "fixed"
        saved_sigma = float(saved_row["noise_std"])
        if abs(noise_std - saved_sigma) > float(args.sigma_tolerance):
            raise ValueError(
                f"{rel_path}: reconstructed sigma {noise_std} "
                f"does not match saved sigma {saved_sigma}"
            )

        clean_raw = np.load(
            Path(run_args["clean_root"]) / rel_path / "clean.npy"
        ).astype(np.float32, copy=False)
        clean, center, scale = normalize_clean(clean_raw)
        noisy = (
            clean
            + rng.standard_normal(clean.shape).astype(np.float32)
            * noise_std
        ).astype(np.float32, copy=False)
        mesh = trimesh.load(
            str(
                Path(run_args["mesh_root"])
                / rel_path
                / "models/model_normalized.obj"
            ),
            process=False,
        )
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
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
        coarse = np.load(
            result_root / "vm" / rel_path / "denoised.npy"
        ).astype(np.float32, copy=False)
        refined = np.load(
            result_root / "vm_refined" / rel_path / "denoised.npy"
        ).astype(np.float32, copy=False)
        coarse_scores = score_prediction(
            coarse,
            clean,
            mesh_vertices,
            mesh_faces,
            cd_noisy,
            p2s_noisy,
        )
        saved_coarse_final = float(saved_row["coarse_final_score"])
        if (
            abs(coarse_scores[2] - saved_coarse_final)
            > float(args.score_tolerance)
        ):
            raise ValueError(
                f"{rel_path}: reconstructed coarse score "
                f"{coarse_scores[2]} does not match saved score "
                f"{saved_coarse_final}"
            )
        for alpha in args.alphas:
            prediction = coarse + float(alpha) * (refined - coarse)
            scores = score_prediction(
                prediction,
                clean,
                mesh_vertices,
                mesh_faces,
                cd_noisy,
                p2s_noisy,
            )
            rows.append({
                "rel_path": rel_path,
                "noise_std": noise_std,
                "noise_band": noise_band,
                "alpha": float(alpha),
                "cd_score": scores[0],
                "p2s_score": scores[1],
                "final_score": scores[2],
                "cd_gain": scores[0] - coarse_scores[0],
                "p2s_gain": scores[1] - coarse_scores[1],
                "final_gain": scores[2] - coarse_scores[2],
            })
        print(
            f"[{shape_index + 1}/{len(usable)}] {rel_path} "
            f"band={noise_band}",
            flush=True,
        )
        write_csv(out_dir / "shape_eval.csv", rows)

    overall = {}
    by_band = {}
    for alpha in args.alphas:
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        overall[str(alpha)] = summarize(alpha_rows)
        by_band[str(alpha)] = {
            band: summarize([
                row for row in alpha_rows if row["noise_band"] == band
            ])
            for band in sorted({row["noise_band"] for row in alpha_rows})
        }
    best_global_alpha = max(
        args.alphas,
        key=lambda alpha: overall[str(alpha)]["final_gain"],
    )
    bands = sorted({row["noise_band"] for row in rows})
    best_alpha_by_band = {
        band: max(
            args.alphas,
            key=lambda alpha: by_band[str(alpha)][band]["final_gain"],
        )
        for band in bands
    }
    adaptive_rows = [
        row
        for row in rows
        if row["alpha"] == best_alpha_by_band[row["noise_band"]]
    ]
    summary = {
        "shape_count": len(usable),
        "overall": overall,
        "by_noise_band": by_band,
        "best_global_alpha": float(best_global_alpha),
        "best_global": overall[str(best_global_alpha)],
        "best_alpha_by_band": {
            band: float(alpha)
            for band, alpha in best_alpha_by_band.items()
        },
        "band_adaptive_oracle": summarize(adaptive_rows),
        "args": vars(args),
        "source_run_args": run_args,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
