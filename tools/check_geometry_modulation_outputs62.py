import argparse
import csv
import json
import random
import sys
from pathlib import Path

import jittor as jt
import numpy as np
import trimesh
from omegaconf import OmegaConf
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups  # noqa: E402
from src.model.parse import get_model  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs_result/outputs6.2/checkpoints/vm/checkpoint_best.pkl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_result/analysis_outputs/geometry_modulation_outputs6.2"


def normalize_pc(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / max(float(scale), 1e-12)).astype(np.float32, copy=False)


def load_model(checkpoint):
    model_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs/model/vm.yaml"),
        resolve=True,
    )
    transform_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "configs/transform/vm.yaml"),
        resolve=True,
    )
    model = get_model(model_config=model_cfg, transform_config=transform_cfg)
    model.load(str(checkpoint))
    model.eval()
    return model


def read_datalist(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def existing_noisy_paths(noisy_root, datalist):
    paths = []
    for rel_path in datalist:
        path = noisy_root / rel_path / "noisy.npy"
        if path.exists():
            paths.append((rel_path, path))
    return paths


def existing_mesh_paths(mesh_root, datalist):
    paths = []
    for rel_path in datalist:
        path = mesh_root / rel_path / "models/model_normalized.obj"
        if path.exists():
            paths.append((rel_path, path))
    return paths


def make_centered_patch(points, rng, patch_size):
    points = points.astype(np.float32, copy=False)
    if points.shape[0] < 4:
        raise ValueError("point cloud is too small for patch sampling")
    k = min(int(patch_size), points.shape[0])
    seed_idx = int(rng.integers(0, points.shape[0]))
    seed = points[seed_idx]
    _, idx = cKDTree(points).query(seed[None, :], k=k)
    idx = np.asarray(idx).reshape(-1)
    patch = points[idx] - seed[None, :]
    return patch.astype(np.float32, copy=False), seed_idx


def sample_noisy_patch(rel_path, npy_path, rng, patch_size):
    points = np.load(npy_path).astype(np.float32, copy=False)
    patch, seed_idx = make_centered_patch(points, rng, patch_size)
    return {
        "source": "test_noisy",
        "rel_path": rel_path,
        "seed_idx": seed_idx,
        "noise_std": None,
        "patch": patch,
    }


def sample_mesh_patch(rel_path, mesh_path, rng, patch_size):
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    clean, _, _, _ = sample_vertex_groups(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        num_samples=32768,
        num_vertex_samples=1024,
    )
    clean = normalize_pc(clean.astype(np.float32, copy=False))
    noise_std = float(rng.uniform(0.005, 0.020))
    noisy = clean + rng.laplace(0, noise_std, size=clean.shape).astype(np.float32)
    patch, seed_idx = make_centered_patch(noisy, rng, patch_size)
    return {
        "source": "dataset_clean_mesh_noise",
        "rel_path": rel_path,
        "seed_idx": seed_idx,
        "noise_std": noise_std,
        "patch": patch,
    }


def to_numpy(var):
    return var.detach().numpy().astype(np.float64, copy=False)


def forward_geometry_and_gate(model, patch):
    x = jt.array(patch[None, :, :].astype(np.float32, copy=False))
    with jt.no_grad():
        _, geometry_feat, gate_embedding = model.encoder(x, return_condition=True)
    return to_numpy(geometry_feat)[0], to_numpy(gate_embedding)[0]


def split_gate(gate):
    num_blocks = gate.shape[-1] // 6
    blocks = []
    for block_idx in range(num_blocks):
        block = gate[:, block_idx * 6 : (block_idx + 1) * 6]
        blocks.append(
            {
                "scale": block[:, :3],
                "temperature": block[:, 3:],
            }
        )
    return blocks


def l2_norm(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt((arr * arr).sum()))


def checkpoint_weight_norms(model):
    state = model.state_dict()
    rows = []
    for block_idx in range(model.encoder.num_blocks):
        row = {"block": block_idx}
        for proj in ("scale_gate_proj", "temperature_proj"):
            weight_key = f"encoder.block_{block_idx}.{proj}.weight"
            bias_key = f"encoder.block_{block_idx}.{proj}.bias"
            weight = state[weight_key].numpy()
            bias = state[bias_key].numpy()
            row[f"{proj}_weight_l2"] = l2_norm(weight)
            row[f"{proj}_bias_l2"] = l2_norm(bias)
            row[f"{proj}_combined_l2"] = l2_norm(weight) + l2_norm(bias)
        rows.append(row)
    return rows


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def pairwise_distances(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    rows = []
    for i in range(vectors.shape[0]):
        for j in range(i + 1, vectors.shape[0]):
            diff = vectors[i] - vectors[j]
            rows.append(float(np.sqrt((diff * diff).sum())))
    return rows


def summarize_patch(sample, geometry, gate):
    blocks = split_gate(gate)
    row = {
        "source": sample["source"],
        "rel_path": sample["rel_path"],
        "seed_idx": sample["seed_idx"],
        "noise_std": sample["noise_std"],
        "patch_size": int(sample["patch"].shape[0]),
        "geometry_mean_l2": l2_norm(geometry.mean(axis=0)),
        "geometry_point_std_mean": float(geometry.std(axis=0).mean()),
        "gate_mean_l2": l2_norm(gate.mean(axis=0)),
        "gate_point_std_mean": float(gate.std(axis=0).mean()),
    }
    for block_idx, block in enumerate(blocks):
        scale = block["scale"]
        temperature = block["temperature"]
        row[f"block_{block_idx}_scale_gate_mean_abs_from_1"] = float(
            np.abs(scale - 1.0).mean()
        )
        row[f"block_{block_idx}_temperature_mean_abs_from_1"] = float(
            np.abs(temperature - 1.0).mean()
        )
        row[f"block_{block_idx}_scale_gate_point_std_mean"] = float(
            scale.std(axis=0).mean()
        )
        row[f"block_{block_idx}_temperature_point_std_mean"] = float(
            temperature.std(axis=0).mean()
        )
    return row


def controlled_perturbation(model, patch, seed):
    rng = np.random.default_rng(seed)
    geom_a, gate_a = forward_geometry_and_gate(model, patch)
    geom_repeat, gate_repeat = forward_geometry_and_gate(model, patch)
    jitter = rng.normal(0, 0.01, size=patch.shape).astype(np.float32)
    scaled = patch * np.array([1.08, 0.94, 1.03], dtype=np.float32)[None, :]
    perturbed = (scaled + jitter).astype(np.float32, copy=False)
    geom_b, gate_b = forward_geometry_and_gate(model, perturbed)
    return {
        "repeat_geometry_max_abs_diff": float(np.abs(geom_a - geom_repeat).max()),
        "repeat_gate_max_abs_diff": float(np.abs(gate_a - gate_repeat).max()),
        "perturbed_geometry_mean_l2_diff": l2_norm(geom_a.mean(axis=0) - geom_b.mean(axis=0)),
        "perturbed_gate_mean_l2_diff": l2_norm(gate_a.mean(axis=0) - gate_b.mean(axis=0)),
        "perturbed_gate_max_abs_diff": float(np.abs(gate_a - gate_b).max()),
    }


def make_samples(args, rng):
    samples = []
    test_paths = existing_noisy_paths(
        Path(args.noisy_root),
        read_datalist(PROJECT_ROOT / args.test_datalist),
    )
    mesh_paths = existing_mesh_paths(
        Path(args.mesh_root),
        read_datalist(PROJECT_ROOT / args.mesh_datalist),
    )
    if not test_paths and not mesh_paths:
        raise FileNotFoundError("No usable test_noisy or dataset_clean samples were found.")

    rng.shuffle(test_paths)
    rng.shuffle(mesh_paths)
    for rel_path, path in test_paths[: args.num_noisy_patches]:
        samples.append(sample_noisy_patch(rel_path, path, rng, args.patch_size))
    for rel_path, path in mesh_paths[: args.num_mesh_patches]:
        samples.append(sample_mesh_patch(rel_path, path, rng, args.patch_size))
    if len(samples) < 3:
        raise ValueError("Need at least 3 patches to compute pairwise correlations.")
    return samples


def classify(summary):
    weight_ok = summary["checks"]["checkpoint_projection_weights_nonzero"]
    non_uniform = summary["checks"]["gate_non_uniform"]
    different = summary["checks"]["different_geometry_changes_gate"]
    corr = summary["pairwise"]["gate_geometry_pearson"]
    repeat_ok = summary["checks"]["same_input_deterministic"]

    if not weight_ok or not non_uniform or not different:
        return "FAIL"
    if repeat_ok and np.isfinite(corr) and corr > 0.15:
        return "PASS"
    return "WEAK"


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary):
    lines = [
        "# Geometry Modulation Check: outputs6.2",
        "",
        f"Conclusion: **{summary['conclusion']}**",
        "",
        "## Environment",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- device: `{summary['device']}`",
        f"- patch_size: `{summary['patch_size']}`",
        f"- samples: `{summary['num_samples']}`",
        f"- note: {summary['environment_note']}",
        "",
        "## Main Checks",
        "",
    ]
    for key, value in summary["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Pairwise Geometry/Gate Response",
        "",
        f"- Pearson: `{summary['pairwise']['gate_geometry_pearson']:.6f}`",
        f"- Spearman: `{summary['pairwise']['gate_geometry_spearman']:.6f}`",
        f"- geometry distance mean: `{summary['pairwise']['geometry_distance_mean']:.6f}`",
        f"- gate distance mean: `{summary['pairwise']['gate_distance_mean']:.6f}`",
        "",
        "## Per Block",
        "",
    ]
    for block in summary["blocks"]:
        lines.append(
            "- block {block}: scale_abs_from_1=`{scale:.6f}`, "
            "temperature_abs_from_1=`{temp:.6f}`, corr=`{corr:.6f}`".format(
                block=block["block"],
                scale=block["scale_gate_mean_abs_from_1"],
                temp=block["temperature_mean_abs_from_1"],
                corr=block["gate_pairwise_corr_with_geometry"],
            )
        )
    lines += [
        "",
        "## Controlled Perturbation",
        "",
    ]
    for key, value in summary["controlled_perturbation"].items():
        lines.append(f"- {key}: `{value:.10f}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mesh-root", default=str(PROJECT_ROOT / "dataset_clean"))
    parser.add_argument("--noisy-root", default=str(PROJECT_ROOT / "test_noisy"))
    parser.add_argument("--mesh-datalist", default="datalist/validate.txt")
    parser.add_argument("--test-datalist", default="datalist/test.txt")
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--num-noisy-patches", type=int, default=4)
    parser.add_argument("--num-mesh-patches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    model = load_model(checkpoint)
    weight_rows = checkpoint_weight_norms(model)
    samples = make_samples(args, rng)

    patch_rows = []
    geometry_vectors = []
    gate_vectors = []
    per_block_gate_vectors = [[] for _ in range(model.encoder.num_blocks)]
    all_block_scale_abs = [[] for _ in range(model.encoder.num_blocks)]
    all_block_temp_abs = [[] for _ in range(model.encoder.num_blocks)]

    for idx, sample in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] {sample['source']} {sample['rel_path']}", flush=True)
        geometry, gate = forward_geometry_and_gate(model, sample["patch"])
        patch_rows.append(summarize_patch(sample, geometry, gate))
        geometry_vectors.append(geometry.mean(axis=0))
        gate_vectors.append(gate.mean(axis=0))
        for block_idx, block in enumerate(split_gate(gate)):
            block_vec = np.concatenate(
                [block["scale"].mean(axis=0), block["temperature"].mean(axis=0)]
            )
            per_block_gate_vectors[block_idx].append(block_vec)
            all_block_scale_abs[block_idx].append(float(np.abs(block["scale"] - 1.0).mean()))
            all_block_temp_abs[block_idx].append(
                float(np.abs(block["temperature"] - 1.0).mean())
            )

    geometry_distances = pairwise_distances(np.stack(geometry_vectors, axis=0))
    gate_distances = pairwise_distances(np.stack(gate_vectors, axis=0))
    blocks = []
    for block_idx in range(model.encoder.num_blocks):
        block_gate_distances = pairwise_distances(
            np.stack(per_block_gate_vectors[block_idx], axis=0)
        )
        blocks.append(
            {
                "block": block_idx,
                "scale_gate_mean_abs_from_1": float(np.mean(all_block_scale_abs[block_idx])),
                "temperature_mean_abs_from_1": float(np.mean(all_block_temp_abs[block_idx])),
                "gate_pairwise_corr_with_geometry": pearson(
                    geometry_distances,
                    block_gate_distances,
                ),
            }
        )

    controlled = controlled_perturbation(model, samples[0]["patch"], args.seed + 17)
    max_projection_norm = max(
        row["scale_gate_proj_combined_l2"] + row["temperature_proj_combined_l2"]
        for row in weight_rows
    )
    gate_abs_from_1 = max(
        max(row[f"block_{i}_scale_gate_mean_abs_from_1"] for i in range(model.encoder.num_blocks))
        for row in patch_rows
    )
    temp_abs_from_1 = max(
        max(row[f"block_{i}_temperature_mean_abs_from_1"] for i in range(model.encoder.num_blocks))
        for row in patch_rows
    )
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "device": "cuda" if args.use_cuda else "cpu",
        "environment_note": (
            "CUDA forward was previously observed to run, but CUDA-side numpy/stat "
            "conversion may require cupy in this environment; this check defaults to CPU."
        ),
        "seed": args.seed,
        "patch_size": args.patch_size,
        "num_samples": len(samples),
        "sample_sources": {
            "test_noisy": sum(1 for s in samples if s["source"] == "test_noisy"),
            "dataset_clean_mesh_noise": sum(
                1 for s in samples if s["source"] == "dataset_clean_mesh_noise"
            ),
        },
        "checkpoint_weight_norms": weight_rows,
        "pairwise": {
            "geometry_distance_mean": float(np.mean(geometry_distances)),
            "gate_distance_mean": float(np.mean(gate_distances)),
            "gate_geometry_pearson": pearson(geometry_distances, gate_distances),
            "gate_geometry_spearman": spearman(geometry_distances, gate_distances),
        },
        "blocks": blocks,
        "controlled_perturbation": controlled,
        "checks": {
            "checkpoint_projection_weights_nonzero": bool(max_projection_norm > 1e-8),
            "same_input_deterministic": bool(controlled["repeat_gate_max_abs_diff"] < 1e-8),
            "gate_non_uniform": bool(max(gate_abs_from_1, temp_abs_from_1) > 1e-6),
            "different_geometry_changes_gate": bool(float(np.mean(gate_distances)) > 1e-6),
            "controlled_perturbation_changes_gate": bool(
                controlled["perturbed_gate_mean_l2_diff"] > 1e-6
            ),
        },
    }
    summary["conclusion"] = classify(summary)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "patch_metrics.csv", patch_rows)
    write_report(out_dir / "report.md", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
