import argparse
import csv
import json
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

from scripts.pretrain_global_encoder import encode_global_token, geometry_targets, make_view
from src.data.utils import sample_vertex_groups
from src.model.vm import VelocityModule


def normalize_pc(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc**2).sum(axis=1).max()).max()
    return (pc / scale).astype(np.float32, copy=False)


def read_datalist(paths):
    rels = []
    seen = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                rel = line.strip().replace("\\", "/")
                if rel and not rel.startswith("#") and rel not in seen:
                    rels.append(rel)
                    seen.add(rel)
    return rels


def sample_patch(mesh_root, rel_path, rng, num_samples, patch_size, noise_std):
    mesh_path = Path(mesh_root) / rel_path / "models" / "model_normalized.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    clean, _, _, _ = sample_vertex_groups(
        vertices=vertices,
        faces=faces,
        num_samples=num_samples,
        num_vertex_samples=min(1024, num_samples // 16),
    )
    clean = normalize_pc(clean.astype(np.float32, copy=False))
    noisy = clean + rng.laplace(0.0, noise_std, size=clean.shape).astype(np.float32)
    seed_idx = int(rng.integers(0, noisy.shape[0]))
    _, nn_idx = cKDTree(noisy).query(noisy[seed_idx][None, :], k=patch_size)
    nn_idx = nn_idx[0]
    seed = noisy[seed_idx][None, :]
    return (clean[nn_idx] - seed).astype(np.float32, copy=False)


def l2_normalize(x, eps=1e-8):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def encode_views(model, views, batch_size=8):
    outs = []
    with jt.no_grad():
        for start in range(0, views.shape[0], batch_size):
            token = encode_global_token(model, jt.array(views[start : start + batch_size]))
            outs.append(token.numpy().astype(np.float32, copy=False))
    return l2_normalize(np.concatenate(outs, axis=0))


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
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
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def summarize_tokens(name, tokens_clean, tokens_weak, tokens_a, tokens_b, geom):
    n = tokens_clean.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    same_strong = np.sum(tokens_a * tokens_b, axis=1)
    clean_weak = np.sum(tokens_clean * tokens_weak, axis=1)
    diff_strong = (tokens_a @ tokens_b.T)[offdiag]
    diff_clean = (tokens_clean @ tokens_clean.T)[offdiag]

    geom_norm = geom / np.maximum(geom.std(axis=0, keepdims=True), 1e-6)
    token_sims = []
    geom_dists = []
    sim_mat = tokens_clean @ tokens_clean.T
    geom_dist_mat = np.sqrt(((geom_norm[:, None, :] - geom_norm[None, :, :]) ** 2).sum(axis=-1))
    nearest_rows = []
    for i in range(n):
        order = np.argsort(-sim_mat[i])
        nn = [idx for idx in order if idx != i][0]
        nearest_rows.append(
            {
                "model": name,
                "patch": i,
                "nearest_patch": int(nn),
                "token_cos": float(sim_mat[i, nn]),
                "geometry_dist": float(geom_dist_mat[i, nn]),
            }
        )
        for j in range(i + 1, n):
            token_sims.append(float(sim_mat[i, j]))
            geom_dists.append(float(geom_dist_mat[i, j]))

    random_geom = geom_dist_mat[offdiag]
    nn_geom = np.asarray([r["geometry_dist"] for r in nearest_rows], dtype=np.float64)
    return (
        {
            "model": name,
            "n_patches": n,
            "same_strong_mean": float(same_strong.mean()),
            "clean_weak_mean": float(clean_weak.mean()),
            "diff_strong_mean": float(diff_strong.mean()),
            "diff_clean_mean": float(diff_clean.mean()),
            "same_minus_diff_gap": float(same_strong.mean() - diff_strong.mean()),
            "token_geom_spearman": spearman(token_sims, -np.asarray(geom_dists)),
            "nearest_geom_dist_mean": float(nn_geom.mean()),
            "random_geom_dist_mean": float(random_geom.mean()),
            "nearest_vs_random_ratio": float(nn_geom.mean() / max(random_geom.mean(), 1e-8)),
            "token_dim_std_mean": float(tokens_clean.std(axis=0).mean()),
        },
        nearest_rows,
    )


def build_model(model_config_path, transform_config_path, ckpt=None):
    model_cfg = OmegaConf.to_container(OmegaConf.load(model_config_path), resolve=True)
    model_cfg.pop("__target__", None)
    model_cfg["global_encoder_pretrain_ckpt"] = None
    transform_cfg = OmegaConf.to_container(OmegaConf.load(transform_config_path), resolve=True)
    model = VelocityModule(model_cfg, transform_cfg)
    if ckpt is not None:
        model.load_global_encoder_pretrain(str(ckpt))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_root", default=r"E:\Code\competition2_EdgeConv\dataset_clean")
    parser.add_argument(
        "--datalist",
        nargs="+",
        default=[
            r"E:\Code\competition2_EdgeConv\datalist\train.txt",
            r"E:\Code\competition2_EdgeConv\datalist\validate.txt",
        ],
    )
    parser.add_argument("--ckpt", default="pretrain/global_encoder/global_encoder_best.pkl")
    parser.add_argument("--num_meshes", type=int, default=10)
    parser.add_argument("--patches_per_mesh", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=8192)
    parser.add_argument("--patch_size", type=int, default=1000)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", default="pretrain/global_encoder/patch_eval")
    args = parser.parse_args()

    jt.flags.use_cuda = int(args.use_cuda)
    rng = np.random.default_rng(args.seed)
    rels = read_datalist(args.datalist)
    rng.shuffle(rels)

    patches = []
    patch_rows = []
    for rel in rels:
        if len({r["mesh"] for r in patch_rows}) >= args.num_meshes:
            break
        mesh_path = Path(args.mesh_root) / rel / "models" / "model_normalized.obj"
        if not mesh_path.exists():
            continue
        for patch_idx in range(args.patches_per_mesh):
            patch = sample_patch(
                args.mesh_root,
                rel,
                rng,
                args.num_samples,
                args.patch_size,
                noise_std=float(rng.uniform(0.005, 0.020)),
            )
            patches.append(patch)
            patch_rows.append({"patch": len(patches) - 1, "mesh": rel, "mesh_patch": patch_idx})

    pc_clean = np.stack(patches, axis=0)
    weak_cfg = {
        "min_keep_ratio": 1.0,
        "noise_std_min": 0.002,
        "noise_std_max": 0.008,
        "rotate_degrees": 2.0,
    }
    strong_cfg = {
        "min_keep_ratio": 1.0,
        "noise_std_min": 0.010,
        "noise_std_max": 0.030,
        "rotate_degrees": 5.0,
        "surface_jitter": True,
        "surface_jitter_knn": 12,
        "surface_jitter_alpha_min": 0.05,
        "surface_jitter_alpha_max": 0.25,
    }
    weak = make_view(pc_clean, weak_cfg, rng)
    strong_a = make_view(pc_clean, strong_cfg, rng)
    strong_b = make_view(pc_clean, strong_cfg, rng)
    geom = geometry_targets(pc_clean, np.random.default_rng(args.seed + 1)).astype(np.float32)

    pretrained = build_model("configs/model/vm.yaml", "configs/transform/vm.yaml", args.ckpt)
    random_model = build_model("configs/model/vm.yaml", "configs/transform/vm.yaml", None)

    summaries = []
    nearest = []
    for name, model in [("pretrained", pretrained), ("random_init", random_model)]:
        tokens_clean = encode_views(model, pc_clean)
        tokens_weak = encode_views(model, weak)
        tokens_a = encode_views(model, strong_a)
        tokens_b = encode_views(model, strong_b)
        summary, nearest_rows = summarize_tokens(
            name,
            tokens_clean,
            tokens_weak,
            tokens_a,
            tokens_b,
            geom,
        )
        summaries.append(summary)
        nearest.extend(nearest_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "nearest_token_neighbors.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["model", "patch", "nearest_patch", "token_cos", "geometry_dist"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nearest)
    with (output_dir / "patches.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["patch", "mesh", "mesh_patch", "normal_var", "linearity", "planarity", "surface_var"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, target in zip(patch_rows, geom):
            writer.writerow(
                {
                    **row,
                    "normal_var": float(target[0]),
                    "linearity": float(target[1]),
                    "planarity": float(target[2]),
                    "surface_var": float(target[3]),
                }
            )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
