import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import sample_vertex_groups


def relu(x):
    return np.maximum(x, 0.0)


def linear(x, weight, bias):
    return x @ weight.T + bias


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight.reshape(1, 1, -1) + bias.reshape(1, 1, -1)


def token_block(state, prefix, x):
    dim = x.shape[-1]
    q = linear(x, state[f"{prefix}.q_proj.weight"], state[f"{prefix}.q_proj.bias"])
    k = linear(x, state[f"{prefix}.k_proj.weight"], state[f"{prefix}.k_proj.bias"])
    v = linear(x, state[f"{prefix}.v_proj.weight"], state[f"{prefix}.v_proj.bias"])
    logits = (q @ np.swapaxes(k, 1, 2)) * (dim ** -0.5)
    logits = logits - logits.max(axis=-1, keepdims=True)
    attn = np.exp(logits)
    attn = attn / np.maximum(attn.sum(axis=-1, keepdims=True), 1e-8)
    out = attn @ v
    out = linear(out, state[f"{prefix}.out_proj.weight"], state[f"{prefix}.out_proj.bias"])
    x = layer_norm(
        x + out,
        state[f"{prefix}.attn_norm.weight"],
        state[f"{prefix}.attn_norm.bias"],
    )

    ffn = linear(x, state[f"{prefix}.ffn_lin_1.weight"], state[f"{prefix}.ffn_lin_1.bias"])
    ffn = relu(ffn)
    ffn = linear(ffn, state[f"{prefix}.ffn_lin_2.weight"], state[f"{prefix}.ffn_lin_2.bias"])
    return layer_norm(
        x + ffn,
        state[f"{prefix}.ffn_norm.weight"],
        state[f"{prefix}.ffn_norm.bias"],
    )


def encode_global_token_numpy(state, pc):
    x = linear(pc, state["encoder.input_proj_1.weight"], state["encoder.input_proj_1.bias"])
    x = relu(x)
    x = linear(x, state["encoder.input_proj_2.weight"], state["encoder.input_proj_2.bias"])
    x = relu(x)
    b, _, c = x.shape
    global_token = np.broadcast_to(
        state["encoder.global_token_generator.global_token"],
        (b, 1, c),
    ).copy()
    tokens = np.concatenate([global_token, x], axis=1)
    block_idx = 0
    while f"encoder.global_token_generator.block_{block_idx}.q_proj.weight" in state:
        tokens = token_block(
            state,
            f"encoder.global_token_generator.block_{block_idx}",
            tokens,
        )
        block_idx += 1
    return tokens[:, 0, :].astype(np.float32, copy=False)


def normalize_pc(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1).max()).max()
    return (pc / scale).astype(np.float32, copy=False)


def orientation_variation(normals):
    if normals.shape[0] == 0:
        return 0.0
    tensor = np.zeros((3, 3), dtype=np.float64)
    for normal in normals:
        n = normal.astype(np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        tensor += np.outer(n, n)
    tensor /= max(normals.shape[0], 1)
    eigvals = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    return float(1.0 - eigvals[0])


def estimate_point_sharpness(points, k=24, max_points=96, rng=None):
    if points.shape[0] <= k + 2:
        return 0.0, 0.0
    if rng is None:
        rng = np.random.default_rng()
    if points.shape[0] > max_points:
        sample_idx = rng.choice(points.shape[0], size=max_points, replace=False)
    else:
        sample_idx = np.arange(points.shape[0])
    tree = cKDTree(points)
    normals = []
    surface_vars = []
    for idx in sample_idx:
        _, nn_idx = tree.query(points[idx], k=min(k, points.shape[0]))
        neigh = points[nn_idx]
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)
        total = float(eigvals.sum())
        if total > 1e-12:
            surface_vars.append(float(eigvals[0] / total))
        normals.append(eigvecs[:, 0])
    normal_var = orientation_variation(np.asarray(normals))
    surface_var = float(np.mean(surface_vars)) if surface_vars else 0.0
    return normal_var, surface_var


def pca_geometry(points):
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0))[::-1]
    l1, l2, l3 = eigvals
    if l1 <= 1e-12:
        return 0.0, 0.0
    linearity = float((l1 - l2) / l1)
    planarity = float((l2 - l3) / l1)
    return linearity, planarity


def geometry_targets(pc_clean, rng):
    targets = []
    for patch in pc_clean:
        point_normal_var, point_surface_var = estimate_point_sharpness(
            patch,
            rng=rng,
        )
        linearity, planarity = pca_geometry(patch)
        targets.append(
            [
                np.clip(point_normal_var, 0.0, 1.0),
                np.clip(linearity, 0.0, 1.0),
                np.clip(planarity, 0.0, 1.0),
                np.clip(point_surface_var, 0.0, 1.0),
            ]
        )
    return np.asarray(targets, dtype=np.float32)


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


def sample_patches(mesh_root, rel_path, rng, num_samples, patch_size, patches_per_mesh):
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
        num_vertex_samples=min(1024, max(1, num_samples // 16)),
    )
    clean = normalize_pc(clean.astype(np.float32, copy=False))
    tree = cKDTree(clean)
    patches = []
    seeds = rng.choice(clean.shape[0], size=patches_per_mesh, replace=False)
    for seed_idx in seeds:
        _, nn_idx = tree.query(clean[seed_idx][None, :], k=patch_size)
        seed = clean[seed_idx][None, :]
        patches.append((clean[nn_idx[0]] - seed).astype(np.float32, copy=False))
    return patches


def l2_normalize(x, eps=1e-8):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


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


def pairwise_geom_dist(geom):
    geom_norm = (geom - geom.mean(axis=0, keepdims=True)) / np.maximum(
        geom.std(axis=0, keepdims=True),
        1e-6,
    )
    return np.sqrt(((geom_norm[:, None, :] - geom_norm[None, :, :]) ** 2).sum(axis=-1))


def pca_2d(x):
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_svg(path, coords, geom, rows):
    width, height = 980, 720
    pad = 64
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    sx = lambda v: pad + (v - x_min) / max(x_max - x_min, 1e-8) * (width - 2 * pad)
    sy = lambda v: height - pad - (v - y_min) / max(y_max - y_min, 1e-8) * (height - 2 * pad)
    color_value = geom[:, 0]
    c_min, c_max = float(color_value.min()), float(color_value.max())

    def color(v):
        t = (float(v) - c_min) / max(c_max - c_min, 1e-8)
        r = int(49 + t * (214 - 49))
        g = int(130 + t * (39 - 130))
        b = int(189 + t * (40 - 189))
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif; fill:#222}.title{font-size:22px;font-weight:700}.small{font-size:11px}</style>',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">Global Encoder Token PCA, colored by normal variation</text>',
        f'<line x1="{pad}" x2="{width-pad}" y1="{height-pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" x2="{pad}" y1="{pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    for i, row in enumerate(rows):
        parts.append(
            f'<circle cx="{sx(coords[i,0]):.1f}" cy="{sy(coords[i,1]):.1f}" r="5" '
            f'fill="{color(geom[i,0])}" fill-opacity="0.82"><title>{row["patch"]} {row["mesh"]} normal={geom[i,0]:.3f}</title></circle>'
        )
    parts.append(f'<text x="{pad}" y="{height-22}" class="small">PCA-1</text>')
    parts.append(f'<text x="{width-130}" y="{height-22}" class="small">red = high normal variation</text>')
    parts.append('</svg>')
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--datalist", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-meshes", type=int, default=30)
    parser.add_argument("--patches-per-mesh", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with Path(args.ckpt).open("rb") as f:
        state = pickle.load(f)

    rels = read_datalist(args.datalist)
    rng.shuffle(rels)
    patches = []
    patch_rows = []
    mesh_count = 0
    for rel in rels:
        if mesh_count >= args.num_meshes:
            break
        mesh_path = Path(args.mesh_root) / rel / "models" / "model_normalized.obj"
        if not mesh_path.exists():
            continue
        try:
            new_patches = sample_patches(
                args.mesh_root,
                rel,
                rng,
                args.num_samples,
                args.patch_size,
                args.patches_per_mesh,
            )
        except Exception as exc:
            print(f"skip {rel}: {exc}", flush=True)
            continue
        mesh_count += 1
        for mesh_patch, patch in enumerate(new_patches):
            patches.append(patch)
            patch_rows.append(
                {
                    "patch": len(patches) - 1,
                    "mesh": rel,
                    "mesh_patch": mesh_patch,
                }
            )

    pc = np.stack(patches, axis=0)
    geom = geometry_targets(pc, np.random.default_rng(args.seed + 1)).astype(np.float32)
    tokens = l2_normalize(encode_global_token_numpy(state, pc))
    sim = tokens @ tokens.T
    geom_dist = pairwise_geom_dist(geom)
    n = len(tokens)
    offdiag = ~np.eye(n, dtype=bool)

    nearest_rows = []
    same_mesh_hits = []
    nearest_geom = []
    random_geom = geom_dist[offdiag]
    for i in range(n):
        order = np.argsort(-sim[i])
        nn = [idx for idx in order if idx != i][0]
        same_mesh = patch_rows[i]["mesh"] == patch_rows[nn]["mesh"]
        same_mesh_hits.append(float(same_mesh))
        nearest_geom.append(float(geom_dist[i, nn]))
        nearest_rows.append(
            {
                "patch": i,
                "mesh": patch_rows[i]["mesh"],
                "nearest_patch": int(nn),
                "nearest_mesh": patch_rows[nn]["mesh"],
                "same_mesh": int(same_mesh),
                "token_cos": float(sim[i, nn]),
                "geometry_dist": float(geom_dist[i, nn]),
                "normal_var": float(geom[i, 0]),
                "linearity": float(geom[i, 1]),
                "planarity": float(geom[i, 2]),
                "surface_var": float(geom[i, 3]),
            }
        )

    token_sims = []
    geom_dists = []
    same_mesh_sims = []
    diff_mesh_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            token_sims.append(float(sim[i, j]))
            geom_dists.append(float(geom_dist[i, j]))
            if patch_rows[i]["mesh"] == patch_rows[j]["mesh"]:
                same_mesh_sims.append(float(sim[i, j]))
            else:
                diff_mesh_sims.append(float(sim[i, j]))

    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "n_patches": n,
        "n_meshes": mesh_count,
        "token_dim_std_mean": float(tokens.std(axis=0).mean()),
        "token_pair_cos_mean": float(np.mean(token_sims)),
        "token_pair_cos_std": float(np.std(token_sims)),
        "token_geom_spearman": spearman(token_sims, -np.asarray(geom_dists)),
        "nearest_geom_dist_mean": float(np.mean(nearest_geom)),
        "random_geom_dist_mean": float(np.mean(random_geom)),
        "nearest_vs_random_ratio": float(np.mean(nearest_geom) / max(float(np.mean(random_geom)), 1e-8)),
        "same_mesh_nearest_rate": float(np.mean(same_mesh_hits)),
        "same_mesh_pair_cos_mean": float(np.mean(same_mesh_sims)) if same_mesh_sims else None,
        "diff_mesh_pair_cos_mean": float(np.mean(diff_mesh_sims)) if diff_mesh_sims else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "nearest_token_neighbors.csv", nearest_rows)
    patch_out = []
    for row, g in zip(patch_rows, geom):
        patch_out.append(
            {
                **row,
                "normal_var": float(g[0]),
                "linearity": float(g[1]),
                "planarity": float(g[2]),
                "surface_var": float(g[3]),
            }
        )
    write_csv(output_dir / "patches.csv", patch_out)
    coords = pca_2d(tokens)
    save_svg(output_dir / "token_pca.svg", coords, geom, patch_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
