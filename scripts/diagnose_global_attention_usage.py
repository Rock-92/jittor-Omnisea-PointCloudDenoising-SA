import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_global_token_similarity import (
    choose_patch_groups,
    load_model,
    render_patch_contact_sheet,
    sample_patch,
    strip_arrays,
    write_csv,
)
from src.model.feature import (
    apply_point_linear,
    gather_neighbors,
    get_knn_idx,
)


def tensor_norm(x):
    return jt.sqrt((x * x).sum(dim=-1) + 1e-12)


def numpy_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def execute_block_with_global_attention_stats(block, feat, xyz, graph_knn_idx, global_token):
    x_norm = block.attn_norm(feat)
    q = apply_point_linear(block.q_proj, x_norm)
    k = apply_point_linear(block.k_proj, x_norm)
    v = apply_point_linear(block.v_proj, x_norm)

    B, N, _ = feat.shape
    global_norm = block.global_norm(global_token)
    k_global = apply_point_linear(block.global_k_proj, global_norm)
    v_global = apply_point_linear(block.global_v_proj, global_norm)

    scale_outputs = []
    stat_rows = []
    for scale_k in block.knn_scales:
        idx = graph_knn_idx[:, :, :scale_k]
        k_neighbors = gather_neighbors(k, idx)
        v_neighbors = gather_neighbors(v, idx)
        xyz_neighbors = gather_neighbors(xyz, idx)
        rel_pos = xyz_neighbors - xyz.unsqueeze(2)

        local_logits = (q.unsqueeze(2) * k_neighbors).sum(dim=-1) * block.scale
        local_logits = local_logits + block.rel_pos_bias(rel_pos)

        k_global_scale = k_global.unsqueeze(1).broadcast((B, N, 1, block.dim))
        v_global_scale = v_global.unsqueeze(1).broadcast((B, N, 1, block.dim))
        global_logits = (
            (q.unsqueeze(2) * k_global_scale).sum(dim=-1) * block.scale
            + block.global_attn_bias.reshape(1, 1, 1).broadcast((B, N, 1))
        )

        attn_logits = jt.concat([local_logits, global_logits], dim=2)
        attn = nn.softmax(attn_logits, dim=-1)
        local_attn = attn[:, :, :scale_k]
        global_attn = attn[:, :, -1:]

        local_out = (local_attn.unsqueeze(-1) * v_neighbors).sum(dim=2)
        global_out = (global_attn.unsqueeze(-1) * v_global_scale).sum(dim=2)
        scale_out = local_out + global_out
        scale_outputs.append(scale_out)

        attn_np = attn.detach().numpy()[0]
        global_weight = attn_np[:, -1]
        local_max = attn_np[:, :scale_k].max(axis=1)
        uniform = 1.0 / float(scale_k + 1)

        global_norm_np = tensor_norm(global_out).detach().numpy()[0]
        local_norm_np = tensor_norm(local_out).detach().numpy()[0]
        total_norm_np = tensor_norm(scale_out).detach().numpy()[0]
        contribution_ratio = global_norm_np / np.maximum(total_norm_np, 1e-12)

        gw_stats = numpy_stats(global_weight)
        cr_stats = numpy_stats(contribution_ratio)
        stat_rows.append({
            "scale_k": int(scale_k),
            "uniform_weight": uniform,
            "global_weight_mean": gw_stats["mean"],
            "global_weight_median": gw_stats["median"],
            "global_weight_p90": gw_stats["p90"],
            "global_weight_p99": gw_stats["p99"],
            "global_weight_max": gw_stats["max"],
            "global_over_uniform_rate": float(np.mean(global_weight > uniform)),
            "global_over_2x_uniform_rate": float(np.mean(global_weight > 2.0 * uniform)),
            "global_top_attention_rate": float(np.mean(global_weight >= local_max)),
            "global_contrib_ratio_mean": cr_stats["mean"],
            "global_contrib_ratio_median": cr_stats["median"],
            "global_contrib_ratio_p90": cr_stats["p90"],
            "global_contrib_ratio_p99": cr_stats["p99"],
            "global_contrib_ratio_max": cr_stats["max"],
            "local_out_norm_mean": float(np.mean(local_norm_np)),
            "global_out_norm_mean": float(np.mean(global_norm_np)),
            "total_out_norm_mean": float(np.mean(total_norm_np)),
        })

    out = jt.concat(scale_outputs, dim=-1)
    out = apply_point_linear(block.out_proj, out)
    feat = feat + out

    ffn = block.ffn_norm(feat)
    ffn = apply_point_linear(block.ffn_lin_1, ffn)
    ffn = block.act(ffn)
    ffn = apply_point_linear(block.ffn_lin_2, ffn)
    return feat + ffn, stat_rows


def extract_attention_usage(model, patch_noisy):
    encoder = model.encoder
    x = jt.array(patch_noisy[None, :, :])
    with jt.no_grad():
        graph_knn_idx = get_knn_idx(x, x, encoder.max_knn, offset=1)
        reuse_knn_idx = None

        feat = apply_point_linear(encoder.input_proj_1, x)
        feat = encoder.act(feat)
        feat = apply_point_linear(encoder.input_proj_2, feat)
        feat = encoder.act(feat)
        global_token = encoder.global_token_generator(feat)

        rows = []
        for block_idx, block in enumerate(encoder.blocks, start=1):
            if block_idx == 1:
                block_knn_idx = graph_knn_idx
                knn_source = "xyz"
            elif block_idx == 2:
                reuse_knn_idx = get_knn_idx(feat, feat, encoder.max_knn, offset=1)
                block_knn_idx = reuse_knn_idx
                knn_source = "feature_recomputed"
            else:
                block_knn_idx = reuse_knn_idx
                knn_source = "feature_reused"

            feat, stat_rows = execute_block_with_global_attention_stats(
                block,
                feat,
                x,
                block_knn_idx,
                global_token,
            )
            for row in stat_rows:
                rows.append({
                    "block": block_idx,
                    "knn_source": knn_source,
                    **row,
                })
    return rows


def grouped_summary(rows, keys):
    grouped = {}
    metrics = [
        "global_weight_mean",
        "global_weight_median",
        "global_over_uniform_rate",
        "global_over_2x_uniform_rate",
        "global_top_attention_rate",
        "global_contrib_ratio_mean",
        "global_contrib_ratio_median",
    ]
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        result = {k: v for k, v in zip(keys, key)}
        result["count"] = len(items)
        for metric in metrics:
            result[metric] = float(np.mean([float(item[metric]) for item in items]))
        out.append(result)
    return out


def render_usage_plot(summary_rows, out_path):
    groups = sorted(set(row["group"] for row in summary_rows))
    blocks = sorted(set(int(row["block"]) for row in summary_rows))
    scales = sorted(set(int(row["scale_k"]) for row in summary_rows))

    fig, axes = plt.subplots(len(scales), 1, figsize=(9.0, 3.2 * len(scales)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    x = np.arange(len(blocks))
    width = 0.24
    colors = {
        "sharp": "#d62728",
        "straight": "#4e79a7",
        "smooth": "#59a14f",
    }
    for ax, scale in zip(axes, scales):
        for offset_idx, group in enumerate(groups):
            vals = []
            for block in blocks:
                match = [
                    row
                    for row in summary_rows
                    if row["group"] == group
                    and int(row["block"]) == block
                    and int(row["scale_k"]) == scale
                ]
                vals.append(match[0]["global_weight_mean"] if match else math.nan)
            ax.bar(
                x + (offset_idx - (len(groups) - 1) / 2.0) * width,
                vals,
                width=width,
                label=group,
                color=colors.get(group),
            )
        ax.axhline(1.0 / (scale + 1), color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(f"K={scale} global attention weight mean")
        ax.set_ylabel("weight")
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"block {b}" for b in blocks])
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs2.0/checkpoints/vm/checkpoint_best.pkl")
    parser.add_argument("--run-config", default="outputs2.0/runs/train/20260526_071257/config.json")
    parser.add_argument("--mesh-root", default="E:/Code/competition2_EdgeConv/dataset_clean")
    parser.add_argument("--datalist", default="datalist/train.txt")
    parser.add_argument("--out-dir", default="outputs2.0/patch_diagnostics/global_attention_usage")
    parser.add_argument("--candidates", type=int, default=120)
    parser.add_argument("--select-per-group", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--surface-samples", type=int, default=32768)
    parser.add_argument("--noise-std-min", type=float, default=0.005)
    parser.add_argument("--noise-std-max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--use-cuda", type=int, default=1)
    args = parser.parse_args()

    jt.flags.use_cuda = int(args.use_cuda)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_root = Path(args.mesh_root)
    rel_paths = [
        line.strip()
        for line in (PROJECT_ROOT / args.datalist).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    chosen = [rel_paths[int(rng.integers(0, len(rel_paths)))] for _ in range(args.candidates)]

    candidates = []
    for i, rel_path in enumerate(chosen, 1):
        print(f"scan [{i}/{len(chosen)}] {rel_path}", flush=True)
        try:
            candidates.append(sample_patch(rel_path, mesh_root, rng, args, i))
        except Exception as exc:
            print(f"skip {rel_path}: {exc}", flush=True)

    selected = choose_patch_groups(candidates, args.select_per_group)
    write_csv([strip_arrays(row) for row in selected], out_dir / "selected_patches.csv")
    render_patch_contact_sheet(selected, out_dir / "selected_patch_contact_sheet.png")

    model = load_model(PROJECT_ROOT / args.checkpoint, PROJECT_ROOT / args.run_config)
    rows = []
    for i, item in enumerate(selected, 1):
        print(f"attention [{i}/{len(selected)}] {item['group']} {item['rel_path']}", flush=True)
        for stat in extract_attention_usage(model, item["patch_noisy"]):
            rows.append({
                "patch_index": i - 1,
                "group": item["group"],
                "rel_path": item["rel_path"],
                "category": item["category"],
                "point_normal_var": item["point_normal_var"],
                "mesh_normal_var": item["mesh_normal_var"],
                "linearity": item["linearity"],
                "point_surface_var": item["point_surface_var"],
                **stat,
            })

    write_csv(rows, out_dir / "attention_usage_by_patch.csv")
    group_block_scale = grouped_summary(rows, ["group", "block", "scale_k"])
    block_scale = grouped_summary(rows, ["block", "scale_k"])
    write_csv(group_block_scale, out_dir / "attention_usage_by_group_block_scale.csv")
    write_csv(block_scale, out_dir / "attention_usage_by_block_scale.csv")
    render_usage_plot(group_block_scale, out_dir / "global_attention_weight_by_group.png")

    summary = {
        "checkpoint": str((PROJECT_ROOT / args.checkpoint).resolve()),
        "run_config": str((PROJECT_ROOT / args.run_config).resolve()),
        "mesh_root": str(mesh_root.resolve()),
        "datalist": str((PROJECT_ROOT / args.datalist).resolve()),
        "candidates_scanned": len(candidates),
        "selected_count": len(selected),
        "group_counts": {
            group: int(sum(1 for item in selected if item["group"] == group))
            for group in sorted(set(item["group"] for item in selected))
        },
        "patch_size": args.patch_size,
        "seed": args.seed,
        "notes": {
            "uniform_weight": "For K local neighbors plus 1 global token, uniform is 1/(K+1).",
            "global_top_attention_rate": "Fraction of points where global token has at least the largest local-neighbor attention.",
            "global_contrib_ratio": "Norm(global_attn * V_global) / norm(local_out + global_out), averaged over points.",
        },
        "by_block_scale": block_scale,
        "by_group_block_scale": group_block_scale,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
