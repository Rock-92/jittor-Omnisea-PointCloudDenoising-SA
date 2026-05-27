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
    apply_global_modulation,
    apply_point_linear,
    apply_residual_gate,
    gather_neighbors,
    get_knn_idx,
    knn_dot,
    knn_weighted_sum,
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


def modulation_stats(values, prefix):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(values))),
        f"{prefix}_p90_abs": float(np.quantile(np.abs(values), 0.90)),
        f"{prefix}_max_abs": float(np.max(np.abs(values))),
    }


def execute_block_with_global_attention_stats(block, feat, xyz, graph_knn_idx, global_token):
    x_norm = block.attn_norm(feat)
    (
        gamma_attn,
        beta_attn,
        gate_attn,
        gamma_ffn,
        beta_ffn,
        gate_ffn,
    ) = block.global_conditioner(global_token)
    x_mod = apply_global_modulation(x_norm, gamma_attn, beta_attn)
    attn_mod_delta = tensor_norm(x_mod - x_norm) / (tensor_norm(x_norm) + 1e-12)

    gamma_attn_np = gamma_attn.detach().numpy()
    beta_attn_np = beta_attn.detach().numpy()
    gate_attn_np = gate_attn.detach().numpy()
    gamma_ffn_np = gamma_ffn.detach().numpy()
    beta_ffn_np = beta_ffn.detach().numpy()
    gate_ffn_np = gate_ffn.detach().numpy()
    attn_mod_delta_np = attn_mod_delta.detach().numpy()
    common_stats = {
        **modulation_stats(gamma_attn_np, "gamma_attn"),
        **modulation_stats(beta_attn_np, "beta_attn"),
        **modulation_stats(gate_attn_np, "gate_attn"),
        **modulation_stats(gamma_ffn_np, "gamma_ffn"),
        **modulation_stats(beta_ffn_np, "beta_ffn"),
        **modulation_stats(gate_ffn_np, "gate_ffn"),
        "attn_mod_delta_ratio_mean": float(np.mean(attn_mod_delta_np)),
        "attn_mod_delta_ratio_p90": float(np.quantile(attn_mod_delta_np, 0.90)),
    }

    q = apply_point_linear(block.q_proj, x_mod)
    k = apply_point_linear(block.k_proj, x_mod)
    v = apply_point_linear(block.v_proj, x_mod)

    scale_outputs = []
    stat_rows = []
    for scale_k in block.knn_scales:
        idx = graph_knn_idx[:, :, :scale_k]
        k_neighbors = gather_neighbors(k, idx)
        v_neighbors = gather_neighbors(v, idx)
        xyz_neighbors = gather_neighbors(xyz, idx)
        rel_pos = xyz_neighbors - xyz.unsqueeze(2)

        local_logits = knn_dot(q, k_neighbors, block.scale)
        local_logits = local_logits + block.rel_pos_bias(rel_pos)
        attn = nn.softmax(local_logits, dim=-1)
        scale_out = knn_weighted_sum(attn, v_neighbors)
        scale_outputs.append(scale_out)

        attn_np = attn.detach().numpy()[0]
        entropy = -(attn_np * np.log(np.maximum(attn_np, 1e-12))).sum(axis=1)
        uniform_entropy = math.log(float(scale_k))

        scale_norm_np = tensor_norm(scale_out).detach().numpy()[0]
        gated_scale_out = apply_residual_gate(scale_out, gate_attn)
        gated_norm_np = tensor_norm(gated_scale_out).detach().numpy()[0]
        total_norm_np = tensor_norm(scale_out).detach().numpy()[0]
        gate_ratio = gated_norm_np / np.maximum(total_norm_np, 1e-12)
        entropy_stats = numpy_stats(entropy / max(uniform_entropy, 1e-12))
        gate_ratio_stats = numpy_stats(gate_ratio)
        stat_rows.append({
            "scale_k": int(scale_k),
            **common_stats,
            "local_attention_entropy_ratio_mean": entropy_stats["mean"],
            "local_attention_entropy_ratio_median": entropy_stats["median"],
            "local_attention_entropy_ratio_p90": entropy_stats["p90"],
            "attn_gate_norm_ratio_mean": gate_ratio_stats["mean"],
            "attn_gate_norm_ratio_median": gate_ratio_stats["median"],
            "attn_gate_norm_ratio_p90": gate_ratio_stats["p90"],
            "local_out_norm_mean": float(np.mean(scale_norm_np)),
            "total_out_norm_mean": float(np.mean(total_norm_np)),
        })

    out = jt.concat(scale_outputs, dim=-1)
    out = apply_point_linear(block.out_proj, out)
    out = apply_residual_gate(out, gate_attn)
    feat = feat + out

    ffn = block.ffn_norm(feat)
    ffn = apply_global_modulation(ffn, gamma_ffn, beta_ffn)
    ffn = apply_point_linear(block.ffn_lin_1, ffn)
    ffn = block.act(ffn)
    ffn = apply_point_linear(block.ffn_lin_2, ffn)
    ffn = apply_residual_gate(ffn, gate_ffn)
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
        "gamma_attn_abs_mean",
        "beta_attn_abs_mean",
        "gate_attn_abs_mean",
        "gamma_ffn_abs_mean",
        "beta_ffn_abs_mean",
        "gate_ffn_abs_mean",
        "attn_mod_delta_ratio_mean",
        "attn_gate_norm_ratio_mean",
        "local_attention_entropy_ratio_mean",
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
                vals.append(match[0]["attn_mod_delta_ratio_mean"] if match else math.nan)
            ax.bar(
                x + (offset_idx - (len(groups) - 1) / 2.0) * width,
                vals,
                width=width,
                label=group,
                color=colors.get(group),
            )
        ax.set_title(f"K={scale} global modulation delta ratio")
        ax.set_ylabel("ratio")
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
            "attn_mod_delta_ratio": "Norm of the global-conditioned attention input change divided by the original normalized feature norm.",
            "attn_gate_norm_ratio": "Norm after the global-conditioned attention residual gate divided by the ungated local attention norm.",
            "local_attention_entropy_ratio": "Local KNN attention entropy divided by log(K); lower means more concentrated local attention.",
        },
        "by_block_scale": block_scale,
        "by_group_block_scale": group_block_scale,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
