import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jittor as jt
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.hard_patch_common import (  # noqa: E402
    displacement_metrics,
    load_hard_patch_npz,
    load_model,
    quantile_summary,
    score_prediction,
    write_json,
)


DEFAULT_INITIAL = (
    PROJECT_ROOT
    / "outputs_result/outputs_EdgeConvBrancg/checkpoints/vm_ssl/checkpoint_best.pkl"
)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs_result/outputs_analysis/hard_patch_overfit/hard_patches.npz"
)
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "outputs_result/outputs_analysis/hard_patch_overfit/train"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_result/outputs_analysis/hard_patch_overfit/diagnosis"


def predict(model, pc_noisy_np, sigma_np, batch_size, mode):
    preds = []
    model.eval()
    with jt.no_grad():
        for start in range(0, pc_noisy_np.shape[0], batch_size):
            end = min(start + batch_size, pc_noisy_np.shape[0])
            pc_noisy = jt.array(pc_noisy_np[start:end])
            sigma = jt.array(sigma_np[start:end])
            if mode == "heun":
                pred, _ = model.denoise_langevin_dynamics(pc_noisy)
            elif mode == "fixed":
                pred = model.predict_clean(pc_noisy, sigma=sigma)
            else:
                raise ValueError(f"unsupported mode: {mode}")
            preds.append(pred.detach().numpy().astype(np.float32, copy=False))
    return np.concatenate(preds, axis=0)


def evaluate_checkpoint(name, checkpoint, data, batch_size, mode):
    model = load_model(checkpoint)
    pred = predict(model, data["pc_noisy"], data["score_sigma"], batch_size, mode)
    rows = []
    for i in range(data["pc_noisy"].shape[0]):
        row = {
            "index": i,
            "checkpoint_name": name,
            "rel_path": str(data["rel_path"][i]),
            "seed_idx": int(data["seed_idx"][i]),
            "patch_scale": float(data["patch_scale"][i]),
            "geometry_category": str(data["geometry_category"][i]),
            **score_prediction(data["pc_noisy"][i], data["pc_clean"][i], pred[i]),
            **displacement_metrics(data["pc_noisy"][i], data["pc_clean"][i], pred[i]),
        }
        rows.append(row)
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "score": quantile_summary([row["cd_score"] for row in rows]),
        "length_ratio": quantile_summary([row["length_ratio_mean"] for row in rows]),
        "cosine": quantile_summary([row["cosine_mean"] for row in rows]),
        "under_length_rate": quantile_summary([row["under_length_rate"] for row in rows]),
    }
    return rows, summary


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def join_deltas(initial_rows, other_rows, other_name):
    by_idx = {row["index"]: row for row in other_rows}
    rows = []
    for base in initial_rows:
        other = by_idx[base["index"]]
        rows.append(
            {
                "index": base["index"],
                "rel_path": base["rel_path"],
                "geometry_category": base["geometry_category"],
                "patch_scale": base["patch_scale"],
                "initial_score": base["cd_score"],
                f"{other_name}_score": other["cd_score"],
                f"{other_name}_score_gain": other["cd_score"] - base["cd_score"],
                "initial_length_ratio": base["length_ratio_mean"],
                f"{other_name}_length_ratio": other["length_ratio_mean"],
                f"{other_name}_length_ratio_gain": other["length_ratio_mean"] - base["length_ratio_mean"],
                "initial_cosine": base["cosine_mean"],
                f"{other_name}_cosine": other["cosine_mean"],
                f"{other_name}_cosine_gain": other["cosine_mean"] - base["cosine_mean"],
                "initial_under_length_rate": base["under_length_rate"],
                f"{other_name}_under_length_rate": other["under_length_rate"],
            }
        )
    return rows


def category_summary(delta_rows, score_key):
    grouped = defaultdict(list)
    for row in delta_rows:
        grouped[row["geometry_category"]].append(row)
    out = []
    for category, rows in sorted(grouped.items()):
        out.append(
            {
                "geometry_category": category,
                "count": len(rows),
                "initial_score_mean": float(np.mean([row["initial_score"] for row in rows])),
                "score_gain_mean": float(np.mean([row[score_key] for row in rows])),
                "initial_length_ratio_mean": float(np.mean([row["initial_length_ratio"] for row in rows])),
            }
        )
    return out


def render_plots(summary, deltas, out_dir):
    out_dir = Path(out_dir)
    gains = [row["best_score_gain"] for row in deltas]
    init_ratio = [row["initial_length_ratio"] for row in deltas]
    best_ratio = [row["best_length_ratio"] for row in deltas]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.hist(gains, bins=18, color="#4e79a7", alpha=0.85)
    ax.axvline(np.mean(gains), color="#222222", linestyle="--", label=f"mean {np.mean(gains):.2f}")
    ax.set_xlabel("Best overfit score gain")
    ax.set_ylabel("Patch count")
    ax.set_title("Hard patch overfit score gains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_gain_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(init_ratio, best_ratio, s=18, alpha=0.75)
    lo = min(min(init_ratio), min(best_ratio))
    hi = max(max(init_ratio), max(best_ratio))
    ax.plot([lo, hi], [lo, hi], color="#222222", linestyle="--", linewidth=1)
    ax.set_xlabel("Initial length ratio")
    ax.set_ylabel("Best overfit length ratio")
    ax.set_title("Displacement length ratio before/after overfit")
    fig.tight_layout()
    fig.savefig(out_dir / "length_ratio_before_after.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--initial-checkpoint", default=str(DEFAULT_INITIAL))
    parser.add_argument("--best-checkpoint", default=str(DEFAULT_TRAIN_DIR / "checkpoint_best.pkl"))
    parser.add_argument("--last-checkpoint", default=str(DEFAULT_TRAIN_DIR / "checkpoint_last.pkl"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=["fixed", "heun"], default="fixed")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_hard_patch_npz(args.dataset)

    checkpoints = [
        ("initial", args.initial_checkpoint),
        ("best", args.best_checkpoint),
        ("last", args.last_checkpoint),
    ]
    all_rows = []
    summaries = {}
    rows_by_name = {}
    for name, ckpt in checkpoints:
        if not Path(ckpt).exists():
            print(f"Skip missing checkpoint: {ckpt}", flush=True)
            continue
        rows, summary = evaluate_checkpoint(name, ckpt, data, args.batch_size, args.mode)
        rows_by_name[name] = rows
        summaries[name] = summary
        all_rows.extend(rows)
        write_csv(out_dir / f"{name}_eval.csv", rows)
        write_json(out_dir / f"{name}_summary.json", summary)
        print(name, summary, flush=True)

    if "initial" in rows_by_name and "best" in rows_by_name:
        best_delta = join_deltas(rows_by_name["initial"], rows_by_name["best"], "best")
        write_csv(out_dir / "best_delta.csv", best_delta)
    else:
        best_delta = []
    if "initial" in rows_by_name and "last" in rows_by_name:
        last_delta = join_deltas(rows_by_name["initial"], rows_by_name["last"], "last")
        write_csv(out_dir / "last_delta.csv", last_delta)
    else:
        last_delta = []

    if best_delta:
        best_score_gain = [row["best_score_gain"] for row in best_delta]
        best_ratio_gain = [row["best_length_ratio_gain"] for row in best_delta]
        best_cosine_gain = [row["best_cosine_gain"] for row in best_delta]
        conclusion = "CAN_OVERFIT_HARD_PATCHES"
        if np.mean(best_score_gain) < 5.0:
            conclusion = "HARD_PATCH_OVERFIT_WEAK"
        elif np.mean(best_ratio_gain) > 0.05 and np.mean(best_cosine_gain) > -0.02:
            conclusion = "CAN_OVERFIT_AND_FIX_UNDER_LENGTH"
        final_summary = {
            "mode": args.mode,
            "dataset": str(Path(args.dataset).resolve()),
            "conclusion": conclusion,
            "checkpoint_summaries": summaries,
            "best_score_gain": quantile_summary(best_score_gain),
            "best_length_ratio_gain": quantile_summary(best_ratio_gain),
            "best_cosine_gain": quantile_summary(best_cosine_gain),
            "best_category_summary": category_summary(best_delta, "best_score_gain"),
            "initial_category_counts": dict(Counter(row["geometry_category"] for row in best_delta)),
        }
        render_plots(final_summary, best_delta, out_dir)
    else:
        final_summary = {
            "mode": args.mode,
            "dataset": str(Path(args.dataset).resolve()),
            "conclusion": "MISSING_INITIAL_OR_BEST_CHECKPOINT",
            "checkpoint_summaries": summaries,
        }

    write_csv(out_dir / "all_eval_rows.csv", all_rows)
    write_json(out_dir / "diagnosis_summary.json", final_summary)
    print(final_summary, flush=True)
    print(f"Wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
