import argparse
import csv
import sys
from pathlib import Path

import jittor as jt
import numpy as np


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


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs_result/outputs_EdgeConvBrancg/checkpoints/vm_ssl/checkpoint_best.pkl"
)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "outputs_result/outputs_analysis/hard_patch_overfit/hard_patches.npz"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_result/outputs_analysis/hard_patch_overfit/train"


def iter_batches(num_items, batch_size, rng, shuffle=True):
    indices = np.arange(num_items)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, num_items, batch_size):
        yield indices[start : start + batch_size]


def set_train_scope(model, scope):
    if scope == "all":
        return
    for p in model.parameters():
        p.stop_grad()
    if scope == "decoder":
        modules = [model.decoder]
    elif scope == "decoder_edge":
        modules = [model.decoder]
        if getattr(model, "use_edgeconv_branch", False):
            modules.append(model.edgeconv_branch)
    else:
        raise ValueError(f"unsupported train scope: {scope}")
    for module in modules:
        for p in module.parameters():
            p.start_grad()


def collect_train_parameters(model, scope):
    if scope == "all":
        return model.parameters()
    if scope == "decoder":
        return model.decoder.parameters()
    if scope == "decoder_edge":
        params = list(model.decoder.parameters())
        if getattr(model, "use_edgeconv_branch", False):
            params.extend(list(model.edgeconv_branch.parameters()))
        return params
    raise ValueError(f"unsupported train scope: {scope}")


def edm_weighted_mse(model, pc_noisy, pc_clean, sigma, loss_mode="mse", under_length_weight=0.0):
    pc_pred = model.predict_clean(pc_noisy, sigma=sigma)
    mse = ((pc_pred - pc_clean) ** 2.0).sum(dim=-1)
    sigma_for_weight = model.clamp_edm_sigma(sigma)
    sigma2 = sigma_for_weight ** 2.0
    sigma_data2 = model.sigma_data ** 2.0
    weight = (sigma2 + sigma_data2) / (sigma2 * sigma_data2)
    loss = (mse * weight.reshape(pc_noisy.shape[0], 1)).mean()
    if loss_mode == "mse":
        return loss, pc_pred
    target_len = jt.sqrt(((pc_clean - pc_noisy) ** 2.0).sum(dim=-1) + 1e-8)
    pred_len = jt.sqrt(((pc_pred - pc_noisy) ** 2.0).sum(dim=-1) + 1e-8)
    if loss_mode == "mse_underlen":
        under = jt.maximum(target_len - pred_len, 0.0)
        under_loss = ((under / model.clamp_edm_sigma(sigma).reshape(pc_noisy.shape[0], 1)) ** 2.0).mean()
        loss = loss + float(under_length_weight) * under_loss
        return loss, pc_pred
    raise ValueError(f"unsupported loss mode: {loss_mode}")


def evaluate_model(model, pc_noisy_np, pc_clean_np, sigma_np, batch_size, mode="fixed"):
    model.eval()
    rows = []
    preds = []
    with jt.no_grad():
        for start in range(0, pc_noisy_np.shape[0], batch_size):
            end = min(start + batch_size, pc_noisy_np.shape[0])
            pc_noisy = jt.array(pc_noisy_np[start:end])
            sigma = jt.array(sigma_np[start:end])
            if mode == "heun":
                pc_pred, _ = model.denoise_langevin_dynamics(pc_noisy)
            elif mode == "fixed":
                pc_pred = model.predict_clean(pc_noisy, sigma=sigma)
            else:
                raise ValueError(f"unsupported eval mode: {mode}")
            preds.append(pc_pred.detach().numpy().astype(np.float32, copy=False))
    pred_np = np.concatenate(preds, axis=0)
    for i in range(pc_noisy_np.shape[0]):
        row = {
            "index": i,
            **score_prediction(pc_noisy_np[i], pc_clean_np[i], pred_np[i]),
            **displacement_metrics(pc_noisy_np[i], pc_clean_np[i], pred_np[i]),
        }
        rows.append(row)
    summary = {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--train-scope", choices=["decoder", "decoder_edge", "all"], default="decoder_edge")
    parser.add_argument("--loss-mode", choices=["mse", "mse_underlen"], default="mse")
    parser.add_argument("--under-length-weight", type=float, default=0.05)
    parser.add_argument("--eval-mode", choices=["fixed", "heun"], default="fixed")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    data = load_hard_patch_npz(args.dataset)
    pc_noisy_np = data["pc_noisy"]
    pc_clean_np = data["pc_clean"]
    sigma_np = data["score_sigma"]

    model = load_model(checkpoint)
    set_train_scope(model, args.train_scope)
    optimizer = jt.optim.Adam(collect_train_parameters(model, args.train_scope), lr=args.lr)

    before_rows, before_summary = evaluate_model(
        model,
        pc_noisy_np,
        pc_clean_np,
        sigma_np,
        batch_size=args.batch_size,
        mode=args.eval_mode,
    )
    write_csv(out_dir / "before_eval.csv", before_rows)
    write_json(out_dir / "before_summary.json", before_summary)
    print("Before:", before_summary, flush=True)

    best_score = before_summary["score"]["mean"]
    best_epoch = -1
    history = []
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch_idx in iter_batches(pc_noisy_np.shape[0], args.batch_size, rng, shuffle=True):
            pc_noisy = jt.array(pc_noisy_np[batch_idx])
            pc_clean = jt.array(pc_clean_np[batch_idx])
            sigma = jt.array(sigma_np[batch_idx])
            loss, _ = edm_weighted_mse(
                model,
                pc_noisy,
                pc_clean,
                sigma,
                loss_mode=args.loss_mode,
                under_length_weight=args.under_length_weight,
            )
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            rows, summary = evaluate_model(
                model,
                pc_noisy_np,
                pc_clean_np,
                sigma_np,
                batch_size=args.batch_size,
                mode=args.eval_mode,
            )
            score_mean = summary["score"]["mean"]
            record = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "score_mean": score_mean,
                "score_median": summary["score"]["median"],
                "length_ratio_mean": summary["length_ratio"]["mean"],
                "cosine_mean": summary["cosine"]["mean"],
                "under_length_rate_mean": summary["under_length_rate"]["mean"],
            }
            history.append(record)
            write_csv(out_dir / "epoch_log.csv", history)
            print(record, flush=True)
            if score_mean >= best_score:
                best_score = score_mean
                best_epoch = epoch
                model.save(str(out_dir / "checkpoint_best.pkl"))
                write_csv(out_dir / "best_eval.csv", rows)
                write_json(out_dir / "best_summary.json", summary)
        model.train()

    model.save(str(out_dir / "checkpoint_last.pkl"))
    final_rows, final_summary = evaluate_model(
        model,
        pc_noisy_np,
        pc_clean_np,
        sigma_np,
        batch_size=args.batch_size,
        mode=args.eval_mode,
    )
    write_csv(out_dir / "final_eval.csv", final_rows)
    write_json(out_dir / "final_summary.json", final_summary)
    write_json(
        out_dir / "train_summary.json",
        {
            "checkpoint": str(checkpoint.resolve()),
            "dataset": str(Path(args.dataset).resolve()),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "final_summary": final_summary,
            "args": vars(args),
        },
    )
    print("Final:", final_summary, flush=True)
    print(f"Wrote outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
