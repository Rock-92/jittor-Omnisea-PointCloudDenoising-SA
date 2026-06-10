import argparse
import sys
from collections import Counter
from pathlib import Path

import jittor as jt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.hard_patch_common import (  # noqa: E402
    evaluate_patch,
    geometry_category,
    load_model,
    local_geometry,
    pca_geometry,
    quantile_summary,
    read_datalist,
    sample_patch,
    save_hard_patch_npz,
    write_csv,
    write_json,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs_result/outputs_EdgeConvBrancg/checkpoints/vm_ssl/checkpoint_best.pkl"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs_result/outputs_analysis/hard_patch_overfit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mesh-root", default=str(PROJECT_ROOT / "dataset_clean"))
    parser.add_argument("--datalist", default="datalist/validate.txt")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num-hard", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--mode", choices=["heun", "fixed"], default="heun")
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
    mesh_root = Path(args.mesh_root)

    usable = [
        rel
        for rel in read_datalist(PROJECT_ROOT / args.datalist)
        if (mesh_root / rel / "models/model_normalized.obj").exists()
    ]
    if not usable:
        raise FileNotFoundError(f"No mesh files found under {mesh_root}")

    model = load_model(checkpoint)
    model.eval()

    rows = []
    patches = {}
    for idx in range(args.candidates):
        rel_path = usable[int(rng.integers(0, len(usable)))]
        print(f"[{idx + 1}/{args.candidates}] {rel_path}", flush=True)
        patch = sample_patch(
            rel_path=rel_path,
            mesh_root=mesh_root,
            rng=rng,
            patch_size=args.patch_size,
            noise_std=args.noise_std,
        )
        score, _ = evaluate_patch(model, patch, mode=args.mode, sigma=args.noise_std)
        geom = {
            **pca_geometry(patch["patch_clean"]),
            **local_geometry(patch["patch_clean"], seed=args.seed + idx),
        }
        x = jt.array(patch["patch_noisy"][None, :, :])
        with jt.no_grad():
            patch_scale = float(model.get_patch_scale(x).item())
        row = {
            "candidate_index": idx,
            "rel_path": rel_path,
            "seed_idx": patch["seed_idx"],
            "noise_std": args.noise_std,
            "patch_scale": patch_scale,
            **score,
            **geom,
        }
        row["geometry_category"] = geometry_category(row)
        rows.append(row)
        patches[idx] = patch

    rows_sorted = sorted(rows, key=lambda row: row["cd_score"])
    selected = rows_sorted[: args.num_hard]
    dataset_path = out_dir / "hard_patches.npz"
    save_hard_patch_npz(dataset_path, selected, patches)

    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "device": "cuda" if args.use_cuda else "cpu",
        "seed": args.seed,
        "mode": args.mode,
        "candidates": args.candidates,
        "num_hard": args.num_hard,
        "patch_size": args.patch_size,
        "noise_std": args.noise_std,
        "dataset": str(dataset_path.resolve()),
        "all_score": quantile_summary([row["cd_score"] for row in rows]),
        "hard_score": quantile_summary([row["cd_score"] for row in selected]),
        "hard_patch_scale": quantile_summary([row["patch_scale"] for row in selected]),
        "hard_category_counts": dict(Counter(row["geometry_category"] for row in selected)),
    }
    write_csv(out_dir / "candidate_patch_records.csv", rows)
    write_csv(out_dir / "selected_hard_patch_records.csv", selected)
    write_json(out_dir / "build_summary.json", summary)
    print(summary, flush=True)
    print(f"Wrote hard patch dataset: {dataset_path}", flush=True)


if __name__ == "__main__":
    main()
