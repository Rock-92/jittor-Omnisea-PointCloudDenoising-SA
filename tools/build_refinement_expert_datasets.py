import argparse
import json
from pathlib import Path

import numpy as np


EXPERT_RANGES = {
    "low": (0.005, 0.011),
    "medium": (0.009, 0.016),
    "high": (0.014, 0.020001),
}


def subset_npz(source_path, destination_path, indices):
    source = np.load(source_path, allow_pickle=True)
    output = {}
    sample_count = source["score_sigma"].shape[0]
    for key in source.files:
        value = source[key]
        output[key] = value[indices] if value.shape[0] == sample_count else value
    np.savez_compressed(destination_path, **output)


def subset_coarse(source_path, destination_path, indices):
    if not source_path.exists():
        return False
    source = np.load(source_path)
    np.savez_compressed(
        destination_path,
        pc_coarse=source["pc_coarse"][indices],
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default="outputs/refinement_v2_dataset",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/refinement_expert_datasets",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_root = Path(args.out_root)
    summary = {"source_dir": str(source_dir), "experts": {}}
    for expert, (lower, upper) in EXPERT_RANGES.items():
        expert_dir = out_root / expert
        expert_dir.mkdir(parents=True, exist_ok=True)
        expert_summary = {"sigma_range": [lower, upper]}
        for split in ["train", "val"]:
            source_path = source_dir / f"{split}_patches.npz"
            source = np.load(source_path, allow_pickle=True)
            sigma = source["score_sigma"].reshape(-1)
            indices = np.flatnonzero(
                (sigma >= lower) & (sigma < upper)
            )
            if indices.size == 0:
                raise ValueError(f"{expert}/{split} has no patches")
            subset_npz(
                source_path,
                expert_dir / f"{split}_patches.npz",
                indices,
            )
            coarse_copied = subset_coarse(
                source_dir / f"{split}_coarse.npz",
                expert_dir / f"{split}_coarse.npz",
                indices,
            )
            expert_summary[split] = {
                "patches": int(indices.size),
                "shapes": int(len(set(
                    source["rel_path"][indices].tolist()
                ))),
                "sigma_min": float(sigma[indices].min()),
                "sigma_max": float(sigma[indices].max()),
                "coarse_cache_copied": coarse_copied,
            }
        summary["experts"][expert] = expert_summary
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
