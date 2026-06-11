import argparse
import json
import sys
from pathlib import Path

import jittor as jt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.noise_classifier import PatchNoiseClassifier  # noqa: E402
from tools.train_multistage_refinement_probe import (  # noqa: E402
    cache_coarse,
    load_patch_file,
)
from tools.train_patch_noise_classifier import (  # noqa: E402
    classification_summary,
    predict,
    sigma_to_label,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/vm_ssl/checkpoint_best.pkl",
    )
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--coarse-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--coarse-mode", choices=["fixed", "heun"], default="heun")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=0.020)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_patch_file(args.dataset)
    coarse = cache_coarse(
        args.checkpoint,
        data,
        args.coarse_cache,
        args.batch_size,
        args.coarse_mode,
    )
    sample_count = min(int(args.num_points), data["pc_noisy"].shape[1])
    point_indices = np.linspace(
        0,
        data["pc_noisy"].shape[1] - 1,
        sample_count,
        dtype=np.int32,
    )
    model = PatchNoiseClassifier(
        k=args.k,
        local_dim=args.local_dim,
        hidden_dim=args.hidden_dim,
    )
    model.load(args.classifier_checkpoint)
    predictions, predicted_sigma = predict(
        model,
        data["pc_noisy"],
        coarse,
        point_indices,
        args.batch_size,
        args.sigma_min,
        args.sigma_max,
    )
    true_sigma = data["score_sigma"].reshape(-1)
    summary = classification_summary(
        sigma_to_label(true_sigma),
        predictions,
        true_sigma,
        predicted_sigma,
    )
    summary["args"] = vars(args)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
