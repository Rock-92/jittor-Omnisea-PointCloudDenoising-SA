import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAIN_CONFIG = "configs/pretrain/global_dino.yaml"
DEFAULT_TRAIN_TASK = "configs/task/train_vm_dino.yaml"
DEFAULT_PRETRAIN_CKPT = "outputs/pretrain/global_encoder/global_encoder_best.pkl"


def run_step(name, cmd):
    print(f"\n===== {name} =====", flush=True)
    print(" ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run global encoder pretraining and then denoising training."
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pretrain-config", default=DEFAULT_PRETRAIN_CONFIG)
    parser.add_argument("--train-task", default=DEFAULT_TRAIN_TASK)
    parser.add_argument("--pretrain-ckpt", default=DEFAULT_PRETRAIN_CKPT)
    parser.add_argument(
        "--force-pretrain",
        action="store_true",
        help="Run pretraining even when the pretrain checkpoint already exists.",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining and start denoising training immediately.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Override pretraining epochs for this run.",
    )
    parser.add_argument(
        "--max-pretrain-steps-per-epoch",
        type=int,
        default=None,
        help="Debug override for pretraining steps per epoch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    python = sys.executable
    pretrain_ckpt = PROJECT_ROOT / args.pretrain_ckpt

    if args.skip_pretrain:
        print("Skipping global encoder pretraining by request.", flush=True)
    elif args.force_pretrain or not pretrain_ckpt.exists():
        pretrain_cmd = [
            python,
            "scripts/pretrain_global_encoder.py",
            "--config",
            args.pretrain_config,
            "--seed",
            str(args.seed),
        ]
        if args.pretrain_epochs is not None:
            pretrain_cmd.extend(["--epochs", str(args.pretrain_epochs)])
        if args.max_pretrain_steps_per_epoch is not None:
            pretrain_cmd.extend(
                [
                    "--max-steps-per-epoch",
                    str(args.max_pretrain_steps_per_epoch),
                ]
            )
        run_step("pretrain global encoder", pretrain_cmd)
    else:
        print(
            f"Found existing pretrain checkpoint, skip pretraining: {pretrain_ckpt}",
            flush=True,
        )

    if not pretrain_ckpt.exists():
        raise FileNotFoundError(
            f"Pretrain checkpoint not found before main training: {pretrain_ckpt}"
        )

    train_cmd = [
        python,
        "run.py",
        "--task",
        args.train_task,
        "--seed",
        str(args.seed),
    ]
    run_step("main denoising training", train_cmd)


if __name__ == "__main__":
    main()
