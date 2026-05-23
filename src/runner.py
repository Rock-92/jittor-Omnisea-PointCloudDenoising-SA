import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

import jittor as jt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.asset import Asset
from src.data.dataset import DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

jt.flags.use_cuda = 1

LOADED_CONFIGS: Dict[str, Dict] = {}


def load_config(name: str, path: str) -> Dict:
    """Load a yaml config and keep a copy for run reproducibility."""
    if path.endswith(".yaml"):
        path = path.removesuffix(".yaml")
    path += ".yaml"
    print(f"\033[92mload {name} config: {path}\033[0m")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    LOADED_CONFIGS[name] = {
        "path": path,
        "config": config,
    }
    return config  # type: ignore[return-value]


def save_run_metadata(mode: str, seed: int) -> str:
    """Save command and resolved configs to outputs/runs/<mode>/<timestamp>."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", "runs", mode, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "seed": seed,
                "configs": LOADED_CONFIGS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return run_dir


def debug_fn(data: PCDatasetModule):
    train_dataloader = data.train_dataloader()
    assert train_dataloader is not None, "train_dataloader is None, cannot debug"
    for batch in tqdm(train_dataloader):
        batch: List[Asset]


def parse_dataset_config(data_config: Dict):
    train_dataset_config = None
    validate_dataset_config = None
    predict_dataset_config = None

    if data_config.get("train_dataset", None) is not None:
        train_dataset_config = DatasetConfig.parse(**data_config["train_dataset"])

    if data_config.get("validate_dataset", None) is not None:
        validate_dataset_config = (
            DatasetConfig.parse(**data_config["validate_dataset"]).split_by_cls()
        )

    if data_config.get("predict_dataset", None) is not None:
        predict_dataset_config = (
            DatasetConfig.parse(**data_config["predict_dataset"]).split_by_cls()
        )

    return train_dataset_config, validate_dataset_config, predict_dataset_config


def main(argv: Optional[List[str]] = None):
    LOADED_CONFIGS.clear()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=123)
    args = parser.parse_args(argv)

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    task = load_config("task", args.task)
    mode = task["mode"]
    assert mode in ["train", "predict", "debug", "validate"]
    components = task["components"]

    data_config = load_config("data", os.path.join("configs/data", components["data"]))
    train_dataset_config, validate_dataset_config, predict_dataset_config = (
        parse_dataset_config(data_config)
    )

    transform_config = load_config(
        "transform",
        os.path.join("configs/transform", components["transform"]),
    )

    model_config_name = components.get("model", None)
    if model_config_name is None:
        model = None
    else:
        model_config = load_config(
            "model",
            os.path.join("configs/model", model_config_name),
        )
        model = get_model(model_config=model_config, transform_config=transform_config)

    if model is None:
        train_transform = Transform.parse(**transform_config.get("train_transform", {}))
        validate_transform = Transform.parse(
            **transform_config.get("validate_transform", {})
        )
        predict_transform = Transform.parse(**transform_config.get("predict_transform", {}))
    else:
        train_transform = model.get_train_transform()
        validate_transform = model.get_validate_transform()
        predict_transform = model.get_predict_transform()

    dataset_module = PCDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_dataset_config,
        validate_dataset_config=validate_dataset_config,
        predict_dataset_config=predict_dataset_config,
        train_transform=train_transform,
        validate_transform=validate_transform,
        predict_transform=predict_transform,
        debug=task.get("debug", False),
    )

    train_config_name = components.get("train", None)
    train_config = {}
    if train_config_name is not None:
        train_config = load_config(
            "train",
            os.path.join("configs/train", train_config_name),
        )

    optimizer_config = task.get("optimizer", train_config.get("optimizer", None))
    loss_config = task.get("loss", train_config.get("loss", None))
    trainer_config = task.get("trainer", train_config.get("trainer", None))

    load_ckpt = task.get("load_ckpt", None)
    if load_ckpt is not None and model is not None:
        if not os.path.exists(load_ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {load_ckpt}. "
                "Update `load_ckpt` in the task config or train the model first."
            )
        model.load(load_ckpt)

    writer_config = task.get("writer", None)

    system_config_name = components.get("system", None)
    if system_config_name is not None:
        system_config = load_config(
            "system",
            os.path.join("configs/system", system_config_name),
        )
        system = get_system(
            dataset_module=dataset_module,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            trainer_config=trainer_config,
            writer=get_writer(**writer_config) if writer_config is not None else None,
            **system_config,
        )
    else:
        system = None

    run_dir = save_run_metadata(mode=mode, seed=args.seed)
    print(f"\033[92msaved run metadata: {run_dir}\033[0m")

    if mode == "debug":
        debug_fn(data=dataset_module)
    elif mode == "train":
        assert system is not None, "system is None, cannot train"
        system.train()
    elif mode == "predict":
        assert system is not None, "system is None, cannot predict"
        system.predict()
    else:
        raise ValueError(f"unsupported mode: {mode}")


if __name__ == "__main__":
    main()
