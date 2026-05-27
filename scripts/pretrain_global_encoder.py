import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import jittor as jt
from jittor import nn, optim
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DatasetConfig, PCDatasetModule
from src.model.feature import apply_point_linear
from src.model.vm import VelocityModule


def load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def to_numpy_float32(value):
    if isinstance(value, jt.Var):
        value = value.numpy()
    return value.astype(np.float32, copy=False)


def parse_dataset_config(data_config):
    train_dataset_config = None
    if data_config.get("train_dataset", None) is not None:
        train_dataset_config = DatasetConfig.parse(**data_config["train_dataset"])
    return train_dataset_config


def encode_global_token(model, pc):
    encoder = model.encoder
    feat = apply_point_linear(encoder.input_proj_1, pc)
    feat = encoder.act(feat)
    feat = apply_point_linear(encoder.input_proj_2, feat)
    feat = encoder.act(feat)
    return encoder.global_token_generator(feat)[:, 0, :]


def l2_normalize(x, eps=1e-6):
    return x / jt.sqrt((x * x).sum(dim=-1, keepdims=True) + eps)


class DinoHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=1024, bottleneck_dim=256, out_dim=1024):
        super().__init__()
        self.lin_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, bottleneck_dim)
        self.lin_3 = nn.Linear(bottleneck_dim, out_dim)
        self.act = nn.ReLU()

    def execute(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        x = l2_normalize(x)
        return self.lin_3(x)


class GeometryHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=4):
        super().__init__()
        self.lin_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def execute(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        return jt.sigmoid(self.lin_2(x))


def soft_cross_entropy(teacher_probs, student_logits, student_temp):
    student_probs = nn.softmax(student_logits / student_temp, dim=-1)
    return -(teacher_probs * jt.log(student_probs + 1e-8)).sum(dim=-1).mean()


def smooth_l1_loss(pred, target, beta=0.1):
    diff = jt.abs(pred - target)
    return jt.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta).mean()


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
        targets.append([
            np.clip(point_normal_var, 0.0, 1.0),
            np.clip(linearity, 0.0, 1.0),
            np.clip(planarity, 0.0, 1.0),
            np.clip(point_surface_var, 0.0, 1.0),
        ])
    return np.asarray(targets, dtype=np.float32)


def random_rotation(max_degrees, rng):
    if max_degrees <= 0:
        return np.eye(3, dtype=np.float32)
    angle = float(rng.uniform(-max_degrees, max_degrees)) * math.pi / 180.0
    axis = rng.normal(size=(3,))
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z = axis / axis_norm
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.asarray([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float32)


def make_view(pc_clean, cfg, rng):
    B, N, _ = pc_clean.shape
    view = np.empty_like(pc_clean, dtype=np.float32)
    min_keep = int(round(N * float(cfg.get("min_keep_ratio", 1.0))))
    min_keep = max(1, min(N, min_keep))
    noise_min = float(cfg.get("noise_std_min", 0.0))
    noise_max = float(cfg.get("noise_std_max", noise_min))
    rotate_degrees = float(cfg.get("rotate_degrees", 0.0))
    for b in range(B):
        patch = pc_clean[b]
        keep_count = int(rng.integers(min_keep, N + 1))
        keep_idx = rng.choice(N, size=keep_count, replace=False)
        sample_idx = rng.choice(keep_idx, size=N, replace=keep_count < N)
        points = patch[sample_idx].copy()
        rng.shuffle(points)
        rot = random_rotation(rotate_degrees, rng)
        points = points @ rot.T
        noise_std = float(rng.uniform(noise_min, noise_max))
        if noise_std > 0:
            points += rng.laplace(0.0, noise_std, size=points.shape).astype(np.float32)
        view[b] = points.astype(np.float32, copy=False)
    return view


def copy_params(dst, src):
    dst.load_parameters(src.state_dict())


def stop_params(module):
    for param in module.parameters():
        param.stop_grad()


def ema_update(dst_params, src_params, momentum):
    for dst, src in zip(dst_params, src_params):
        dst.update(dst * momentum + src * (1.0 - momentum))


def momentum_schedule(step, total_steps, base, final):
    if total_steps <= 1:
        return final
    progress = step / float(total_steps - 1)
    return final - (final - base) * (math.cos(math.pi * progress) + 1.0) / 2.0


def global_encoder_state(model):
    state = model.state_dict(to="numpy")
    return {
        key: value
        for key, value in state.items()
        if model.is_global_encoder_param_name(key)
    }


def save_global_checkpoint(model, path, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    jt.save(global_encoder_state(model), path)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain/global_dino.yaml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = load_yaml(args.config)
    components = cfg["components"]
    data_config = load_yaml(Path("configs/data") / f"{components['data']}.yaml")
    transform_config = load_yaml(Path("configs/transform") / f"{components['transform']}.yaml")
    model_config = load_yaml(Path("configs/model") / f"{components['model']}.yaml")
    model_config.pop("__target__", None)

    student_model = VelocityModule(model_config, transform_config)
    teacher_model = VelocityModule(model_config, transform_config)
    copy_params(teacher_model, student_model)

    head_cfg = cfg.get("head", {})
    student_head = DinoHead(
        in_dim=student_model.encoder.embedding_dim,
        hidden_dim=int(head_cfg.get("hidden_dim", 1024)),
        bottleneck_dim=int(head_cfg.get("bottleneck_dim", 256)),
        out_dim=int(head_cfg.get("out_dim", 1024)),
    )
    teacher_head = DinoHead(
        in_dim=student_model.encoder.embedding_dim,
        hidden_dim=int(head_cfg.get("hidden_dim", 1024)),
        bottleneck_dim=int(head_cfg.get("bottleneck_dim", 256)),
        out_dim=int(head_cfg.get("out_dim", 1024)),
    )
    copy_params(teacher_head, student_head)
    geo_head = GeometryHead(
        in_dim=student_model.encoder.embedding_dim,
        hidden_dim=int(cfg.get("geometry", {}).get("hidden_dim", 256)),
        out_dim=4,
    )
    stop_params(teacher_model)
    stop_params(teacher_head)

    train_dataset_config = parse_dataset_config(data_config)
    dataset_module = PCDatasetModule(
        process_fn=student_model._process_fn,
        train_dataset_config=train_dataset_config,
        train_transform=student_model.get_train_transform(),
        debug=False,
    )

    optim_cfg = dict(cfg.get("optimizer", {}))
    optim_target = optim_cfg.pop("__target__", "adam")
    if optim_target != "adam":
        raise ValueError("pretrain_global_encoder currently supports adam only")
    optimizer = optim.Adam(
        student_model.global_encoder_parameters()
        + student_head.parameters()
        + geo_head.parameters(),
        **optim_cfg,
    )

    pretrain_cfg = cfg.get("pretrain", {})
    epochs = int(args.epochs if args.epochs is not None else pretrain_cfg.get("epochs", 30))
    output_dir = Path(pretrain_cfg.get("output_dir", "outputs/pretrain/global_encoder"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "config": cfg,
                "data_config": data_config,
                "transform_config": transform_config,
                "model_config": model_config,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    dino_cfg = cfg.get("dino", {})
    geometry_cfg = cfg.get("geometry", {})
    teacher_temp = float(dino_cfg.get("teacher_temp", 0.04))
    student_temp = float(dino_cfg.get("student_temp", 0.1))
    center_momentum = float(dino_cfg.get("center_momentum", 0.9))
    ema_base = float(dino_cfg.get("ema_momentum_base", 0.996))
    ema_final = float(dino_cfg.get("ema_momentum_final", 0.999))
    geometry_weight = float(geometry_cfg.get("weight", 0.05))
    weak_view_cfg = cfg.get("weak_view", {})
    strong_view_cfg = cfg.get("strong_view", {})
    center = jt.zeros((1, int(head_cfg.get("out_dim", 1024)))).stop_grad()

    train_dataloader = dataset_module.train_dataloader()
    assert train_dataloader is not None, "train dataloader is None"
    steps_per_epoch = max(1, math.ceil(len(train_dataloader) / train_dataloader.batch_size))
    if args.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = max(1, epochs * steps_per_epoch)
    global_step = 0
    best_loss = float("inf")
    log_rows = []
    log_path = run_dir / "pretrain_log.csv"

    for epoch in range(epochs):
        student_model.train()
        student_head.train()
        geo_head.train()
        teacher_model.eval()
        teacher_head.eval()

        pbar = tqdm(dataset_module.train_dataloader(), total=steps_per_epoch)
        epoch_losses = []
        epoch_dino_losses = []
        epoch_geo_losses = []
        for step_idx, batch in enumerate(pbar):
            if args.max_steps_per_epoch is not None and step_idx >= args.max_steps_per_epoch:
                break
            pc_clean = to_numpy_float32(batch["pc_clean"])
            patch_size = pc_clean.shape[-2]
            pc_clean = pc_clean.reshape(-1, patch_size, 3)
            teacher_view = jt.array(make_view(pc_clean, weak_view_cfg, rng))
            student_view = jt.array(make_view(pc_clean, strong_view_cfg, rng))
            geo_target = jt.array(geometry_targets(pc_clean, rng))

            with jt.no_grad():
                teacher_token = encode_global_token(teacher_model, teacher_view)
                teacher_logits = teacher_head(teacher_token)
                teacher_probs = nn.softmax((teacher_logits - center) / teacher_temp, dim=-1)

            student_token = encode_global_token(student_model, student_view)
            student_logits = student_head(student_token)
            dino_loss = soft_cross_entropy(teacher_probs, student_logits, student_temp)
            geo_pred = geo_head(student_token)
            geo_loss = smooth_l1_loss(geo_pred, geo_target)
            loss = dino_loss + geometry_weight * geo_loss

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()

            with jt.no_grad():
                batch_center = teacher_logits.mean(dim=0, keepdims=True)
                center.update(center * center_momentum + batch_center * (1.0 - center_momentum))
                momentum = momentum_schedule(global_step, total_steps, ema_base, ema_final)
                ema_update(
                    teacher_model.global_encoder_parameters(),
                    student_model.global_encoder_parameters(),
                    momentum,
                )
                ema_update(teacher_head.parameters(), student_head.parameters(), momentum)

            loss_val = float(loss.item())
            dino_val = float(dino_loss.item())
            geo_val = float(geo_loss.item())
            epoch_losses.append(loss_val)
            epoch_dino_losses.append(dino_val)
            epoch_geo_losses.append(geo_val)
            pbar.set_description(
                f"Epoch {epoch}, loss={loss_val:.5f}, "
                f"dino={dino_val:.5f}, geo={geo_val:.5f}"
            )
            global_step += 1

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        mean_dino_loss = float(np.mean(epoch_dino_losses)) if epoch_dino_losses else float("nan")
        mean_geo_loss = float(np.mean(epoch_geo_losses)) if epoch_geo_losses else float("nan")
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "dino_loss": mean_dino_loss,
            "geo_loss": mean_geo_loss,
            "lr": float(optimizer.lr),
        }
        log_rows.append(row)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "loss", "dino_loss", "geo_loss", "lr"],
            )
            writer.writeheader()
            writer.writerows(log_rows)

        last_path = output_dir / "global_encoder_last.pkl"
        save_global_checkpoint(
            student_model,
            str(last_path),
            {
                "epoch": epoch,
                "loss": mean_loss,
                "seed": args.seed,
                "config": str(Path(args.config).resolve()),
                "selection_metric": "last",
            },
        )
        if np.isfinite(mean_loss) and mean_loss <= best_loss:
            best_loss = mean_loss
            best_path = output_dir / "global_encoder_best.pkl"
            save_global_checkpoint(
                student_model,
                str(best_path),
                {
                    "epoch": epoch,
                    "loss": mean_loss,
                    "seed": args.seed,
                    "config": str(Path(args.config).resolve()),
                    "selection_metric": "min_pretrain_loss",
                },
            )
            print(f"Saved best global encoder checkpoint: {best_path} loss={mean_loss:.6f}")

    if best_loss < float("inf"):
        print(f"Pretraining complete. Best loss={best_loss:.6f}")
        print(f"Best checkpoint: {output_dir / 'global_encoder_best.pkl'}")
    else:
        print("Pretraining complete. No checkpoint was saved because no train step ran.")


if __name__ == "__main__":
    main()
