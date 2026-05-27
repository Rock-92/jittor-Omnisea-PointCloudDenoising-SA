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


def scheduled_loss_weights(epoch, schedule_cfg, fallback_geometry_weight):
    if not schedule_cfg or not bool(schedule_cfg.get("enabled", False)):
        return 1.0, float(fallback_geometry_weight)

    hold_epochs = int(schedule_cfg.get("hold_epochs", 5))
    transition_end_epoch = int(schedule_cfg.get("transition_end_epoch", 15))
    start_dino_weight = float(schedule_cfg.get("start_dino_weight", 0.10))
    start_geometry_weight = float(schedule_cfg.get("start_geometry_weight", 0.90))
    end_dino_weight = float(schedule_cfg.get("end_dino_weight", 0.95))
    end_geometry_weight = float(schedule_cfg.get("end_geometry_weight", 0.05))

    if epoch < hold_epochs:
        t = 0.0
    elif epoch >= transition_end_epoch:
        t = 1.0
    else:
        denom = max(1, transition_end_epoch - hold_epochs)
        t = float(epoch - hold_epochs) / float(denom)

    dino_weight = start_dino_weight + t * (end_dino_weight - start_dino_weight)
    geometry_weight = start_geometry_weight + t * (
        end_geometry_weight - start_geometry_weight
    )
    return dino_weight, geometry_weight


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


def surface_neighbor_jitter(points, reference_points, cfg, rng):
    """
    Move each point within the local sampled surface neighborhood.

    The pretrain batch contains cached clean surface samples, not mesh faces, so
    this approximates a small on-surface slide by interpolating toward a nearby
    clean surface sample. The object-level surface stays the same while the
    concrete sampled point locations change.
    """
    if points.shape[0] <= 1:
        return points
    knn = int(cfg.get("surface_jitter_knn", 12))
    knn = max(1, min(knn, reference_points.shape[0] - 1))
    alpha_min = float(cfg.get("surface_jitter_alpha_min", 0.05))
    alpha_max = float(cfg.get("surface_jitter_alpha_max", 0.25))
    if alpha_max <= 0.0:
        return points
    alpha_min = max(0.0, min(alpha_min, alpha_max))

    tree = cKDTree(reference_points)
    _, idx = tree.query(points, k=knn + 1)
    if idx.ndim == 1:
        idx = idx[:, None]
    candidates = idx[:, 1:] if idx.shape[1] > 1 else idx
    choice = rng.integers(0, candidates.shape[1], size=points.shape[0])
    target = reference_points[candidates[np.arange(points.shape[0]), choice]]
    alpha = rng.uniform(alpha_min, alpha_max, size=(points.shape[0], 1)).astype(np.float32)
    return (points + alpha * (target - points)).astype(np.float32, copy=False)


def normalize_barycentric(barycentric):
    barycentric = to_numpy_float32(barycentric)
    barycentric = np.maximum(barycentric, 0.0)
    denom = barycentric.sum(axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return barycentric / denom


def random_barycentric(num_points, rng):
    weights = rng.random((num_points, 3), dtype=np.float32)
    return normalize_barycentric(weights)


def surface_mesh_jitter(surface_info, sample_idx, cfg, rng):
    """
    Move points exactly on cached mesh triangles when cache metadata is present.

    Each cached clean point has a source triangle and barycentric coordinate.
    The jitter mixes that coordinate with a random point on the same triangle,
    keeping the student view on the object surface while changing the concrete
    sampled locations.
    """
    if surface_info is None:
        return None
    required = [
        "mesh_vertices",
        "mesh_faces",
        "surface_face_index",
        "surface_barycentric",
        "patch_seed",
    ]
    if any(surface_info.get(key) is None for key in required):
        return None

    vertices = to_numpy_float32(surface_info["mesh_vertices"])
    faces = surface_info["mesh_faces"]
    if isinstance(faces, jt.Var):
        faces = faces.numpy()
    faces = np.asarray(faces, dtype=np.int64)
    source_face_index = surface_info["surface_face_index"]
    if isinstance(source_face_index, jt.Var):
        source_face_index = source_face_index.numpy()
    face_index = np.asarray(source_face_index, dtype=np.int64)[sample_idx]
    barycentric = surface_info["surface_barycentric"][sample_idx]
    patch_seed = to_numpy_float32(surface_info["patch_seed"]).reshape(1, 3)
    if face_index.size == 0 or vertices.size == 0 or faces.size == 0:
        return None

    alpha_min = float(cfg.get("surface_jitter_alpha_min", 0.05))
    alpha_max = float(cfg.get("surface_jitter_alpha_max", 0.25))
    if alpha_max <= 0.0:
        return None
    alpha_min = max(0.0, min(alpha_min, alpha_max))
    alpha = rng.uniform(alpha_min, alpha_max, size=(sample_idx.shape[0], 1)).astype(np.float32)

    barycentric = normalize_barycentric(barycentric)
    target_barycentric = random_barycentric(sample_idx.shape[0], rng)
    moved_barycentric = normalize_barycentric(
        barycentric + alpha * (target_barycentric - barycentric)
    )
    triangles = vertices[faces[face_index]]
    points_abs = (triangles * moved_barycentric[:, :, None]).sum(axis=1)
    return (points_abs - patch_seed).astype(np.float32, copy=False)


def make_view(pc_clean, cfg, rng, surface_infos=None):
    B, N, _ = pc_clean.shape
    view = np.empty_like(pc_clean, dtype=np.float32)
    min_keep = int(round(N * float(cfg.get("min_keep_ratio", 1.0))))
    min_keep = max(1, min(N, min_keep))
    noise_min = float(cfg.get("noise_std_min", 0.0))
    noise_max = float(cfg.get("noise_std_max", noise_min))
    rotate_degrees = float(cfg.get("rotate_degrees", 0.0))
    use_surface_jitter = bool(cfg.get("surface_jitter", False))
    for b in range(B):
        patch = pc_clean[b]
        keep_count = int(rng.integers(min_keep, N + 1))
        keep_idx = rng.choice(N, size=keep_count, replace=False)
        sample_idx = rng.choice(keep_idx, size=N, replace=keep_count < N)
        points = patch[sample_idx].copy()
        if use_surface_jitter:
            surface_info = (
                surface_infos[b]
                if surface_infos is not None and b < len(surface_infos)
                else None
            )
            surface_points = surface_mesh_jitter(surface_info, sample_idx, cfg, rng)
            if surface_points is None:
                points = surface_neighbor_jitter(points, patch, cfg, rng)
            else:
                points = surface_points
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


def start_params(module_or_params):
    if hasattr(module_or_params, "parameters"):
        params = module_or_params.parameters()
    else:
        params = module_or_params
    for param in params:
        param.start_grad()


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


def state_delta_stats(curr_state, prev_state):
    total_sq = 0.0
    max_abs = 0.0
    changed = 0
    for key, curr in curr_state.items():
        if key not in prev_state:
            continue
        diff = curr - prev_state[key]
        total_sq += float((diff * diff).sum())
        key_max = float(np.max(np.abs(diff))) if diff.size else 0.0
        max_abs = max(max_abs, key_max)
        if key_max > 0.0:
            changed += 1
    return {
        "global_param_delta_l2": float(math.sqrt(total_sq)),
        "global_param_delta_max": max_abs,
        "global_param_changed_keys": changed,
    }


def save_global_checkpoint(model, path, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    jt.save(global_encoder_state(model), path)
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_surface_sidecars(asset_path, cache):
    if asset_path is None:
        return None
    cache_dir = str(Path(asset_path).parent)
    if cache_dir in cache:
        return cache[cache_dir]

    mesh_path = Path(cache_dir) / "mesh.npz"
    surface_path = Path(cache_dir) / "surface_sample.npz"
    if not mesh_path.exists() or not surface_path.exists():
        cache[cache_dir] = None
        return None

    with np.load(mesh_path) as mesh_npz:
        mesh_vertices = mesh_npz["vertices"].astype(np.float32, copy=False)
        mesh_faces = mesh_npz["faces"].astype(np.int32, copy=False)
    with np.load(surface_path) as surface_npz:
        surface_face_index = surface_npz["face_index"].astype(np.int32, copy=False)
        surface_barycentric = surface_npz["barycentric"].astype(np.float32, copy=False)

    cache[cache_dir] = {
        "mesh_vertices": mesh_vertices,
        "mesh_faces": mesh_faces,
        "surface_face_index": surface_face_index,
        "surface_barycentric": surface_barycentric,
    }
    return cache[cache_dir]


def build_patch_surface_info(asset, cache):
    if asset.meta is None:
        return None
    patch_index = asset.meta.get("patch_index")
    patch_seed = asset.meta.get("patch_seed")
    center = asset.meta.get("normalize_center")
    scale = asset.meta.get("normalize_scale")
    if patch_index is None or patch_seed is None or center is None or scale is None:
        return None
    scale = float(scale)
    if scale < 1e-12:
        return None

    sidecars = load_surface_sidecars(asset.path, cache)
    if sidecars is None:
        return None

    patch_index = patch_index.astype(np.int64, copy=False)
    max_index = int(patch_index.max()) if patch_index.size > 0 else -1
    if max_index >= sidecars["surface_face_index"].shape[0]:
        return None

    center = center.reshape(1, 3).astype(np.float32, copy=False)
    mesh_vertices = ((sidecars["mesh_vertices"] - center) / scale).astype(
        np.float32,
        copy=False,
    )
    return {
        "mesh_vertices": mesh_vertices,
        "mesh_faces": sidecars["mesh_faces"],
        "surface_face_index": sidecars["surface_face_index"][patch_index],
        "surface_barycentric": sidecars["surface_barycentric"][patch_index],
        "patch_seed": patch_seed.astype(np.float32, copy=False),
    }


class PretrainProcessFn:
    def __init__(self, model):
        self.model = model
        self.surface_cache = {}

    def __call__(self, batch):
        processed = self.model.process_fn(batch)
        for item, asset in zip(processed, batch):
            item["non"] = {
                "surface": build_patch_surface_info(asset, self.surface_cache),
            }
        return processed


def flatten_surface_infos(surface_batch, patch_count):
    if surface_batch is None:
        return None
    flat = []
    for surface_info in surface_batch:
        if surface_info is None:
            flat.extend([None] * patch_count)
            continue
        info_patch_count = int(surface_info["surface_face_index"].shape[0])
        for patch_idx in range(info_patch_count):
            flat.append(
                {
                    "mesh_vertices": surface_info["mesh_vertices"],
                    "mesh_faces": surface_info["mesh_faces"],
                    "surface_face_index": surface_info["surface_face_index"][patch_idx],
                    "surface_barycentric": surface_info["surface_barycentric"][patch_idx],
                    "patch_seed": surface_info["patch_seed"][patch_idx],
                }
            )
        if info_patch_count < patch_count:
            flat.extend([None] * (patch_count - info_patch_count))
    return flat


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
    # In Jittor, stop_grad after load_parameters can also silence the copied
    # student tensors. Re-enable student gradients explicitly.
    start_params(student_model.global_encoder_parameters())
    start_params(student_head)
    start_params(geo_head)

    train_dataset_config = parse_dataset_config(data_config)
    dataset_module = PCDatasetModule(
        process_fn=PretrainProcessFn(student_model),
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
    base_geometry_weight = float(geometry_cfg.get("weight", 0.05))
    loss_schedule_cfg = cfg.get("loss_schedule", {}) or {}
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
    prev_global_state = global_encoder_state(student_model)

    for epoch in range(epochs):
        student_model.train()
        student_head.train()
        geo_head.train()
        teacher_model.eval()
        teacher_head.eval()

        pbar = tqdm(dataset_module.train_dataloader(), total=steps_per_epoch)
        dino_weight, geometry_weight = scheduled_loss_weights(
            epoch,
            loss_schedule_cfg,
            fallback_geometry_weight=base_geometry_weight,
        )
        epoch_losses = []
        epoch_dino_losses = []
        epoch_geo_losses = []
        for step_idx, batch in enumerate(pbar):
            if args.max_steps_per_epoch is not None and step_idx >= args.max_steps_per_epoch:
                break
            pc_clean_raw = to_numpy_float32(batch["pc_clean"])
            patch_count = pc_clean_raw.shape[1] if pc_clean_raw.ndim == 4 else 1
            patch_size = pc_clean_raw.shape[-2]
            pc_clean = pc_clean_raw.reshape(-1, patch_size, 3)
            surface_infos = flatten_surface_infos(batch.get("surface"), patch_count)
            teacher_view = jt.array(make_view(pc_clean, weak_view_cfg, rng))
            student_view = jt.array(
                make_view(
                    pc_clean,
                    strong_view_cfg,
                    rng,
                    surface_infos=surface_infos,
                )
            )
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
            loss = dino_weight * dino_loss + geometry_weight * geo_loss

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
                f"dino={dino_val:.5f}, geo={geo_val:.5f}, "
                f"w=({dino_weight:.2f},{geometry_weight:.2f})"
            )
            global_step += 1

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        mean_dino_loss = float(np.mean(epoch_dino_losses)) if epoch_dino_losses else float("nan")
        mean_geo_loss = float(np.mean(epoch_geo_losses)) if epoch_geo_losses else float("nan")
        lr_value = float(optimizer.lr)
        improved = np.isfinite(mean_loss) and mean_loss <= best_loss
        curr_global_state = global_encoder_state(student_model)
        delta_stats = state_delta_stats(curr_global_state, prev_global_state)
        prev_global_state = {
            key: value.copy()
            for key, value in curr_global_state.items()
        }
        row = {
            "epoch": epoch,
            "loss": mean_loss,
            "dino_loss": mean_dino_loss,
            "geo_loss": mean_geo_loss,
            "dino_weight": dino_weight,
            "geometry_weight": geometry_weight,
            "lr": lr_value,
            **delta_stats,
        }
        log_rows.append(row)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "loss",
                    "dino_loss",
                    "geo_loss",
                    "dino_weight",
                    "geometry_weight",
                    "lr",
                    "global_param_delta_l2",
                    "global_param_delta_max",
                    "global_param_changed_keys",
                ],
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
        if improved:
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
        best_text = (
            f"{best_loss:.6f}" if best_loss < float("inf") else "nan"
        )
        status = "new best" if improved else f"best={best_text}"
        print(
            f"Epoch {epoch} summary: "
            f"steps={len(epoch_losses)}, "
            f"loss={mean_loss:.6f}, "
            f"dino_loss={mean_dino_loss:.6f}, "
            f"geo_loss={mean_geo_loss:.6f}, "
            f"weights=({dino_weight:.3f},{geometry_weight:.3f}), "
            f"lr={lr_value:.8g}, "
            f"global_delta_l2={delta_stats['global_param_delta_l2']:.6g}, "
            f"global_changed={delta_stats['global_param_changed_keys']}, "
            f"{status}",
            flush=True,
        )

    if best_loss < float("inf"):
        print(f"Pretraining complete. Best loss={best_loss:.6f}")
        print(f"Best checkpoint: {output_dir / 'global_encoder_best.pkl'}")
    else:
        print("Pretraining complete. No checkpoint was saved because no train step ran.")


if __name__ == "__main__":
    main()
