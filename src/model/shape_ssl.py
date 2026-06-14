from math import ceil
from pathlib import Path
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn
from scipy.spatial import cKDTree

from .feature import (
    MultiScaleLocalSelfAttentionBlock,
    PointLayerNorm,
    apply_edge_linear,
    apply_point_linear,
    gather_neighbors,
    get_knn_idx,
)
from .vm import VelocityModule, get_random_indices
from ..data.asset import Asset


def farthest_point_indices(points, count):
    count = min(max(int(count), 1), points.shape[0])
    selected = np.empty((count,), dtype=np.int64)
    selected[0] = 0
    min_distance = ((points - points[0]) ** 2.0).sum(axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(min_distance))
        distance = (
            (points - points[selected[index]]) ** 2.0
        ).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
    return selected


def build_region_layout(points, region_count, points_per_region):
    center_indices = farthest_point_indices(points, region_count)
    _, neighbor_indices = cKDTree(points).query(
        points[center_indices],
        k=min(int(points_per_region), points.shape[0]),
    )
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]
    return center_indices.astype(np.int32), neighbor_indices


def region_arrays(points, center_indices, neighbor_indices):
    centers = points[center_indices]
    local_points = points[neighbor_indices] - centers[:, None, :]
    return (
        local_points.astype(np.float32, copy=False),
        centers.astype(np.float32, copy=False),
    )


def mixed_region_mask(
    centers,
    mask_ratio_min,
    mask_ratio_max,
    spatial_fraction,
):
    region_count = centers.shape[0]
    ratio = float(np.random.uniform(mask_ratio_min, mask_ratio_max))
    mask_count = min(
        region_count - 1,
        max(1, int(round(region_count * ratio))),
    )
    spatial_count = min(
        mask_count,
        max(1, int(round(mask_count * spatial_fraction))),
    )
    anchor = int(np.random.randint(region_count))
    distances = ((centers - centers[anchor]) ** 2.0).sum(axis=1)
    spatial_indices = np.argsort(distances)[:spatial_count]
    mask = np.zeros((region_count,), dtype=np.float32)
    mask[spatial_indices] = 1.0
    remaining = np.flatnonzero(mask < 0.5)
    random_count = mask_count - spatial_count
    if random_count > 0:
        chosen = np.random.choice(
            remaining,
            size=random_count,
            replace=False,
        )
        mask[chosen] = 1.0
    return mask, np.float32(mask.mean())


class MaskedShapeProcessor(nn.Module):
    """Region PointNet + spatial transformer masked shape autoencoder."""

    def __init__(
        self,
        token_dim=128,
        region_knn=(8, 16),
        num_blocks=4,
        points_per_region=64,
        relative_bias_dim=32,
        max_reconstruction_radius=0.08,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.region_knn = [int(value) for value in region_knn]
        self.num_blocks = int(num_blocks)
        self.points_per_region = int(points_per_region)
        self.max_reconstruction_radius = float(
            max_reconstruction_radius
        )

        self.point_mlp_1 = nn.Linear(3, 64)
        self.point_mlp_2 = nn.Linear(64, self.token_dim)
        self.center_proj_1 = nn.Linear(3, 64)
        self.center_proj_2 = nn.Linear(64, self.token_dim)
        self.mask_token = jt.randn((1, 1, self.token_dim)) * 0.02
        self.act = nn.ReLU()
        self.blocks = []
        for index in range(self.num_blocks):
            block = MultiScaleLocalSelfAttentionBlock(
                dim=self.token_dim,
                knn_scales=self.region_knn,
                ffn_hidden_dim=self.token_dim * 2,
                relative_position_bias_hidden_dim=relative_bias_dim,
                global_attn_bias_init=0.5,
            )
            setattr(self, f"block_{index}", block)
            self.blocks.append(block)
        self.output_norm = PointLayerNorm(self.token_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim * 2),
            nn.ReLU(),
            nn.Linear(self.token_dim * 2, self.points_per_region * 3),
        )

    def encode(self, region_points, region_centers, mask=None):
        batch_size, region_count, _, _ = region_points.shape
        feature = self.act(
            apply_edge_linear(self.point_mlp_1, region_points)
        )
        feature = self.act(
            apply_edge_linear(self.point_mlp_2, feature)
        ).max(dim=2)
        position = self.act(
            apply_point_linear(self.center_proj_1, region_centers)
        )
        position = apply_point_linear(self.center_proj_2, position)
        if mask is not None:
            mask = mask.reshape(batch_size, region_count, 1)
            mask_token = self.mask_token.broadcast(
                (batch_size, region_count, self.token_dim)
            )
            feature = feature * (1.0 - mask) + mask_token * mask
        feature = feature + position

        max_knn = min(max(self.region_knn), region_count - 1)
        neighbor_idx = get_knn_idx(
            region_centers,
            region_centers,
            k=max_knn,
            offset=1,
        )
        for block in self.blocks:
            global_token = feature.mean(dim=1, keepdims=True)
            feature = block(
                feature,
                neighbor_idx,
                global_token=global_token,
                xyz=region_centers,
            )
        feature = self.output_norm(feature)
        global_token = self.global_proj(
            feature.mean(dim=1)
        ).reshape(batch_size, 1, self.token_dim)
        return feature, global_token

    def reconstruct(self, tokens):
        batch_size, region_count, _ = tokens.shape
        output = self.reconstruction_head(
            tokens.reshape(-1, self.token_dim)
        )
        output = jt.tanh(output) * self.max_reconstruction_radius
        return output.reshape(
            batch_size,
            region_count,
            self.points_per_region,
            3,
        )

    def execute(self, region_points, region_centers, mask=None):
        tokens, global_token = self.encode(
            region_points,
            region_centers,
            mask=mask,
        )
        reconstruction_tokens = tokens + global_token.broadcast(
            (tokens.shape[0], tokens.shape[1], tokens.shape[2])
        )
        return self.reconstruct(reconstruction_tokens), tokens, global_token


class MaskedShapePretrainModule(VelocityModule):
    """Self-supervised masked reconstruction wrapper for the shape processor."""

    def __init__(self, model_config, transform_config):
        # ModelSpec initialization without constructing a VM denoiser.
        nn.Module.__init__(self)
        self.model_config = dict(model_config)
        self.transform_config = dict(transform_config)
        self._is_predict = False
        cfg = self.model_config
        self.region_count = int(cfg.get("region_count", 256))
        self.points_per_region = int(cfg.get("points_per_region", 64))
        self.mask_ratio_min = float(cfg.get("mask_ratio_min", 0.4))
        self.mask_ratio_max = float(cfg.get("mask_ratio_max", 0.7))
        self.spatial_mask_fraction = float(
            cfg.get("spatial_mask_fraction", 0.7)
        )
        self.noise_std_min = float(cfg.get("noise_std_min", 0.005))
        self.noise_std_max = float(cfg.get("noise_std_max", 0.020))
        self.noise_type = str(cfg.get("noise_type", "laplace"))
        self.reconstruction_scale = float(
            cfg.get("reconstruction_scale", 0.01)
        )
        self.fscore_threshold = float(cfg.get("fscore_threshold", 0.01))
        self.processor = MaskedShapeProcessor(
            token_dim=int(cfg.get("token_dim", 128)),
            region_knn=cfg.get("region_knn", [8, 16]),
            num_blocks=int(cfg.get("num_blocks", 4)),
            points_per_region=self.points_per_region,
            relative_bias_dim=int(cfg.get("relative_bias_dim", 32)),
            max_reconstruction_radius=float(
                cfg.get("max_reconstruction_radius", 0.08)
            ),
        )

    def get_train_transform(self):
        return super().get_train_transform()

    def get_validate_transform(self):
        return super().get_validate_transform()

    def get_predict_transform(self):
        return super().get_predict_transform()

    def reconstruction_metrics(self, prediction, target, mask):
        difference = (
            prediction.unsqueeze(3) - target.unsqueeze(2)
        )
        distance2 = (difference ** 2.0).sum(dim=-1)
        pred_to_target = distance2.min(dim=3)
        target_to_pred = distance2.min(dim=2)
        region_cd = (
            pred_to_target.mean(dim=2)
            + target_to_pred.mean(dim=2)
        )
        weight = mask.reshape(mask.shape[0], mask.shape[1])
        denominator = weight.sum() + 1e-6
        masked_cd = (region_cd * weight).sum() / denominator
        masked_rmse = jt.sqrt(masked_cd * 0.5 + 1e-8)

        threshold2 = self.fscore_threshold ** 2.0
        precision_region = (
            (pred_to_target < threshold2).float().mean(dim=2)
        )
        recall_region = (
            (target_to_pred < threshold2).float().mean(dim=2)
        )
        precision = (precision_region * weight).sum() / denominator
        recall = (recall_region * weight).sum() / denominator
        fscore = 2.0 * precision * recall / (
            precision + recall + 1e-6
        )
        loss = masked_cd / max(self.reconstruction_scale ** 2.0, 1e-8)
        return {
            "masked_reconstruction_loss": loss,
            "masked_chamfer": masked_cd,
            "masked_rmse": masked_rmse,
            "masked_fscore": fscore,
            "masked_precision": precision,
            "masked_recall": recall,
            "mask_ratio": weight.mean(),
        }

    def training_step(self, batch: Dict) -> Dict:
        region_points = batch["shape_region_input_points"]
        target_points = batch["shape_region_target_points"]
        region_centers = batch["shape_region_centers"]
        mask = batch["shape_region_mask"]
        prediction, _, _ = self.processor(
            region_points,
            region_centers,
            mask=mask,
        )
        metrics = self.reconstruction_metrics(
            prediction,
            target_points,
            mask,
        )
        metrics["noise_std"] = batch["noise_std"].mean()
        return metrics

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        result = []
        for asset in batch:
            if asset.sampled_vertices is None or asset.meta is None:
                raise ValueError("shape pretraining requires cached clean regions")
            centers_idx = asset.meta["region_center_indices"]
            neighbors_idx = asset.meta["region_neighbor_indices"]
            clean = asset.sampled_vertices
            noise_std = float(
                np.random.uniform(
                    self.noise_std_min,
                    self.noise_std_max,
                )
            )
            if self.noise_type == "laplace":
                noise = np.random.laplace(
                    0.0,
                    noise_std,
                    size=clean.shape,
                )
            elif self.noise_type == "gaussian":
                noise = np.random.randn(*clean.shape) * noise_std
            else:
                raise ValueError(
                    f"unsupported pretrain noise type: {self.noise_type}"
                )
            noisy = (
                clean + noise.astype(np.float32, copy=False)
            ).astype(np.float32, copy=False)
            input_points, centers = region_arrays(
                noisy,
                centers_idx,
                neighbors_idx,
            )
            target_points = (
                clean[neighbors_idx] - noisy[centers_idx, None, :]
            ).astype(np.float32, copy=False)
            mask, mask_ratio = mixed_region_mask(
                centers,
                self.mask_ratio_min,
                self.mask_ratio_max,
                self.spatial_mask_fraction,
            )
            result.append(
                {
                    "shape_region_input_points": input_points,
                    "shape_region_target_points": target_points,
                    "shape_region_centers": centers,
                    "shape_region_mask": mask,
                    "noise_std": np.asarray(
                        [noise_std],
                        dtype=np.float32,
                    ),
                }
            )
        return result

    def predict_step(self, batch: Dict):
        raise NotImplementedError("shape pretraining has no prediction mode")


class ShapeTokenInjector(nn.Module):
    def __init__(
        self,
        patch_dim=256,
        token_dim=128,
        context_knn=4,
        relative_bias_dim=32,
    ):
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.token_dim = int(token_dim)
        self.context_knn = int(context_knn)
        self.scale = self.token_dim ** -0.5

        self.query_proj = nn.Linear(self.patch_dim, self.token_dim)
        self.key_proj = nn.Linear(self.token_dim, self.token_dim)
        self.value_proj = nn.Linear(self.token_dim, self.token_dim)
        self.bias_1 = nn.Linear(4, relative_bias_dim)
        self.bias_2 = nn.Linear(relative_bias_dim, 1)
        self.context_proj = nn.Linear(self.token_dim, self.patch_dim)
        self.global_scale = nn.Linear(self.token_dim, self.patch_dim)
        self.global_shift = nn.Linear(self.token_dim, self.patch_dim)
        self.output_norm = PointLayerNorm(self.patch_dim)
        self.context_gate = jt.ones((1,)) * -2.0
        self.act = nn.ReLU()

    def execute(
        self,
        patch_feature,
        point_global,
        region_tokens,
        region_centers,
        global_token,
    ):
        k = min(self.context_knn, region_tokens.shape[1])
        neighbor_idx = get_knn_idx(
            point_global,
            region_centers,
            k=k,
            offset=0,
        )
        keys = gather_neighbors(
            apply_point_linear(self.key_proj, region_tokens),
            neighbor_idx,
        )
        values = gather_neighbors(
            apply_point_linear(self.value_proj, region_tokens),
            neighbor_idx,
        )
        centers = gather_neighbors(region_centers, neighbor_idx)
        relative = centers - point_global.unsqueeze(2)
        distance = jt.sqrt(
            (relative ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
        )
        bias_input = jt.concat([relative, distance], dim=-1)
        bias = self.act(apply_edge_linear(self.bias_1, bias_input))
        bias = apply_edge_linear(self.bias_2, bias).reshape(
            patch_feature.shape[0],
            patch_feature.shape[1],
            k,
        )
        query = apply_point_linear(
            self.query_proj,
            patch_feature,
        ).unsqueeze(2)
        logits = (query * keys).sum(dim=-1) * self.scale + bias
        attention = nn.softmax(logits, dim=-1)
        context = (attention.unsqueeze(-1) * values).sum(dim=2)
        context = apply_point_linear(self.context_proj, context)

        global_flat = global_token.reshape(
            global_token.shape[0],
            global_token.shape[2],
        )
        scale = self.global_scale(global_flat).unsqueeze(1)
        shift = self.global_shift(global_flat).unsqueeze(1)
        conditioned = patch_feature * (1.0 + 0.1 * scale) + 0.1 * shift
        gate = jt.sigmoid(self.context_gate)
        return self.output_norm(conditioned + gate * context), gate


class ShapeContextVelocityModule(VelocityModule):
    """Pure VM denoiser conditioned on tokens from the complete noisy shape."""

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        if self.use_edm:
            raise ValueError("ShapeContextVelocityModule requires use_edm=false")
        cfg = self.model_config
        self.region_count = int(cfg.get("shape_region_count", 256))
        self.points_per_region = int(
            cfg.get("shape_points_per_region", 64)
        )
        self.shape_token_dim = int(cfg.get("shape_token_dim", 128))
        self.shape_processor = MaskedShapeProcessor(
            token_dim=self.shape_token_dim,
            region_knn=cfg.get("shape_region_knn", [8, 16]),
            num_blocks=int(cfg.get("shape_num_blocks", 4)),
            points_per_region=self.points_per_region,
            relative_bias_dim=int(
                cfg.get("shape_relative_bias_dim", 32)
            ),
            max_reconstruction_radius=float(
                cfg.get("shape_max_reconstruction_radius", 0.08)
            ),
        )
        self.shape_injector = ShapeTokenInjector(
            patch_dim=self.encoder.embedding_dim,
            token_dim=self.shape_token_dim,
            context_knn=int(cfg.get("shape_context_knn", 4)),
            relative_bias_dim=int(
                cfg.get("shape_relative_bias_dim", 32)
            ),
        )
        self.shape_pretrained_ckpt = cfg.get(
            "shape_pretrained_ckpt",
            None,
        )
        if self.shape_pretrained_ckpt:
            self.load_shape_pretrained(self.shape_pretrained_ckpt)
        self.shape_processor.mask_token.stop_grad()
        for parameter in self.shape_processor.reconstruction_head.parameters():
            parameter.stop_grad()

    def get_shape_train_parameters(self):
        excluded = {
            id(self.shape_processor.mask_token),
            *{
                id(parameter)
                for parameter in self.shape_processor.reconstruction_head.parameters()
            },
        }
        return [
            parameter
            for parameter in self.shape_processor.parameters()
            if id(parameter) not in excluded
        ]

    def load_shape_pretrained(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"shape_pretrained_ckpt not found: {path}")
        state = jt.load(path)
        prefix = "processor."
        processor_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not processor_state:
            prefix = "shape_processor."
            processor_state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
        if not processor_state:
            raise ValueError(
                f"no processor parameters found in checkpoint: {path}"
            )
        self.shape_processor.load_state_dict(processor_state)
        print(f"Loaded pretrained shape processor: {path}")

    def encode_shape(self, region_points, region_centers):
        return self.shape_processor.encode(
            region_points,
            region_centers,
            mask=None,
        )

    def predict_displacement_context(
        self,
        pc_noisy,
        patch_seed,
        region_points,
        region_centers,
        point_idx=None,
        encoded_shape=None,
    ):
        if encoded_shape is None:
            encoded_shape = self.encode_shape(
                region_points,
                region_centers,
            )
        region_tokens, global_token = encoded_shape
        feature = self.encoder(pc_noisy)
        point_global = pc_noisy + patch_seed
        feature, gate = self.shape_injector(
            feature,
            point_global,
            region_tokens,
            region_centers,
            global_token,
        )
        if point_idx is not None:
            feature = feature[:, point_idx, :]
        displacement = self.decoder(
            feature.reshape(-1, feature.shape[-1])
        ).reshape(feature.shape[0], feature.shape[1], 3)
        return displacement, gate

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        patch_seed = batch["patch_seed"].reshape(-1, 1, 3)
        region_points = batch["shape_region_points"]
        region_centers = batch["shape_region_centers"]
        if region_points.shape[0] != pc_noisy.shape[0]:
            raise ValueError(
                "shape-context training currently requires one patch per shape"
            )

        point_idx = get_random_indices(
            pc_noisy.shape[1],
            self.num_train_points,
        )
        prediction, gate = self.predict_displacement_context(
            pc_noisy,
            patch_seed,
            region_points,
            region_centers,
            point_idx=point_idx,
        )
        target = pc_clean - pc_noisy
        noisy_for_loss = pc_noisy
        clean_for_loss = pc_clean
        if point_idx is not None:
            target = target[:, point_idx, :]
            noisy_for_loss = pc_noisy[:, point_idx, :]
            clean_for_loss = pc_clean[:, point_idx, :]
        displacement_loss = (
            ((prediction - target) ** 2.0) / self.dsm_sigma
        ).sum(dim=-1).mean()
        surface_loss = self.get_normalized_surface_loss(
            pc_pred=noisy_for_loss + prediction,
            pc_clean=pc_clean,
            pc_anchor=clean_for_loss,
        )
        return {
            "displacement_loss": displacement_loss,
            "normalized_surface_loss": surface_loss,
            "context_gate": gate,
        }

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        result = []
        for asset in batch:
            if not self.is_predict():
                if asset.meta is None:
                    raise ValueError("missing cached region metadata")
                noisy = asset.sampled_vertices_noisy
                if noisy is None:
                    raise ValueError("missing noisy full shape")
                centers_idx = asset.meta["region_center_indices"]
                neighbors_idx = asset.meta["region_neighbor_indices"]
                region_points, region_centers = region_arrays(
                    noisy,
                    centers_idx,
                    neighbors_idx,
                )
                result.append(
                    {
                        "pc_noisy": asset.meta["pc_noisy"],
                        "pc_clean": asset.meta["pc_clean"],
                        "patch_seed": asset.meta["patch_seed"],
                        "shape_region_points": region_points,
                        "shape_region_centers": region_centers,
                    }
                )
            else:
                noisy = asset.sampled_vertices_noisy
                if noisy is None:
                    raise ValueError("missing noisy point cloud")
                centers_idx, neighbors_idx = build_region_layout(
                    noisy,
                    self.region_count,
                    self.points_per_region,
                )
                region_points, region_centers = region_arrays(
                    noisy,
                    centers_idx,
                    neighbors_idx,
                )
                result.append(
                    {
                        "pc_noisy": noisy,
                        "shape_region_points": region_points,
                        "shape_region_centers": region_centers,
                    }
                )
        return result

    def validation_predict(self, batch):
        pc_noisy = batch["pc_noisy"]
        patch_size = pc_noisy.shape[-2]
        pc_noisy = pc_noisy.reshape(-1, patch_size, 3)
        patch_seed = batch["patch_seed"].reshape(-1, 1, 3)
        prediction, _ = self.predict_displacement_context(
            pc_noisy,
            patch_seed,
            batch["shape_region_points"],
            batch["shape_region_centers"],
        )
        return pc_noisy + prediction

    def denoise_full_shape(
        self,
        noisy,
        region_points,
        region_centers,
    ):
        noisy_np = noisy.numpy().astype(np.float32, copy=False)
        point_count = noisy_np.shape[0]
        patch_size = min(self.predict_patch_size, point_count)
        patch_count = min(
            point_count,
            max(1, int(self.predict_seed_k * point_count / patch_size)),
        )
        seed_indices = farthest_point_indices(noisy_np, patch_count)
        tree = cKDTree(noisy_np)
        distances, point_indices = tree.query(
            noisy_np[seed_indices],
            k=patch_size,
        )
        distances = np.asarray(distances, dtype=np.float32)
        point_indices = np.asarray(point_indices, dtype=np.int32)
        seeds = noisy_np[seed_indices]
        covered = np.zeros((point_count,), dtype=np.bool_)
        covered[point_indices.reshape(-1)] = True
        missing_indices = np.flatnonzero(~covered).astype(np.int32)
        if missing_indices.size > 0:
            extra_distances, extra_point_indices = tree.query(
                noisy_np[missing_indices],
                k=patch_size,
            )
            distances = np.concatenate(
                [
                    distances,
                    np.asarray(extra_distances, dtype=np.float32),
                ],
                axis=0,
            )
            point_indices = np.concatenate(
                [
                    point_indices,
                    np.asarray(extra_point_indices, dtype=np.int32),
                ],
                axis=0,
            )
            seeds = np.concatenate(
                [seeds, noisy_np[missing_indices]],
                axis=0,
            )
            patch_count += int(missing_indices.size)
            print(
                f"Shape-context patch coverage: added "
                f"{missing_indices.size} patches."
            )
        patches = noisy_np[point_indices] - seeds[:, None, :]
        normalized_distances = distances / np.maximum(
            distances[:, -1:],
            1e-8,
        )

        region_points = region_points.unsqueeze(0)
        region_centers = region_centers.unsqueeze(0)
        encoded_shape = self.encode_shape(
            region_points,
            region_centers,
        )
        outputs = []
        patch_batch = max(
            1,
            int(ceil(point_count / (self.predict_seed_k_alpha * patch_size))),
        )
        for start in range(0, patch_count, patch_batch):
            end = min(start + patch_batch, patch_count)
            current = jt.array(patches[start:end])
            seed = jt.array(seeds[start:end, None, :])
            batch_size = end - start
            tokens = encoded_shape[0].broadcast(
                (
                    batch_size,
                    encoded_shape[0].shape[1],
                    encoded_shape[0].shape[2],
                )
            )
            global_token = encoded_shape[1].broadcast(
                (batch_size, 1, encoded_shape[1].shape[2])
            )
            centers = region_centers.broadcast(
                (batch_size, region_centers.shape[1], 3)
            )
            displacement, _ = self.predict_displacement_context(
                current,
                seed,
                region_points.broadcast(
                    (
                        batch_size,
                        region_points.shape[1],
                        region_points.shape[2],
                        3,
                    )
                ),
                centers,
                encoded_shape=(tokens, global_token),
            )
            outputs.append(
                (current + displacement + seed).numpy()
            )
        patch_predictions = np.concatenate(outputs, axis=0)

        weighted_sum = np.zeros_like(noisy_np)
        weight_sum = np.zeros((point_count,), dtype=np.float32)
        for patch_index in range(patch_count):
            weights = np.exp(
                -self.patch_fusion_tau * normalized_distances[patch_index]
            ).astype(np.float32)
            indices = point_indices[patch_index]
            np.add.at(
                weighted_sum,
                indices,
                patch_predictions[patch_index] * weights[:, None],
            )
            np.add.at(weight_sum, indices, weights)
        output = noisy_np.copy()
        covered = weight_sum > 1e-8
        output[covered] = (
            weighted_sum[covered] / weight_sum[covered, None]
        )
        return output.astype(np.float32, copy=False)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        results = []
        for index in range(batch["pc_noisy"].shape[0]):
            prediction = self.denoise_full_shape(
                batch["pc_noisy"][index],
                batch["shape_region_points"][index],
                batch["shape_region_centers"][index],
            )
            results.append({"pc_denoised": prediction})
        return results
