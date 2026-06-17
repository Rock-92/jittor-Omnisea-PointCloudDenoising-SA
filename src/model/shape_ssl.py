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


def build_region_layout(
    points,
    region_count,
    points_per_region,
    fps_candidate_count=0,
):
    if 0 < int(fps_candidate_count) < points.shape[0]:
        candidate_indices = np.sort(
            np.random.choice(
                points.shape[0],
                size=int(fps_candidate_count),
                replace=False,
            )
        )
        local_indices = farthest_point_indices(
            points[candidate_indices],
            region_count,
        )
        center_indices = candidate_indices[local_indices]
    else:
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


def clean_region_geometry(points):
    centered = points - points.mean(axis=1, keepdims=True)
    cov = np.matmul(
        centered.transpose(0, 2, 1),
        centered,
    ) / max(points.shape[1] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = eigvals[:, ::-1]
    total = np.maximum(eigvals.sum(axis=1), 1e-12)
    largest = np.maximum(eigvals[:, 0], 1e-12)
    linearity = (eigvals[:, 0] - eigvals[:, 1]) / largest
    planarity = (eigvals[:, 1] - eigvals[:, 2]) / largest
    curvature = eigvals[:, 2] / total
    return np.stack(
        [linearity, planarity, curvature],
        axis=1,
    ).astype(np.float32, copy=False)


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

    def encode(
        self,
        region_points,
        region_centers,
        mask=None,
        return_global=True,
    ):
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
        if not return_global:
            return feature, None
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
        self.fps_candidate_count = int(cfg.get("fps_candidate_count", 0))
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
        self.masked_reconstruction_weight = float(
            cfg.get("masked_reconstruction_weight", 1.0)
        )
        self.all_reconstruction_weight = float(
            cfg.get("all_reconstruction_weight", 0.35)
        )
        self.center_displacement_weight = float(
            cfg.get("center_displacement_weight", 1.0)
        )
        self.geometry_weight = float(cfg.get("geometry_weight", 0.2))
        self.consistency_weight = float(
            cfg.get("consistency_weight", 0.1)
        )
        self.center_displacement_scale = float(
            cfg.get("center_displacement_scale", 0.02)
        )
        self.token_dim = int(cfg.get("token_dim", 128))
        self.processor = MaskedShapeProcessor(
            token_dim=self.token_dim,
            region_knn=cfg.get("region_knn", [8, 16]),
            num_blocks=int(cfg.get("num_blocks", 4)),
            points_per_region=self.points_per_region,
            relative_bias_dim=int(cfg.get("relative_bias_dim", 32)),
            max_reconstruction_radius=float(
                cfg.get("max_reconstruction_radius", 0.08)
            ),
        )
        self.center_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, 3),
        )
        self.geometry_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, 3),
        )

    def get_train_transform(self):
        return super().get_train_transform()

    def get_validate_transform(self):
        return super().get_validate_transform()

    def get_predict_transform(self):
        return super().get_predict_transform()

    def reconstruction_metrics(self, prediction, target, weight, prefix):
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
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        denominator = weight.sum() + 1e-6
        chamfer = (region_cd * weight).sum() / denominator
        rmse = jt.sqrt(chamfer * 0.5 + 1e-8)

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
        loss = chamfer / max(self.reconstruction_scale ** 2.0, 1e-8)
        return {
            f"{prefix}_reconstruction_loss": loss,
            f"{prefix}_chamfer": chamfer,
            f"{prefix}_rmse": rmse,
            f"{prefix}_fscore": fscore,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
        }

    def training_step(self, batch: Dict) -> Dict:
        region_points = batch["shape_region_input_points"]
        target_points = batch["shape_region_target_points"]
        region_centers = batch["shape_region_centers"]
        mask = batch["shape_region_mask"]
        masked_prediction, _, _ = self.processor(
            region_points,
            region_centers,
            mask=mask,
        )
        tokens, global_token = self.processor.encode(
            region_points,
            region_centers,
            mask=None,
            return_global=True,
        )
        all_prediction = self.processor.reconstruct(
            tokens
            + global_token.broadcast(
                (tokens.shape[0], tokens.shape[1], tokens.shape[2])
            )
        )
        metrics = self.reconstruction_metrics(
            masked_prediction,
            target_points,
            mask,
            "masked",
        )
        all_weight = jt.ones_like(mask)
        metrics.update(
            self.reconstruction_metrics(
                all_prediction,
                target_points,
                all_weight,
                "all",
            )
        )
        center_target = batch["shape_region_center_target"]
        center_prediction = self.center_head(
            tokens.reshape(-1, self.token_dim)
        ).reshape(tokens.shape[0], tokens.shape[1], 3)
        center_diff = center_prediction - center_target
        center_loss = (
            (center_diff ** 2.0).sum(dim=-1).mean()
            / max(self.center_displacement_scale ** 2.0, 1e-8)
        )
        center_rmse = jt.sqrt(
            (center_diff ** 2.0).sum(dim=-1).mean() + 1e-8
        )
        center_pred_len = jt.sqrt(
            (center_prediction ** 2.0).sum(dim=-1) + 1e-8
        )
        center_target_len = jt.sqrt(
            (center_target ** 2.0).sum(dim=-1) + 1e-8
        )
        center_cosine = (
            (center_prediction * center_target).sum(dim=-1)
            / (center_pred_len * center_target_len + 1e-8)
        ).mean()

        geometry_target = batch["shape_region_geometry"]
        geometry_prediction = self.geometry_head(
            tokens.reshape(-1, self.token_dim)
        ).reshape(tokens.shape[0], tokens.shape[1], 3)
        geometry_loss = ((geometry_prediction - geometry_target) ** 2.0).mean()

        consistency_loss = jt.array(0.0)
        if "shape_region_input_points_view2" in batch:
            _, global_view2 = self.processor.encode(
                batch["shape_region_input_points_view2"],
                batch["shape_region_centers_view2"],
                mask=None,
                return_global=True,
            )
            g1 = global_token.reshape(global_token.shape[0], -1)
            g2 = global_view2.reshape(global_view2.shape[0], -1)
            g1 = g1 / jt.sqrt((g1 ** 2.0).sum(dim=1, keepdims=True) + 1e-8)
            g2 = g2 / jt.sqrt((g2 ** 2.0).sum(dim=1, keepdims=True) + 1e-8)
            consistency_loss = ((g1 - g2) ** 2.0).sum(dim=1).mean()

        pretrain_loss = (
            self.masked_reconstruction_weight
            * metrics["masked_reconstruction_loss"]
            + self.all_reconstruction_weight
            * metrics["all_reconstruction_loss"]
            + self.center_displacement_weight
            * center_loss
            + self.geometry_weight
            * geometry_loss
            + self.consistency_weight
            * consistency_loss
        )
        metrics["pretrain_loss"] = pretrain_loss
        metrics["center_displacement_loss"] = center_loss
        metrics["center_rmse"] = center_rmse
        metrics["center_cosine"] = center_cosine
        metrics["geometry_loss"] = geometry_loss
        metrics["consistency_loss"] = consistency_loss
        metrics["mask_ratio"] = mask.mean()
        metrics["noise_std"] = batch["noise_std"].mean()
        return metrics

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        result = []
        for asset in batch:
            clean = asset.sampled_vertices
            noisy = asset.sampled_vertices_noisy
            if clean is None or noisy is None:
                raise ValueError(
                    "shape pretraining requires sampled clean and noisy shapes"
                )
            centers_idx, neighbors_idx = build_region_layout(
                noisy,
                self.region_count,
                self.points_per_region,
                self.fps_candidate_count,
            )
            if asset.meta is not None and "noise_std" in asset.meta:
                noise_std = float(asset.meta["noise_std"])
            else:
                noise_std = float("nan")
            input_points, centers = region_arrays(
                noisy,
                centers_idx,
                neighbors_idx,
            )
            clean_region = clean[neighbors_idx]
            target_points = (
                clean_region - noisy[centers_idx, None, :]
            ).astype(np.float32, copy=False)
            center_target = (
                clean[centers_idx] - noisy[centers_idx]
            ).astype(np.float32, copy=False)
            geometry_target = clean_region_geometry(clean_region)
            noise_std_view2 = float(
                np.random.uniform(
                    self.noise_std_min,
                    self.noise_std_max,
                )
            )
            if self.noise_type == "laplace":
                noise_view2 = np.random.laplace(
                    0.0,
                    noise_std_view2,
                    size=clean.shape,
                )
            elif self.noise_type == "gaussian":
                noise_view2 = np.random.randn(*clean.shape) * noise_std_view2
            else:
                raise ValueError(
                    f"unsupported pretrain noise type: {self.noise_type}"
                )
            noisy_view2 = (
                clean + noise_view2.astype(np.float32, copy=False)
            ).astype(np.float32, copy=False)
            centers_idx_view2, neighbors_idx_view2 = build_region_layout(
                noisy_view2,
                self.region_count,
                self.points_per_region,
                self.fps_candidate_count,
            )
            input_points_view2, centers_view2 = region_arrays(
                noisy_view2,
                centers_idx_view2,
                neighbors_idx_view2,
            )
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
                    "shape_region_center_target": center_target,
                    "shape_region_geometry": geometry_target,
                    "shape_region_input_points_view2": input_points_view2,
                    "shape_region_centers_view2": centers_view2,
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


class RegionCrossAttentionModulation(nn.Module):
    """Cross-attend to nearby region tokens and modulate one VM SA layer."""

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
        self.context_scale = nn.Linear(self.token_dim, self.patch_dim)
        self.context_shift = nn.Linear(self.token_dim, self.patch_dim)
        self.output_norm = PointLayerNorm(self.patch_dim)
        self.context_gate = jt.ones((1,)) * -2.0
        self.act = nn.ReLU()

        self.context_scale.weight.assign(
            jt.zeros_like(self.context_scale.weight)
        )
        self.context_scale.bias.assign(
            jt.zeros_like(self.context_scale.bias)
        )
        self.context_shift.weight.assign(
            jt.zeros_like(self.context_shift.weight)
        )
        self.context_shift.bias.assign(
            jt.zeros_like(self.context_shift.bias)
        )

    def execute(
        self,
        patch_feature,
        point_global,
        region_tokens,
        region_centers,
        region_neighbor_idx=None,
    ):
        k = min(self.context_knn, region_tokens.shape[1])
        if region_neighbor_idx is None:
            region_neighbor_idx = get_knn_idx(
                point_global,
                region_centers,
                k=k,
                offset=0,
            )
        keys = gather_neighbors(
            apply_point_linear(self.key_proj, region_tokens),
            region_neighbor_idx,
        )
        values = gather_neighbors(
            apply_point_linear(self.value_proj, region_tokens),
            region_neighbor_idx,
        )
        centers = gather_neighbors(region_centers, region_neighbor_idx)
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
        context_residual = apply_point_linear(
            self.context_proj,
            context,
        )
        scale = jt.tanh(
            apply_point_linear(self.context_scale, context)
        )
        shift = apply_point_linear(self.context_shift, context)
        gate = jt.sigmoid(self.context_gate)
        conditioned = (
            patch_feature * (1.0 + gate * scale)
            + gate * shift
            + gate * context_residual
        )
        return self.output_norm(conditioned), gate


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
        self.fps_candidate_count = int(
            cfg.get("shape_region_fps_candidates", 0)
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
        self.shape_context_knn = int(cfg.get("shape_context_knn", 4))
        self.region_cross_blocks = []
        for block_index in range(self.attention_blocks):
            cross_block = RegionCrossAttentionModulation(
                patch_dim=self.encoder.embedding_dim,
                token_dim=self.shape_token_dim,
                context_knn=self.shape_context_knn,
                relative_bias_dim=int(
                    cfg.get("shape_relative_bias_dim", 32)
                ),
            )
            setattr(
                self,
                f"region_cross_block_{block_index}",
                cross_block,
            )
            self.region_cross_blocks.append(cross_block)
        self.shape_pretrained_ckpt = cfg.get(
            "shape_pretrained_ckpt",
            None,
        )
        if self.shape_pretrained_ckpt:
            self.load_shape_pretrained(self.shape_pretrained_ckpt)
        self.shape_processor.mask_token.stop_grad()
        for parameter in self.shape_processor.reconstruction_head.parameters():
            parameter.stop_grad()
        for parameter in self.shape_processor.global_proj.parameters():
            parameter.stop_grad()

    def get_shape_train_parameters(self):
        excluded = {
            id(self.shape_processor.mask_token),
            *{
                id(parameter)
                for parameter in self.shape_processor.reconstruction_head.parameters()
            },
            *{
                id(parameter)
                for parameter in self.shape_processor.global_proj.parameters()
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
        region_tokens, _ = self.shape_processor.encode(
            region_points,
            region_centers,
            mask=None,
            return_global=False,
        )
        return region_tokens

    def encode_patch_with_region_context(
        self,
        pc_noisy,
        point_global,
        region_tokens,
        region_centers,
        point_idx=None,
    ):
        feature = self.encoder.project_input(pc_noisy)
        # This is VM's native patch-global token, not a shape-global token.
        patch_global_token = self.encoder.global_token_generator(feature)
        region_neighbor_idx = get_knn_idx(
            point_global,
            region_centers,
            k=min(self.shape_context_knn, region_tokens.shape[1]),
            offset=0,
        )

        block_outputs = []
        graph_knn_idx = None
        reuse_knn_idx = None
        if self.encoder.legacy_graph_updates:
            graph_knn_idx = get_knn_idx(
                pc_noisy,
                pc_noisy,
                self.encoder.max_knn,
                offset=1,
            )
        gates = []
        for block_index, (
            block,
            weight,
            cross_block,
        ) in enumerate(
            zip(
                self.encoder.blocks,
                self.encoder.block_weights,
                self.region_cross_blocks,
            )
        ):
            if self.encoder.legacy_graph_updates and block_index == 0:
                block_knn_idx = graph_knn_idx
            elif self.encoder.legacy_graph_updates and block_index == 1:
                reuse_knn_idx = get_knn_idx(
                    feature,
                    feature,
                    self.encoder.max_knn,
                    offset=1,
                )
                block_knn_idx = reuse_knn_idx
            elif self.encoder.legacy_graph_updates:
                block_knn_idx = reuse_knn_idx
            elif block_index == 0:
                block_knn_idx = get_knn_idx(
                    pc_noisy,
                    pc_noisy,
                    self.encoder.max_knn,
                    offset=1,
                )
            else:
                block_knn_idx = get_knn_idx(
                    feature,
                    feature,
                    self.encoder.max_knn,
                    offset=1,
                )
            feature = block(
                feature,
                block_knn_idx,
                global_token=patch_global_token,
                xyz=pc_noisy,
            )
            feature, gate = cross_block(
                feature,
                point_global,
                region_tokens,
                region_centers,
                region_neighbor_idx=region_neighbor_idx,
            )
            gates.append(gate)
            block_outputs.append(feature * weight)

        feature = jt.concat(block_outputs, dim=-1)
        feature = apply_point_linear(self.encoder.fuse, feature)
        if point_idx is not None:
            feature = feature[:, point_idx, :]
        return feature, jt.stack(gates).mean()

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
        point_global = pc_noisy + patch_seed
        feature, gate = self.encode_patch_with_region_context(
            pc_noisy,
            point_global,
            encoded_shape,
            region_centers,
            point_idx=point_idx,
        )
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
        pred_len = jt.sqrt((prediction ** 2.0).sum(dim=-1) + 1e-8)
        target_len = jt.sqrt((target ** 2.0).sum(dim=-1) + 1e-8)
        cosine = (prediction * target).sum(dim=-1) / (
            pred_len * target_len + 1e-8
        )
        surface_loss = self.get_normalized_surface_loss(
            pc_pred=noisy_for_loss + prediction,
            pc_clean=pc_clean,
            pc_anchor=clean_for_loss,
        )
        return {
            "displacement_loss": displacement_loss,
            "normalized_surface_loss": surface_loss,
            "region_context_gate": gate,
            "train_length_ratio": pred_len.mean()
            / (target_len.mean() + 1e-8),
            "train_cosine": cosine.mean(),
            "train_negative_cos_rate": (cosine < 0.0).float().mean(),
            "train_pred_len": pred_len.mean(),
            "train_target_len": target_len.mean(),
        }

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        result = []
        for asset in batch:
            if not self.is_predict():
                if asset.meta is None:
                    raise ValueError("missing patch metadata")
                noisy = asset.sampled_vertices_noisy
                if noisy is None:
                    raise ValueError("missing noisy full shape")
                centers_idx, neighbors_idx = build_region_layout(
                    noisy,
                    self.region_count,
                    self.points_per_region,
                    self.fps_candidate_count,
                )
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
                    self.fps_candidate_count,
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
        region_tokens = self.encode_shape(
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
            tokens = region_tokens.broadcast(
                (
                    batch_size,
                    region_tokens.shape[1],
                    region_tokens.shape[2],
                )
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
                encoded_shape=tokens,
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
