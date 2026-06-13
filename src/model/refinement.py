from typing import List

import jittor as jt
from jittor import nn

from .feature import gather_neighbors, get_knn_idx


def apply_point_mlp(module, x):
    batch_size, num_points, channels = x.shape
    out = module(x.reshape(batch_size * num_points, channels))
    return out.reshape(batch_size, num_points, -1)


def apply_neighbor_mlp(module, x):
    batch_size, num_points, num_neighbors, channels = x.shape
    out = module(
        x.reshape(batch_size * num_points * num_neighbors, channels)
    )
    return out.reshape(batch_size, num_points, num_neighbors, -1)


class GeometryResidualRefiner(nn.Module):
    """
    Lightweight single-step projector for correcting a frozen coarse denoiser.

    The model predicts a gated normal displacement plus a smaller tangential
    displacement. Its output is bounded so a failed refiner cannot make a
    large second denoising step.
    """

    def __init__(
        self,
        k=24,
        local_dim=96,
        hidden_dim=128,
        max_residual=0.006,
        tangent_scale=0.25,
        eps=1e-6,
    ):
        super().__init__()
        self.k = int(k)
        self.local_dim = int(local_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_residual = float(max_residual)
        self.tangent_scale = float(tangent_scale)
        self.eps = float(eps)

        # normalized offset, distance, and neighboring coarse displacement
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(7, self.local_dim),
            nn.ReLU(),
            nn.Linear(self.local_dim, self.local_dim),
            nn.ReLU(),
        )
        # coarse coordinate, coarse displacement, local mean, normal,
        # radius, non-planarity, and pooled neighborhood feature
        point_input_dim = 3 + 3 + 3 + 3 + 1 + 1 + self.local_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(point_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.normal_head = nn.Linear(self.hidden_dim, 1)
        self.tangent_head = nn.Linear(self.hidden_dim, 3)
        self.gate_head = nn.Linear(self.hidden_dim, 1)

        # Start very close to identity while allowing gradients to reach the
        # feature extractor from the first optimizer step.
        self.normal_head.weight.update(
            jt.randn(self.normal_head.weight.shape) * 1e-4
        )
        self.normal_head.bias.update(jt.zeros_like(self.normal_head.bias))
        self.tangent_head.weight.update(
            jt.randn(self.tangent_head.weight.shape) * 1e-4
        )
        self.tangent_head.bias.update(jt.zeros_like(self.tangent_head.bias))

    def _estimate_geometry(self, coarse, neighbor_idx):
        neighbors = gather_neighbors(coarse, neighbor_idx)
        offsets = neighbors - coarse.unsqueeze(2)
        sqdist = (offsets ** 2.0).sum(dim=-1)
        radius = jt.sqrt(sqdist.mean(dim=2, keepdims=True) + self.eps)
        normalized_offsets = offsets / (radius.unsqueeze(-1) + self.eps)

        centered = offsets - offsets.mean(dim=2, keepdims=True)
        x = centered[:, :, :, 0]
        y = centered[:, :, :, 1]
        z = centered[:, :, :, 2]
        cxx = (x * x).mean(dim=2)
        cyy = (y * y).mean(dim=2)
        czz = (z * z).mean(dim=2)
        cxy = (x * y).mean(dim=2)
        cxz = (x * z).mean(dim=2)
        cyz = (y * z).mean(dim=2)
        det = (
            cxx * cyy * czz
            + 2.0 * cxy * cxz * cyz
            - cxx * cyz * cyz
            - cyy * cxz * cxz
            - czz * cxy * cxy
        )
        trace = cxx + cyy + czz
        nonplanarity = 27.0 * jt.maximum(det, 0.0) / jt.maximum(
            trace ** 3.0,
            self.eps ** 3.0,
        )
        nonplanarity = jt.minimum(
            jt.maximum(nonplanarity, 0.0),
            1.0,
        ).unsqueeze(-1)

        normal_candidates: List[jt.Var] = []
        max_pairs = min(5, offsets.shape[2] // 2)
        for pair_idx in range(max_pairs):
            first = offsets[:, :, 2 * pair_idx, :]
            second = offsets[:, :, 2 * pair_idx + 1, :]
            normal = jt.stack(
                [
                    first[:, :, 1] * second[:, :, 2]
                    - first[:, :, 2] * second[:, :, 1],
                    first[:, :, 2] * second[:, :, 0]
                    - first[:, :, 0] * second[:, :, 2],
                    first[:, :, 0] * second[:, :, 1]
                    - first[:, :, 1] * second[:, :, 0],
                ],
                dim=-1,
            )
            normal = normal / (
                jt.sqrt((normal ** 2.0).sum(dim=-1, keepdims=True)) + self.eps
            )
            normal_candidates.append(normal)

        reference = normal_candidates[0]
        aligned = []
        for normal in normal_candidates:
            dot = (normal * reference).sum(dim=-1, keepdims=True)
            sign = jt.where(dot >= 0.0, jt.ones_like(dot), -jt.ones_like(dot))
            aligned.append(normal * sign)
        normal = jt.stack(aligned, dim=2).mean(dim=2)
        normal = normal / (
            jt.sqrt((normal ** 2.0).sum(dim=-1, keepdims=True)) + self.eps
        )
        local_mean = offsets.mean(dim=2)
        return neighbors, normalized_offsets, radius, local_mean, normal, nonplanarity

    def execute(self, coarse, noisy):
        k = min(self.k, coarse.shape[1] - 1)
        neighbor_idx = get_knn_idx(coarse, coarse, k=k, offset=1)
        (
            _,
            normalized_offsets,
            radius,
            local_mean,
            normal,
            nonplanarity,
        ) = self._estimate_geometry(coarse, neighbor_idx)

        coarse_displacement = coarse - noisy
        neighbor_displacement = gather_neighbors(coarse_displacement, neighbor_idx)
        neighbor_distance = jt.sqrt(
            (normalized_offsets ** 2.0).sum(dim=-1, keepdims=True)
            + self.eps
        )
        neighbor_input = jt.concat(
            [
                normalized_offsets,
                neighbor_distance,
                neighbor_displacement,
            ],
            dim=-1,
        )
        neighbor_feature = apply_neighbor_mlp(
            self.neighbor_mlp,
            neighbor_input,
        ).mean(dim=2)

        point_input = jt.concat(
            [
                coarse,
                coarse_displacement,
                local_mean,
                normal,
                radius,
                nonplanarity,
                neighbor_feature,
            ],
            dim=-1,
        )
        feature = apply_point_mlp(self.point_mlp, point_input)

        normal_distance = jt.tanh(
            apply_point_mlp(self.normal_head, feature)
        ) * self.max_residual
        tangent_raw = jt.tanh(apply_point_mlp(self.tangent_head, feature))
        tangent = tangent_raw - (
            (tangent_raw * normal).sum(dim=-1, keepdims=True) * normal
        )
        tangent = tangent * (self.max_residual * self.tangent_scale)
        gate = jt.sigmoid(apply_point_mlp(self.gate_head, feature))

        residual = gate * (normal_distance * normal + tangent)
        residual_norm = jt.sqrt(
            (residual ** 2.0).sum(dim=-1, keepdims=True) + self.eps ** 2.0
        )
        residual_scale = jt.minimum(
            jt.ones_like(residual_norm),
            self.max_residual / residual_norm,
        )
        residual = residual * residual_scale
        return coarse + residual, {
            "residual": residual,
            "gate": gate,
            "normal": normal,
            "nonplanarity": nonplanarity,
        }


class DynamicGeometryRefinementStage(nn.Module):
    """Shared refinement stage with a free, bounded 3D residual."""

    def __init__(
        self,
        k=24,
        local_dim=96,
        hidden_dim=128,
        adaptive_v2=False,
        min_residual_ratio=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.k = int(k)
        self.local_dim = int(local_dim)
        self.hidden_dim = int(hidden_dim)
        self.adaptive_v2 = bool(adaptive_v2)
        self.min_residual_ratio = float(min_residual_ratio)
        self.eps = float(eps)

        # Local offset, distance, current displacement, and initial VM
        # displacement.
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(10, self.local_dim),
            nn.ReLU(),
            nn.Linear(self.local_dim, self.local_dim),
            nn.ReLU(),
        )
        # Current point, current/initial displacement, local geometry,
        # pooled local feature, patch summary, and stage progress.
        point_input_dim = (
            3 + 3 + 3 + 3 + 3 + 1 + 1 + self.local_dim + 4 + 1
        )
        if self.adaptive_v2:
            point_input_dim += 1
        self.point_mlp = nn.Sequential(
            nn.Linear(point_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.direction_head = nn.Linear(self.hidden_dim, 3)
        self.length_head = nn.Linear(self.hidden_dim, 1)
        self.confidence_head = nn.Linear(self.hidden_dim, 1)
        if self.adaptive_v2:
            self.noise_strength_head = nn.Sequential(
                nn.Linear(4, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, 1),
            )

        self.direction_head.weight.update(
            jt.randn(self.direction_head.weight.shape) * 1e-4
        )
        self.direction_head.bias.update(
            jt.randn(self.direction_head.bias.shape) * 1e-4
        )
        self.length_head.weight.update(
            jt.randn(self.length_head.weight.shape) * 1e-4
        )
        self.length_head.bias.update(jt.ones_like(self.length_head.bias) * -3.0)
        if self.adaptive_v2:
            self.confidence_head.bias.update(
                jt.ones_like(self.confidence_head.bias) * -2.0
            )
            self.noise_strength_head[-1].bias.update(
                jt.ones_like(self.noise_strength_head[-1].bias) * -1.0
            )

    def _geometry(self, current, neighbor_idx):
        neighbors = gather_neighbors(current, neighbor_idx)
        offsets = neighbors - current.unsqueeze(2)
        sqdist = (offsets ** 2.0).sum(dim=-1)
        radius = jt.sqrt(sqdist.mean(dim=2, keepdims=True) + self.eps)
        normalized_offsets = offsets / (radius.unsqueeze(-1) + self.eps)

        centered = offsets - offsets.mean(dim=2, keepdims=True)
        x = centered[:, :, :, 0]
        y = centered[:, :, :, 1]
        z = centered[:, :, :, 2]
        cxx = (x * x).mean(dim=2)
        cyy = (y * y).mean(dim=2)
        czz = (z * z).mean(dim=2)
        cxy = (x * y).mean(dim=2)
        cxz = (x * z).mean(dim=2)
        cyz = (y * z).mean(dim=2)
        det = (
            cxx * cyy * czz
            + 2.0 * cxy * cxz * cyz
            - cxx * cyz * cyz
            - cyy * cxz * cxz
            - czz * cxy * cxy
        )
        trace = cxx + cyy + czz
        nonplanarity = 27.0 * jt.maximum(det, 0.0) / jt.maximum(
            trace ** 3.0,
            self.eps ** 3.0,
        )
        nonplanarity = jt.minimum(
            jt.maximum(nonplanarity, 0.0),
            1.0,
        ).unsqueeze(-1)

        first = offsets[:, :, 0, :]
        second = offsets[:, :, max(1, offsets.shape[2] // 3), :]
        normal = jt.stack(
            [
                first[:, :, 1] * second[:, :, 2]
                - first[:, :, 2] * second[:, :, 1],
                first[:, :, 2] * second[:, :, 0]
                - first[:, :, 0] * second[:, :, 2],
                first[:, :, 0] * second[:, :, 1]
                - first[:, :, 1] * second[:, :, 0],
            ],
            dim=-1,
        )
        normal = normal / (
            jt.sqrt((normal ** 2.0).sum(dim=-1, keepdims=True)) + self.eps
        )
        return (
            normalized_offsets,
            radius,
            offsets.mean(dim=2),
            normal,
            nonplanarity,
        )

    def execute(
        self,
        current,
        noisy,
        initial_coarse,
        max_residual,
        stage_progress,
    ):
        k = min(self.k, current.shape[1] - 1)
        neighbor_idx = get_knn_idx(current, current, k=k, offset=1)
        (
            normalized_offsets,
            radius,
            local_mean,
            normal,
            nonplanarity,
        ) = self._geometry(current, neighbor_idx)

        current_displacement = current - noisy
        initial_displacement = initial_coarse - noisy
        neighbor_input = jt.concat(
            [
                normalized_offsets,
                jt.sqrt(
                    (normalized_offsets ** 2.0).sum(
                        dim=-1,
                        keepdims=True,
                    )
                    + self.eps
                ),
                gather_neighbors(current_displacement, neighbor_idx),
                gather_neighbors(initial_displacement, neighbor_idx),
            ],
            dim=-1,
        )
        neighbor_feature = apply_neighbor_mlp(
            self.neighbor_mlp,
            neighbor_input,
        ).mean(dim=2)

        batch_size, num_points, _ = current.shape
        displacement_length = jt.sqrt(
            (current_displacement ** 2.0).sum(dim=-1, keepdims=True)
            + self.eps
        )
        patch_summary = jt.concat(
            [
                displacement_length.mean(dim=1, keepdims=True),
                displacement_length.max(dim=1, keepdims=True),
                radius.mean(dim=1, keepdims=True),
                nonplanarity.mean(dim=1, keepdims=True),
            ],
            dim=-1,
        )
        if self.adaptive_v2:
            noise_strength = jt.sigmoid(
                self.noise_strength_head(patch_summary.reshape(batch_size, 4))
            ).reshape(batch_size, 1, 1)
            residual_ratio = (
                self.min_residual_ratio
                + (1.0 - self.min_residual_ratio) * noise_strength
            )
            residual_cap = float(max_residual) * residual_ratio
        else:
            noise_strength = jt.ones((batch_size, 1, 1))
            residual_cap = jt.ones((batch_size, 1, 1)) * float(max_residual)
        patch_summary = patch_summary.broadcast((batch_size, num_points, 4))
        stage_feature = (
            jt.ones((batch_size, num_points, 1)) * float(stage_progress)
        )
        point_parts = [
            current,
            current_displacement,
            initial_displacement,
            local_mean,
            normal,
            radius,
            nonplanarity,
            neighbor_feature,
            patch_summary,
            stage_feature,
        ]
        if self.adaptive_v2:
            point_parts.append(
                noise_strength.broadcast((batch_size, num_points, 1))
            )
        point_input = jt.concat(point_parts, dim=-1)
        feature = apply_point_mlp(self.point_mlp, point_input)

        direction = apply_point_mlp(self.direction_head, feature)
        direction = direction / (
            jt.sqrt((direction ** 2.0).sum(dim=-1, keepdims=True))
            + self.eps
        )
        length = jt.sigmoid(
            apply_point_mlp(self.length_head, feature)
        ) * residual_cap
        confidence = jt.sigmoid(
            apply_point_mlp(self.confidence_head, feature)
        )
        raw_residual = direction * length
        residual = raw_residual * confidence
        return current + residual, {
            "residual": residual,
            "raw_residual": raw_residual,
            "direction": direction,
            "length": length,
            "confidence": confidence,
            "noise_strength": noise_strength,
            "residual_cap": residual_cap,
            "normal": normal,
            "nonplanarity": nonplanarity,
        }


class MultiStageGeometryRefiner(nn.Module):
    """Iterative projector that recomputes geometry after every update."""

    def __init__(
        self,
        num_stages=2,
        stage_max_residuals=(0.012, 0.008),
        k=24,
        local_dim=96,
        hidden_dim=128,
        adaptive_v2=False,
        min_residual_ratio=0.2,
    ):
        super().__init__()
        self.num_stages = int(num_stages)
        residuals = [float(value) for value in stage_max_residuals]
        if len(residuals) < self.num_stages:
            residuals.extend([residuals[-1]] * (self.num_stages - len(residuals)))
        self.stage_max_residuals = residuals[:self.num_stages]
        self.shared_stage = DynamicGeometryRefinementStage(
            k=k,
            local_dim=local_dim,
            hidden_dim=hidden_dim,
            adaptive_v2=adaptive_v2,
            min_residual_ratio=min_residual_ratio,
        )

    def execute(self, coarse, noisy):
        current = coarse
        stage_outputs = []
        for stage_index in range(self.num_stages):
            progress = (
                float(stage_index) / max(float(self.num_stages - 1), 1.0)
            )
            current, aux = self.shared_stage(
                current=current,
                noisy=noisy,
                initial_coarse=coarse,
                max_residual=self.stage_max_residuals[stage_index],
                stage_progress=progress,
            )
            stage_outputs.append({"prediction": current, **aux})
        return current, {"stages": stage_outputs}


class FusionAwareResidualRefiner(nn.Module):
    """Bounded Stage2 residual model conditioned on overlap consensus."""

    def __init__(
        self,
        k=24,
        local_dim=96,
        hidden_dim=192,
        max_residual=0.008,
        eps=1e-6,
    ):
        super().__init__()
        self.k = int(k)
        self.local_dim = int(local_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_residual = float(max_residual)
        self.eps = float(eps)

        # Coarse neighborhood geometry, Stage1 displacement, and the
        # disagreement between this patch and the overlap consensus.
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(10, self.local_dim),
            nn.ReLU(),
            nn.Linear(self.local_dim, self.local_dim),
            nn.ReLU(),
        )
        point_input_dim = 3 + 3 + 3 + 3 + 3 + 1 + 1 + 1 + self.local_dim + 5
        self.point_mlp = nn.Sequential(
            nn.Linear(point_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.direction_head = nn.Linear(self.hidden_dim, 3)
        self.length_head = nn.Linear(self.hidden_dim, 1)
        self.gate_head = nn.Linear(self.hidden_dim, 1)

        self.direction_head.weight.update(
            jt.randn(self.direction_head.weight.shape) * 1e-4
        )
        self.direction_head.bias.update(
            jt.randn(self.direction_head.bias.shape) * 1e-4
        )
        self.length_head.weight.update(
            jt.randn(self.length_head.weight.shape) * 1e-4
        )
        self.length_head.bias.update(
            jt.ones_like(self.length_head.bias) * -2.5
        )
        self.gate_head.weight.update(
            jt.randn(self.gate_head.weight.shape) * 1e-4
        )
        self.gate_head.bias.update(jt.ones_like(self.gate_head.bias) * -2.0)

    def _geometry(self, coarse, neighbor_idx):
        neighbors = gather_neighbors(coarse, neighbor_idx)
        offsets = neighbors - coarse.unsqueeze(2)
        sqdist = (offsets ** 2.0).sum(dim=-1)
        radius = jt.sqrt(sqdist.mean(dim=2, keepdims=True) + self.eps)
        normalized_offsets = offsets / (radius.unsqueeze(-1) + self.eps)

        centered = offsets - offsets.mean(dim=2, keepdims=True)
        x = centered[:, :, :, 0]
        y = centered[:, :, :, 1]
        z = centered[:, :, :, 2]
        cxx = (x * x).mean(dim=2)
        cyy = (y * y).mean(dim=2)
        czz = (z * z).mean(dim=2)
        cxy = (x * y).mean(dim=2)
        cxz = (x * z).mean(dim=2)
        cyz = (y * z).mean(dim=2)
        det = (
            cxx * cyy * czz
            + 2.0 * cxy * cxz * cyz
            - cxx * cyz * cyz
            - cyy * cxz * cxz
            - czz * cxy * cxy
        )
        trace = cxx + cyy + czz
        nonplanarity = 27.0 * jt.maximum(det, 0.0) / jt.maximum(
            trace ** 3.0,
            self.eps ** 3.0,
        )
        nonplanarity = jt.minimum(
            jt.maximum(nonplanarity, 0.0),
            1.0,
        ).unsqueeze(-1)

        first = offsets[:, :, 0, :]
        second = offsets[:, :, max(1, offsets.shape[2] // 3), :]
        normal = jt.stack(
            [
                first[:, :, 1] * second[:, :, 2]
                - first[:, :, 2] * second[:, :, 1],
                first[:, :, 2] * second[:, :, 0]
                - first[:, :, 0] * second[:, :, 2],
                first[:, :, 0] * second[:, :, 1]
                - first[:, :, 1] * second[:, :, 0],
            ],
            dim=-1,
        )
        normal = normal / (
            jt.sqrt((normal ** 2.0).sum(dim=-1, keepdims=True)) + self.eps
        )
        return normalized_offsets, radius, offsets.mean(dim=2), normal, nonplanarity

    def execute(self, coarse, noisy, consensus, patch_distance):
        k = min(self.k, coarse.shape[1] - 1)
        neighbor_idx = get_knn_idx(coarse, coarse, k=k, offset=1)
        (
            normalized_offsets,
            radius,
            local_mean,
            normal,
            nonplanarity,
        ) = self._geometry(coarse, neighbor_idx)

        stage1_displacement = coarse - noisy
        consensus_delta = consensus - coarse
        neighbor_input = jt.concat(
            [
                normalized_offsets,
                jt.sqrt(
                    (normalized_offsets ** 2.0).sum(
                        dim=-1,
                        keepdims=True,
                    )
                    + self.eps
                ),
                gather_neighbors(stage1_displacement, neighbor_idx),
                gather_neighbors(consensus_delta, neighbor_idx),
            ],
            dim=-1,
        )
        neighbor_feature = apply_neighbor_mlp(
            self.neighbor_mlp,
            neighbor_input,
        ).mean(dim=2)

        displacement_length = jt.sqrt(
            (stage1_displacement ** 2.0).sum(dim=-1, keepdims=True)
            + self.eps
        )
        consensus_length = jt.sqrt(
            (consensus_delta ** 2.0).sum(dim=-1, keepdims=True)
            + self.eps
        )
        patch_summary = jt.concat(
            [
                displacement_length.mean(dim=1, keepdims=True),
                displacement_length.max(dim=1, keepdims=True),
                consensus_length.mean(dim=1, keepdims=True),
                radius.mean(dim=1, keepdims=True),
                nonplanarity.mean(dim=1, keepdims=True),
            ],
            dim=-1,
        )
        patch_summary = patch_summary.broadcast(
            (coarse.shape[0], coarse.shape[1], 5)
        )
        point_input = jt.concat(
            [
                coarse,
                stage1_displacement,
                consensus_delta,
                local_mean,
                normal,
                radius,
                nonplanarity,
                patch_distance,
                neighbor_feature,
                patch_summary,
            ],
            dim=-1,
        )
        feature = apply_point_mlp(self.point_mlp, point_input)
        direction = apply_point_mlp(self.direction_head, feature)
        direction = direction / (
            jt.sqrt((direction ** 2.0).sum(dim=-1, keepdims=True))
            + self.eps
        )
        length = (
            jt.sigmoid(apply_point_mlp(self.length_head, feature))
            * self.max_residual
        )
        gate = jt.sigmoid(apply_point_mlp(self.gate_head, feature))
        raw_residual = direction * length
        residual = raw_residual * gate
        return coarse + residual, {
            "residual": residual,
            "raw_residual": raw_residual,
            "direction": direction,
            "length": length,
            "gate": gate,
            "normal": normal,
            "nonplanarity": nonplanarity,
        }
