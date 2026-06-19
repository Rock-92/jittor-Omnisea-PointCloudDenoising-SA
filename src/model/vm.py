from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .edgeconv_baseline import EdgeConvFeatureExtraction
from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    if m is None or m <= 0 or m >= n:
        return None
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class VelocityModule(ModelSpec):
    
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        
        cfg = self.model_config
        # geometry
        self.attention_knn = cfg.get('attention_knn', cfg.get('frame_knn', 16))
        self.input_dim = cfg.get('input_dim', 3)
        self.input_expand_dim = cfg.get('input_expand_dim', 128)
        self.feat_embedding_dim = cfg['feat_embedding_dim']
        self.attention_blocks = cfg.get('attention_blocks', 4)
        self.attention_weight_init = cfg.get('attention_weight_init', 1.0)
        self.attention_ffn_hidden_dim = cfg.get(
            'attention_ffn_hidden_dim',
            self.feat_embedding_dim * 2,
        )
        self.global_token_blocks = cfg.get('global_token_blocks', 4)
        self.global_token_ffn_hidden_dim = cfg.get(
            'global_token_ffn_hidden_dim',
            self.feat_embedding_dim * 2,
        )
        self.relative_position_bias_hidden_dim = cfg.get(
            'relative_position_bias_hidden_dim',
            None,
        )
        self.global_attn_bias_init = float(
            cfg.get('global_attn_bias_init', 1.0)
        )
        self.legacy_graph_updates = bool(
            cfg.get('legacy_graph_updates', False)
        )
        self.decoder_hidden_dims = cfg.get(
            'decoder_hidden_dims',
            [cfg.get('decoder_hidden_dim', 64)],
        )
        self.use_edgeconv_branch = cfg.get('use_edgeconv_branch', False)
        self.edgeconv_branch_k = int(cfg.get('edgeconv_branch_k', 16))
        self.edgeconv_branch_dim = int(cfg.get('edgeconv_branch_dim', 128))
        self.use_hard_aware_loss = cfg.get('use_hard_aware_loss', False)
        self.hard_relative_sigma_ref = float(cfg.get('hard_relative_sigma_ref', 0.18))
        self.hard_weight_scale = float(cfg.get('hard_weight_scale', 1.5))
        self.hard_weight_max = float(cfg.get('hard_weight_max', 3.0))
        self.use_geometry_hard_weight = cfg.get('use_geometry_hard_weight', True)
        self.hard_geometry_k = int(cfg.get('hard_geometry_k', 24))
        self.hard_geometry_top_fraction = float(
            cfg.get('hard_geometry_top_fraction', 0.25)
        )
        self.hard_geometry_ref = float(cfg.get('hard_geometry_ref', 0.45))
        self.hard_geometry_weight_scale = float(
            cfg.get('hard_geometry_weight_scale', 1.5)
        )
        self.use_length_projection_loss = cfg.get(
            'use_length_projection_loss',
            True,
        )
        self.length_projection_over_weight = float(
            cfg.get('length_projection_over_weight', 0.25)
        )
        self.hard_chamfer_weight = float(cfg.get('hard_chamfer_weight', 0.05))
        self.hard_normal_weight = float(cfg.get('hard_normal_weight', 0.02))
        self.use_surface_aligned_loss = cfg.get(
            'use_surface_aligned_loss',
            False,
        )
        self.surface_normal_k = int(cfg.get('surface_normal_k', 3))
        self.use_surface_snap_loss = bool(
            cfg.get('use_surface_snap_loss', False)
        )
        self.surface_snap_tau = float(cfg.get('surface_snap_tau', 0.001))
        self.surface_snap_weight = float(cfg.get('surface_snap_weight', 3.0))
        self.use_surface_coherence_loss = bool(
            cfg.get('use_surface_coherence_loss', False)
        )
        self.use_surface_distribution_loss = bool(
            cfg.get('use_surface_distribution_loss', False)
        )
        self.surface_distribution_repulsion_weight = float(
            cfg.get('surface_distribution_repulsion_weight', 0.2)
        )
        self.surface_coherence_k = int(cfg.get('surface_coherence_k', 8))
        self.surface_coherence_cos = float(
            cfg.get('surface_coherence_cos', 0.5)
        )
        self.surface_coherence_center_cos = float(
            cfg.get('surface_coherence_center_cos', 0.5)
        )
        self.surface_outlier_margin = float(
            cfg.get('surface_outlier_margin', self.surface_snap_tau)
        )
        self.use_surface_branch_loss = bool(
            cfg.get('use_surface_branch_loss', False)
        )
        self.surface_branch_tau = float(
            cfg.get('surface_branch_tau', self.surface_snap_tau)
        )
        self.surface_branch_min_valid = float(
            cfg.get('surface_branch_min_valid', 0.5)
        )
        self.surface_branch_separation_k = int(
            cfg.get('surface_branch_separation_k', 8)
        )
        self.surface_branch_separation_margin = float(
            cfg.get('surface_branch_separation_margin', 0.8)
        )
        
        # patch-based prediction
        self.predict_rounds = cfg.get('predict_rounds', 1)
        self.denoise_num_steps = cfg.get('denoise_num_steps', 1)
        self.predict_patch_size = cfg.get('predict_patch_size', 1000)
        self.predict_seed_k = cfg.get('predict_seed_k', 6)
        self.predict_seed_interval = cfg.get('predict_seed_interval', 200)
        self.predict_seed_k_alpha = cfg.get('predict_seed_k_alpha', 1)
        self.patch_fusion_mode = cfg.get('patch_fusion_mode', 'distance_weighted')
        self.patch_fusion_tau = float(cfg.get('patch_fusion_tau', 2.0))
        
        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']
        self.num_train_points = cfg.get('num_train_points', 0)
        self.use_edm = cfg.get('use_edm', False)
        self.sigma_data = float(cfg.get('sigma_data', 0.10))
        self.edm_default_sigma = float(cfg.get('edm_default_sigma', self.dsm_sigma))
        self.edm_loss_weighting = cfg.get('edm_loss_weighting', True)
        self.noise_embedding_dim = cfg.get('noise_embedding_dim', None)
        self.use_patch_scale_condition = cfg.get('use_patch_scale_condition', False)
        self.patch_scale_eps = float(cfg.get('patch_scale_eps', 1e-4))
        self.use_sigma_head = cfg.get('use_sigma_head', False)
        self.sigma_head_hidden_dim = cfg.get(
            'sigma_head_hidden_dim',
            self.feat_embedding_dim // 2,
        )
        self.edm_sampler = cfg.get('edm_sampler', 'alpha_refine')
        self.edm_sigma_min = float(cfg.get('edm_sigma_min', 1e-4))
        self.edm_sigma_max = float(cfg.get('edm_sigma_max', 0.025))
        self.edm_inference_sigmas = cfg.get(
            'edm_inference_sigmas',
            [self.edm_default_sigma],
        )
        self.edm_inference_alphas = cfg.get(
            'edm_inference_alphas',
            [1.0] * len(self.edm_inference_sigmas),
        )
        if (
            self.edm_sampler == 'alpha_refine'
            and len(self.edm_inference_alphas) != len(self.edm_inference_sigmas)
        ):
            raise ValueError("edm_inference_alphas length must match edm_inference_sigmas")
        
        # networks
        if self.use_edm and self.noise_embedding_dim is None:
            self.noise_embedding_dim = self.feat_embedding_dim
        if self.use_edm:
            self.noise_embed_1 = nn.Linear(1, self.noise_embedding_dim)
            self.noise_embed_2 = nn.Linear(self.noise_embedding_dim, self.noise_embedding_dim)
            self.noise_act = nn.ReLU()

        self.encoder = FeatureExtraction(
            knn_scales=self.attention_knn,
            input_dim=self.input_dim,
            input_expand_dim=self.input_expand_dim,
            embedding_dim=self.feat_embedding_dim,
            num_blocks=self.attention_blocks,
            attention_weight_init=self.attention_weight_init,
            ffn_hidden_dim=self.attention_ffn_hidden_dim,
            global_token_blocks=self.global_token_blocks,
            global_token_ffn_hidden_dim=self.global_token_ffn_hidden_dim,
            noise_embedding_dim=self.noise_embedding_dim if self.use_edm else None,
            relative_position_bias_hidden_dim=(
                self.relative_position_bias_hidden_dim
            ),
            global_attn_bias_init=self.global_attn_bias_init,
            legacy_graph_updates=self.legacy_graph_updates,
        )
        if self.use_edgeconv_branch:
            self.edgeconv_branch = EdgeConvFeatureExtraction(
                k=self.edgeconv_branch_k,
                input_dim=self.input_dim,
                embedding_dim=self.edgeconv_branch_dim,
            )
        
        decoder_input_dim = self.encoder.embedding_dim
        if self.use_edgeconv_branch:
            decoder_input_dim += self.edgeconv_branch_dim
        self.decoder = Decoder(
            z_dim=decoder_input_dim,
            out_dim=3,
            hidden_dims=self.decoder_hidden_dims,
        )
        if self.use_sigma_head:
            self.sigma_head_1 = nn.Linear(
                self.encoder.embedding_dim,
                self.sigma_head_hidden_dim,
            )
            self.sigma_head_2 = nn.Linear(self.sigma_head_hidden_dim, 1)
    
    def _expand_sigma(self, sigma, batch_size):
        if sigma is None:
            sigma = self.edm_default_sigma
        if not isinstance(sigma, jt.Var):
            sigma = jt.ones((batch_size, 1)) * float(sigma)
        elif len(sigma.shape) == 0:
            sigma = jt.ones((batch_size, 1)) * sigma
        elif len(sigma.shape) == 1:
            sigma = sigma.reshape(-1, 1)
        return sigma

    def clamp_edm_sigma(self, sigma):
        return jt.maximum(sigma, self.edm_sigma_min)

    def get_patch_scale(self, pc_noisy):
        center = pc_noisy.mean(dim=1).unsqueeze(1)
        centered = pc_noisy - center
        scale2 = (centered ** 2.0).sum(dim=-1).mean(dim=1)
        scale = jt.sqrt(scale2 + self.patch_scale_eps ** 2.0).reshape(pc_noisy.shape[0], 1)
        return jt.maximum(scale, self.patch_scale_eps)

    def get_noise_embedding(self, sigma, patch_scale=None):
        if not self.use_edm:
            return None
        sigma = self.clamp_edm_sigma(sigma)
        sigma_for_condition = sigma
        if self.use_patch_scale_condition and patch_scale is not None:
            sigma_for_condition = sigma / jt.maximum(patch_scale, self.patch_scale_eps)
            sigma_for_condition = self.clamp_edm_sigma(sigma_for_condition)
        c_noise = jt.log(sigma_for_condition) / 4.0
        emb = self.noise_embed_1(c_noise)
        emb = self.noise_act(emb)
        return self.noise_embed_2(emb)

    def get_edm_coefficients(self, sigma):
        sigma = self.clamp_edm_sigma(sigma)
        sigma2 = sigma ** 2.0
        sigma_data2 = self.sigma_data ** 2.0
        denom = sigma2 + sigma_data2
        c_skip = sigma_data2 / denom
        c_out = sigma * self.sigma_data / jt.sqrt(denom)
        c_in = 1.0 / jt.sqrt(denom)
        return c_skip, c_out, c_in

    def get_patch_geometry_difficulty(self, pc_clean, pc_anchor=None):
        """
        Estimate local non-planarity from clean supervision. The normalized
        covariance determinant is near zero for locally planar neighborhoods
        and grows for curved or volumetric geometry.
        """
        if pc_anchor is None:
            pc_anchor = pc_clean
        k = min(self.hard_geometry_k, pc_clean.shape[1])
        if k < 4:
            return jt.zeros((pc_clean.shape[0], 1))

        dist = ((pc_anchor.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist, k=k, dim=-1, largest=False)
        neighbors = []
        for b in range(pc_clean.shape[0]):
            neighbors.append(pc_clean[b][idx[b]])
        neighbors = jt.stack(neighbors, dim=0)
        centered = neighbors - neighbors.mean(dim=2).unsqueeze(2)

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
            self.patch_scale_eps ** 6.0,
        )
        nonplanarity = jt.minimum(jt.maximum(nonplanarity, 0.0), 1.0)

        top_count = max(
            1,
            int(round(pc_anchor.shape[1] * self.hard_geometry_top_fraction)),
        )
        top_count = min(top_count, pc_anchor.shape[1])
        top_values, _ = jt.topk(
            nonplanarity,
            k=top_count,
            dim=1,
            largest=True,
        )
        return top_values.mean(dim=1).reshape(pc_clean.shape[0], 1)

    def get_hard_patch_weight(self, pc_noisy, sigma, pc_clean=None, pc_anchor=None):
        if not self.use_hard_aware_loss:
            return None
        patch_scale = self.get_patch_scale(pc_noisy)
        relative_sigma = self.clamp_edm_sigma(sigma) / jt.maximum(
            patch_scale,
            self.patch_scale_eps,
        )
        hard = jt.maximum(relative_sigma / self.hard_relative_sigma_ref - 1.0, 0.0)
        scale_weight = 1.0 + self.hard_weight_scale * hard
        weight = scale_weight
        if self.use_geometry_hard_weight and pc_clean is not None:
            geometry = self.get_patch_geometry_difficulty(
                pc_clean=pc_clean,
                pc_anchor=pc_anchor,
            )
            geometry_hard = jt.maximum(
                geometry / self.hard_geometry_ref - 1.0,
                0.0,
            )
            geometry_weight = 1.0 + self.hard_geometry_weight_scale * geometry_hard
            weight = jt.maximum(scale_weight, geometry_weight)
        return jt.minimum(weight, self.hard_weight_max)

    def _loss_sigma2(self, sigma, batch_size):
        if sigma is None:
            sigma = jt.ones((batch_size, 1)) * float(self.dsm_sigma)
        else:
            sigma = self._expand_sigma(sigma, batch_size)
        sigma = self.clamp_edm_sigma(sigma)
        return jt.maximum(
            sigma.reshape(batch_size) ** 2.0,
            self.patch_scale_eps ** 2.0,
        )

    def get_patch_chamfer_components(
        self,
        pc_pred,
        pc_clean,
        sigma=None,
        hard_weight=None,
    ):
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        pred_to_clean = dist.min(dim=2)
        clean_to_pred = dist.min(dim=1)
        sigma2 = self._loss_sigma2(sigma, pc_pred.shape[0])
        pred_loss = pred_to_clean.mean(dim=1) / sigma2
        cover_loss = clean_to_pred.mean(dim=1) / sigma2
        if hard_weight is not None:
            weight = hard_weight.reshape(pc_pred.shape[0])
            pred_loss = pred_loss * weight
            cover_loss = cover_loss * weight
        return pred_loss.mean(), cover_loss.mean()

    def get_patch_chamfer_loss(self, pc_pred, pc_clean, sigma=None, hard_weight=None):
        pred_loss, cover_loss = self.get_patch_chamfer_components(
            pc_pred=pc_pred,
            pc_clean=pc_clean,
            sigma=sigma,
            hard_weight=hard_weight,
        )
        return pred_loss + cover_loss

    def get_surface_distribution_loss(
        self,
        pc_pred,
        pc_clean,
        sigma=None,
        hard_weight=None,
    ):
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        clean_to_pred = dist.min(dim=1)

        clean_pair = ((pc_clean.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        clean_nn, _ = jt.topk(clean_pair, k=2, dim=-1, largest=False)
        clean_spacing = jt.sqrt(clean_nn[:, :, 1] + self.patch_scale_eps ** 2.0)
        target_spacing = clean_spacing.mean(dim=1, keepdims=True)

        pred_pair = ((pc_pred.unsqueeze(2) - pc_pred.unsqueeze(1)) ** 2.0).sum(dim=-1)
        pred_nn, _ = jt.topk(pred_pair, k=2, dim=-1, largest=False)
        pred_spacing = jt.sqrt(pred_nn[:, :, 1] + self.patch_scale_eps ** 2.0)
        repulsion = jt.maximum(target_spacing - pred_spacing, 0.0) ** 2.0

        sigma2 = self._loss_sigma2(sigma, pc_pred.shape[0])
        coverage_loss = clean_to_pred.mean(dim=1) / sigma2
        repulsion_loss = repulsion.mean(dim=1) / sigma2
        loss = coverage_loss + self.surface_distribution_repulsion_weight * repulsion_loss
        if hard_weight is not None:
            loss = loss * hard_weight.reshape(pc_pred.shape[0])
        return loss.mean()

    def get_clean_point_normals(self, pc_clean):
        k = min(max(self.surface_normal_k, 3), pc_clean.shape[1])
        dist = ((pc_clean.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist, k=k, dim=-1, largest=False)
        neighbors = []
        for b in range(pc_clean.shape[0]):
            neighbors.append(pc_clean[b][idx[b]])
        neighbors = jt.stack(neighbors, dim=0)

        p0 = pc_clean
        p1 = neighbors[:, :, 1, :]
        p2 = neighbors[:, :, 2, :]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = jt.stack(
            [
                v1[:, :, 1] * v2[:, :, 2] - v1[:, :, 2] * v2[:, :, 1],
                v1[:, :, 2] * v2[:, :, 0] - v1[:, :, 0] * v2[:, :, 2],
                v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0],
            ],
            dim=-1,
        )
        normal = normal / jt.sqrt(
            (normal ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
        )
        return normal

    def get_nearest_surface_geometry(
        self,
        pc_pred,
        pc_clean,
    ):
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist, k=1, dim=2, largest=False)
        clean_normals = self.get_clean_point_normals(pc_clean)
        anchors = []
        normals = []
        for b in range(pc_pred.shape[0]):
            nearest_idx = idx[b].reshape(-1)
            anchors.append(pc_clean[b][nearest_idx])
            normals.append(clean_normals[b][nearest_idx])
        anchors = jt.stack(anchors, dim=0)
        normals = jt.stack(normals, dim=0)
        signed_plane_dist = ((pc_pred - anchors) * normals).sum(dim=-1)
        return anchors, normals, signed_plane_dist

    def get_nearest_surface_loss(
        self,
        pc_pred,
        pc_clean,
        sigma=None,
        hard_weight=None,
    ):
        _, _, signed_plane_dist = self.get_nearest_surface_geometry(
            pc_pred=pc_pred,
            pc_clean=pc_clean,
        )
        plane_dist = signed_plane_dist ** 2.0
        if self.use_surface_snap_loss and self.surface_snap_weight > 0.0:
            tau = max(self.surface_snap_tau, self.patch_scale_eps)
            tau2 = tau ** 2.0
            plane_dist = plane_dist + self.surface_snap_weight * tau2 * jt.atan(
                plane_dist / tau2
            )
        sigma2 = self._loss_sigma2(sigma, pc_pred.shape[0])
        loss = plane_dist.mean(dim=1) / sigma2
        if hard_weight is not None:
            loss = loss * hard_weight.reshape(pc_pred.shape[0])
        return loss.mean()

    def get_surface_coherence_losses(
        self,
        pc_pred,
        pc_noisy,
        pc_clean,
        sigma=None,
        hard_weight=None,
    ):
        if pc_pred.shape[1] <= 1:
            zero = jt.array(0.0)
            return zero, zero

        k = min(max(self.surface_coherence_k, 1) + 1, pc_pred.shape[1])
        _, normals, signed_plane_dist = self.get_nearest_surface_geometry(
            pc_pred=pc_pred,
            pc_clean=pc_clean,
        )
        dist = ((pc_noisy.unsqueeze(2) - pc_noisy.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist, k=k, dim=-1, largest=False)
        idx = idx[:, :, 1:]

        neigh_pred = []
        neigh_noisy = []
        neigh_normals = []
        for b in range(pc_pred.shape[0]):
            neigh_idx = idx[b]
            neigh_pred.append(pc_pred[b][neigh_idx])
            neigh_noisy.append(pc_noisy[b][neigh_idx])
            neigh_normals.append(normals[b][neigh_idx])
        neigh_pred = jt.stack(neigh_pred, dim=0)
        neigh_noisy = jt.stack(neigh_noisy, dim=0)
        neigh_normals = jt.stack(neigh_normals, dim=0)

        pred_disp = pc_pred - pc_noisy
        pred_len = jt.sqrt((pred_disp ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
        pred_dir = pred_disp / pred_len
        neigh_disp = neigh_pred - neigh_noisy
        neigh_len = jt.sqrt((neigh_disp ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
        neigh_dir = neigh_disp / neigh_len
        dir_i = pred_dir.unsqueeze(2)
        cos = (dir_i * neigh_dir).sum(dim=-1)

        same_forward = jt.where(
            cos > self.surface_coherence_cos,
            jt.ones_like(cos),
            jt.zeros_like(cos),
        )
        pair_vec = neigh_noisy - pc_noisy.unsqueeze(2)
        pair_len = jt.sqrt((pair_vec ** 2.0).sum(dim=-1, keepdims=True) + 1e-8)
        pair_dir = pair_vec / pair_len
        inward_i = (dir_i * pair_dir).sum(dim=-1)
        inward_j = (neigh_dir * (-pair_dir)).sum(dim=-1)
        same_reverse = jt.where(
            (cos < -self.surface_coherence_cos)
            & (inward_i > self.surface_coherence_center_cos)
            & (inward_j > self.surface_coherence_center_cos),
            jt.ones_like(cos),
            jt.zeros_like(cos),
        )
        same_surface = jt.minimum(same_forward + same_reverse, 1.0)

        avg_normal = normals.unsqueeze(2) + neigh_normals
        avg_normal = avg_normal / jt.sqrt(
            (avg_normal ** 2.0).sum(dim=-1, keepdims=True) + 1e-8
        )
        normal_sep = ((neigh_pred - pc_pred.unsqueeze(2)) * avg_normal).sum(dim=-1)
        denom = same_surface.sum(dim=-1) + 1e-6
        coherence_point = ((normal_sep ** 2.0) * same_surface).sum(dim=-1) / denom

        margin = max(self.surface_outlier_margin, self.patch_scale_eps)
        plane_distance = jt.sqrt(signed_plane_dist ** 2.0 + 1e-12)
        far_surface = jt.maximum(plane_distance - margin, 0.0) ** 2.0
        grouped = jt.where(
            same_surface.sum(dim=-1) > 0.5,
            jt.ones_like(plane_distance),
            jt.zeros_like(plane_distance),
        )
        outlier_point = far_surface * (1.0 - grouped)

        sigma2 = self._loss_sigma2(sigma, pc_pred.shape[0])
        coherence_loss = coherence_point.mean(dim=1) / sigma2
        outlier_loss = outlier_point.mean(dim=1) / sigma2
        if hard_weight is not None:
            hard_weight = hard_weight.reshape(pc_pred.shape[0])
            coherence_loss = coherence_loss * hard_weight
            outlier_loss = outlier_loss * hard_weight
        return coherence_loss.mean(), outlier_loss.mean()

    def get_surface_branch_losses(
        self,
        pc_pred,
        pc_clean,
        branch_label,
        branch_valid,
        branch_normal,
        sigma=None,
    ):
        valid = branch_valid
        if len(valid.shape) == 3:
            valid = valid.squeeze(-1)
        valid_sum = valid.sum(dim=1)
        active = jt.where(
            valid_sum >= self.surface_branch_min_valid * pc_pred.shape[1],
            jt.ones_like(valid_sum),
            jt.zeros_like(valid_sum),
        )

        signed = ((pc_pred - pc_clean) * branch_normal).sum(dim=-1)
        abs_dist = jt.sqrt(signed ** 2.0 + self.patch_scale_eps ** 2.0) - self.patch_scale_eps
        tau = max(self.surface_branch_tau, self.patch_scale_eps)
        snap_point = tau * jt.log(1.0 + abs_dist / tau)
        snap_loss = (snap_point * valid).sum(dim=1) / (valid_sum + 1e-6)

        k = min(max(self.surface_branch_separation_k, 1) + 1, pc_clean.shape[1])
        dist = ((pc_clean.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist, k=k, dim=-1, largest=False)
        idx = idx[:, :, 1:]
        neigh_clean = []
        neigh_pred = []
        neigh_label = []
        neigh_valid = []
        for b in range(pc_clean.shape[0]):
            neigh_idx = idx[b]
            neigh_clean.append(pc_clean[b][neigh_idx])
            neigh_pred.append(pc_pred[b][neigh_idx])
            neigh_label.append(branch_label[b][neigh_idx])
            neigh_valid.append(valid[b][neigh_idx])
        neigh_clean = jt.stack(neigh_clean, dim=0)
        neigh_pred = jt.stack(neigh_pred, dim=0)
        neigh_label = jt.stack(neigh_label, dim=0)
        neigh_valid = jt.stack(neigh_valid, dim=0)

        pair_valid = valid.unsqueeze(2) * neigh_valid
        different_branch = jt.where(
            branch_label.unsqueeze(2) != neigh_label,
            jt.ones_like(pair_valid),
            jt.zeros_like(pair_valid),
        )
        pair_mask = pair_valid * different_branch
        normal = branch_normal.unsqueeze(2)
        clean_gap = jt.abs(((neigh_clean - pc_clean.unsqueeze(2)) * normal).sum(dim=-1))
        pred_gap = jt.abs(((neigh_pred - pc_pred.unsqueeze(2)) * normal).sum(dim=-1))
        target_gap = self.surface_branch_separation_margin * clean_gap
        sep_point = jt.maximum(target_gap - pred_gap, 0.0) ** 2.0
        sep_flat = (sep_point * pair_mask).reshape(pc_pred.shape[0], -1)
        pair_flat = pair_mask.reshape(pc_pred.shape[0], -1)
        sep_loss = sep_flat.sum(dim=1) / (pair_flat.sum(dim=1) + 1e-6)

        sigma2 = self._loss_sigma2(sigma, pc_pred.shape[0])
        snap_loss = active * snap_loss / sigma2
        sep_loss = active * sep_loss / sigma2
        active_count = active.sum() + 1e-6
        return snap_loss.sum() / active_count, sep_loss.sum() / active_count

    def get_surface_aligned_losses(
        self,
        pc_pred,
        pc_clean,
        pc_noisy=None,
        sigma=None,
        hard_weight=None,
    ):
        nearest_clean_loss, clean_coverage_loss = self.get_patch_chamfer_components(
            pc_pred=pc_pred,
            pc_clean=pc_clean,
            sigma=sigma,
            hard_weight=hard_weight,
        )
        patch_chamfer_loss = nearest_clean_loss + clean_coverage_loss
        nearest_surface_loss = self.get_nearest_surface_loss(
            pc_pred=pc_pred,
            pc_clean=pc_clean,
            sigma=sigma,
            hard_weight=hard_weight,
        )
        losses = {
            "nearest_clean_loss": nearest_clean_loss,
            "clean_coverage_loss": clean_coverage_loss,
            "patch_chamfer_loss": patch_chamfer_loss,
            "nearest_surface_loss": nearest_surface_loss,
        }
        if self.use_surface_coherence_loss and pc_noisy is not None:
            surface_coherence_loss, surface_outlier_loss = (
                self.get_surface_coherence_losses(
                    pc_pred=pc_pred,
                    pc_noisy=pc_noisy,
                    pc_clean=pc_clean,
                    sigma=sigma,
                    hard_weight=hard_weight,
                )
            )
            losses["surface_coherence_loss"] = surface_coherence_loss
            losses["surface_outlier_loss"] = surface_outlier_loss
        if self.use_surface_distribution_loss:
            losses["surface_distribution_loss"] = self.get_surface_distribution_loss(
                pc_pred=pc_pred,
                pc_clean=pc_clean,
                sigma=sigma,
                hard_weight=hard_weight,
            )
        return losses

    def get_length_projection_loss(
        self,
        pc_pred,
        pc_noisy,
        pc_clean,
        sigma,
        hard_weight=None,
    ):
        target_disp = pc_clean - pc_noisy
        pred_disp = pc_pred - pc_noisy
        target_length = jt.sqrt(
            (target_disp ** 2.0).sum(dim=-1) + self.patch_scale_eps ** 2.0
        )
        target_direction = target_disp / target_length.unsqueeze(-1)
        projected_length = (pred_disp * target_direction).sum(dim=-1)

        length_error = target_length - projected_length
        under_error = jt.maximum(length_error, 0.0)
        over_error = jt.maximum(-length_error, 0.0)
        point_loss = (
            under_error ** 2.0
            + self.length_projection_over_weight * over_error ** 2.0
        )
        sigma2 = self.clamp_edm_sigma(sigma).reshape(pc_pred.shape[0]) ** 2.0
        patch_loss = point_loss.mean(dim=1) / sigma2
        if hard_weight is not None:
            patch_loss = patch_loss * hard_weight.reshape(pc_pred.shape[0])
        return patch_loss.mean()

    def encode_features(self, pc_noisy, sigma=None, point_idx=None):
        """
        pc_noisy: (B, N, 3)
        sigma: optional (B, 1) noise level for EDM FiLM
        point_idx: optional point indices decoded after full-patch encoding
        return:   (B, N, 3) or (B, M, 3)
        """
        B, N, d = pc_noisy.shape
        edgeconv_x = pc_noisy
        noise_emb = None
        global_x = None
        if self.use_edm:
            sigma = self._expand_sigma(sigma, B)
            _, _, c_in = self.get_edm_coefficients(sigma)
            global_x = pc_noisy
            patch_scale = self.get_patch_scale(pc_noisy)
            pc_noisy = pc_noisy * c_in.reshape(B, 1, 1)
            noise_emb = self.get_noise_embedding(sigma, patch_scale=patch_scale)
        feat = self.encoder(
            pc_noisy,
            noise_emb=noise_emb,
            global_x=global_x,
        )  # (B, N, 256)
        if self.use_edgeconv_branch:
            edge_feat = self.edgeconv_branch(edgeconv_x)
            feat = jt.concat([feat, edge_feat], dim=-1)
        if point_idx is not None:
            feat = feat[:, point_idx, :]
        N_out = feat.shape[1]
        F_dim = feat.shape[2]
        return feat

    def predict_raw(self, pc_noisy, sigma=None, point_idx=None):
        B, _, d = pc_noisy.shape
        feat = self.encode_features(pc_noisy, sigma=sigma, point_idx=point_idx)
        N_out = feat.shape[1]
        F_dim = feat.shape[2]
        return self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, d)

    def predict_confidence(self, pc_noisy, sigma=None, point_idx=None):
        B = pc_noisy.shape[0]
        feat = self.encode_features(pc_noisy, sigma=sigma, point_idx=point_idx)
        return jt.ones((B, feat.shape[1], 1))

    def predict_sigma(self, pc_noisy):
        B = pc_noisy.shape[0]
        if not self.use_sigma_head:
            return jt.ones((B, 1)) * self.edm_default_sigma
        pooled = self.encoder.get_global_token(pc_noisy).reshape(B, -1)
        hidden = self.noise_act(self.sigma_head_1(pooled))
        raw = self.sigma_head_2(hidden)
        sigma = self.edm_sigma_min + (
            self.edm_sigma_max - self.edm_sigma_min
        ) * jt.sigmoid(raw)
        return sigma

    def predict_clean(self, pc_noisy, sigma=None, point_idx=None):
        if not self.use_edm:
            return pc_noisy + self.predict_raw(pc_noisy, point_idx=point_idx)
        B = pc_noisy.shape[0]
        sigma = self._expand_sigma(sigma, B)
        feat = self.encode_features(pc_noisy, sigma=sigma, point_idx=point_idx)
        N_out = feat.shape[1]
        F_dim = feat.shape[2]
        raw = self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, pc_noisy.shape[2])
        pc_base = pc_noisy
        if point_idx is not None:
            pc_base = pc_noisy[:, point_idx, :]
        c_skip, c_out, _ = self.get_edm_coefficients(sigma)
        pc_pred = (
            c_skip.reshape(B, 1, 1) * pc_base
            + c_out.reshape(B, 1, 1) * raw
        )
        return pc_pred

    def predict_displacement(self, pc_noisy, sigma=None, point_idx=None):
        clean = self.predict_clean(pc_noisy, sigma=sigma, point_idx=point_idx)
        pc_base = pc_noisy
        if point_idx is not None:
            pc_base = pc_noisy[:, point_idx, :]
        return clean - pc_base
    
    def get_normalized_surface_loss(
        self,
        pc_pred,
        pc_clean,
        pc_anchor,
        reduction="mean",
    ):
        """
        Penalize point-to-local-plane distance. Each plane is estimated around
        the paired clean supervision point.
        """
        dist = ((pc_anchor.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist, k=3, dim=-1, largest=False)
        neighbors = []
        for b in range(pc_clean.shape[0]):
            neighbors.append(pc_clean[b][idx[b]])
        neighbors = jt.stack(neighbors, dim=0)

        p0 = pc_anchor
        p1 = neighbors[:, :, 1, :]
        p2 = neighbors[:, :, 2, :]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = jt.stack(
            [
                v1[:, :, 1] * v2[:, :, 2] - v1[:, :, 2] * v2[:, :, 1],
                v1[:, :, 2] * v2[:, :, 0] - v1[:, :, 0] * v2[:, :, 2],
                v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0],
            ],
            dim=-1,
        )
        normal = normal / (((normal ** 2.0).sum(dim=-1) + 1e-8) ** 0.5).unsqueeze(-1)
        plane_dist = (((pc_pred - p0) * normal).sum(dim=-1) ** 2.0)
        loss = (plane_dist / self.dsm_sigma).mean(dim=1)
        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError(f"unsupported surface loss reduction: {reduction}")
        return loss.mean()

    def get_supervised_losses(
        self,
        pc_noisy,
        pc_clean,
        score_sigma=None,
        branch_label=None,
        branch_valid=None,
        branch_normal=None,
    ):
        """
        pc_noisy: (B, N, 3)
        pc_clean: (B, N, 3)
        """
        B = pc_noisy.shape[0]
        score_sigma = self._expand_sigma(score_sigma, B)
        target = pc_clean - pc_noisy
        point_idx = None
        if not self.use_surface_aligned_loss:
            point_idx = get_random_indices(pc_noisy.shape[1], self.num_train_points)
        pc_noisy_for_loss = pc_noisy
        pc_clean_for_loss = pc_clean
        if point_idx is not None:
            target = target[:, point_idx, :]
            pc_noisy_for_loss = pc_noisy[:, point_idx, :]
            pc_clean_for_loss = pc_clean[:, point_idx, :]
            if branch_label is not None:
                branch_label = branch_label[:, point_idx]
            if branch_valid is not None:
                branch_valid = branch_valid[:, point_idx]
            if branch_normal is not None:
                branch_normal = branch_normal[:, point_idx, :]
        if self.use_edm:
            sigma_pred = self.predict_sigma(pc_noisy)
            pc_pred = self.predict_clean(
                pc_noisy,
                sigma=score_sigma,
                point_idx=point_idx,
            )
            mse = ((pc_pred - pc_clean_for_loss) ** 2.0).sum(dim=-1)
            hard_weight = self.get_hard_patch_weight(
                pc_noisy=pc_noisy,
                sigma=score_sigma,
                pc_clean=pc_clean,
                pc_anchor=pc_clean_for_loss,
            )
            if self.edm_loss_weighting:
                sigma_for_weight = self.clamp_edm_sigma(score_sigma)
                sigma2 = sigma_for_weight ** 2.0
                sigma_data2 = self.sigma_data ** 2.0
                weight = (sigma2 + sigma_data2) / (sigma2 * sigma_data2)
                mse = mse * weight.reshape(B, 1)
            if hard_weight is not None:
                mse = mse * hard_weight.reshape(B, 1)
            displacement_loss = mse.mean()
            losses = {
                "displacement_loss": displacement_loss,
            }
            if self.use_surface_aligned_loss:
                losses.update(
                    self.get_surface_aligned_losses(
                        pc_pred=pc_pred,
                        pc_noisy=pc_noisy_for_loss,
                        pc_clean=pc_clean_for_loss,
                        sigma=score_sigma,
                        hard_weight=hard_weight,
                    )
                )
            if (
                self.use_surface_branch_loss
                and branch_label is not None
                and branch_valid is not None
                and branch_normal is not None
            ):
                branch_snap_loss, branch_separation_loss = (
                    self.get_surface_branch_losses(
                        pc_pred=pc_pred,
                        pc_clean=pc_clean_for_loss,
                        branch_label=branch_label,
                        branch_valid=branch_valid,
                        branch_normal=branch_normal,
                        sigma=score_sigma,
                    )
                )
                losses["branch_snap_loss"] = branch_snap_loss
                losses["branch_separation_loss"] = branch_separation_loss
            if self.use_hard_aware_loss and self.hard_chamfer_weight > 0:
                losses["patch_chamfer_loss"] = self.get_patch_chamfer_loss(
                    pc_pred=pc_pred,
                    pc_clean=pc_clean_for_loss,
                    sigma=score_sigma,
                    hard_weight=hard_weight,
                )
            if self.use_hard_aware_loss and self.hard_normal_weight > 0:
                normal_loss = self.get_normalized_surface_loss(
                    pc_pred=pc_pred,
                    pc_clean=pc_clean,
                    pc_anchor=pc_clean_for_loss,
                    reduction="none",
                )
                if hard_weight is not None:
                    normal_loss = normal_loss * hard_weight.reshape(B)
                losses["hard_normal_loss"] = normal_loss.mean()
            if self.use_length_projection_loss:
                losses["length_projection_loss"] = self.get_length_projection_loss(
                    pc_pred=pc_pred,
                    pc_noisy=pc_noisy_for_loss,
                    pc_clean=pc_clean_for_loss,
                    sigma=score_sigma,
                    hard_weight=hard_weight,
                )
            if self.use_sigma_head:
                sigma_target = self.clamp_edm_sigma(score_sigma)
                sigma_loss = (
                    (jt.log(self.clamp_edm_sigma(sigma_pred)) - jt.log(sigma_target)) ** 2.0
                ).mean()
                losses["sigma_loss"] = sigma_loss
            return losses
        else:
            pred_dir = self.predict_displacement(pc_noisy, point_idx=point_idx)
            displacement_loss = (((pred_dir - target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        normalized_surface_loss = self.get_normalized_surface_loss(
            pc_pred=pc_noisy_for_loss + pred_dir,
            pc_clean=pc_clean,
            pc_anchor=pc_clean_for_loss,
        )

        losses = {
            "displacement_loss": displacement_loss,
            "normalized_surface_loss": normalized_surface_loss,
        }
        if self.use_surface_aligned_loss:
            losses.update(
                self.get_surface_aligned_losses(
                    pc_pred=pc_noisy_for_loss + pred_dir,
                    pc_noisy=pc_noisy_for_loss,
                    pc_clean=pc_clean_for_loss,
                    sigma=score_sigma,
                )
            )
        if (
            self.use_surface_branch_loss
            and branch_label is not None
            and branch_valid is not None
            and branch_normal is not None
        ):
            branch_snap_loss, branch_separation_loss = self.get_surface_branch_losses(
                pc_pred=pc_noisy_for_loss + pred_dir,
                pc_clean=pc_clean_for_loss,
                branch_label=branch_label,
                branch_valid=branch_valid,
                branch_normal=branch_normal,
                sigma=score_sigma,
            )
            losses["branch_snap_loss"] = branch_snap_loss
            losses["branch_separation_loss"] = branch_separation_loss
        return losses

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=None):
        """
        pcl_noisy: (B, N, 3)
        """
        if num_steps is None:
            num_steps = self.denoise_num_steps
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            if self.use_edm:
                if self.edm_sampler == 'heun':
                    pcl_next = self.edm_heun_sampler(pcl_next)
                elif self.edm_sampler == 'euler':
                    pcl_next = self.edm_euler_sampler(pcl_next)
                elif self.edm_sampler == 'alpha_refine':
                    for sigma, alpha in zip(self.edm_inference_sigmas, self.edm_inference_alphas):
                        pc_clean = self.predict_clean(pcl_next, sigma=float(sigma))
                        pcl_next = pcl_next + float(alpha) * (pc_clean - pcl_next)
                else:
                    raise ValueError(f"unsupported edm_sampler: {self.edm_sampler}")
            else:
                for it in range(num_steps):
                    pred_dir = self.predict_displacement(pcl_next)
                    pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None

    def _sigma_to_var(self, sigma, batch_size):
        if isinstance(sigma, jt.Var):
            return self.clamp_edm_sigma(sigma.reshape(batch_size, 1))
        sigma_eval = max(float(sigma), self.edm_sigma_min)
        return jt.ones((batch_size, 1)) * sigma_eval

    def edm_derivative(self, x, sigma):
        sigma_var = self._sigma_to_var(sigma, x.shape[0])
        denoised = self.predict_clean(x, sigma=sigma_var)
        return (x - denoised) / sigma_var.reshape(x.shape[0], 1, 1)

    def get_inference_sigmas(self, x):
        if self.use_sigma_head:
            sigma_start = self.predict_sigma(x)
            return [sigma_start, sigma_start * 0.5, sigma_start * 0.25, 0.0]
        return [float(v) for v in self.edm_inference_sigmas]

    def edm_euler_sampler(self, x):
        sigmas = self.get_inference_sigmas(x)
        if len(sigmas) < 2:
            return self.predict_clean(x, sigma=sigmas[0] if sigmas else self.edm_default_sigma)
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            d = self.edm_derivative(x, sigma)
            step = sigma_next - sigma
            if isinstance(step, jt.Var):
                step = step.reshape(x.shape[0], 1, 1)
            x = x + step * d
        return x

    def edm_heun_sampler(self, x):
        sigmas = self.get_inference_sigmas(x)
        if len(sigmas) < 2:
            return self.predict_clean(x, sigma=sigmas[0] if sigmas else self.edm_default_sigma)
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            d = self.edm_derivative(x, sigma)
            step = sigma_next - sigma
            step_apply = step.reshape(x.shape[0], 1, 1) if isinstance(step, jt.Var) else step
            x_euler = x + step_apply * d
            if not isinstance(sigma_next, jt.Var) and sigma_next <= 0:
                x = x_euler
                continue
            d_next = self.edm_derivative(x_euler, sigma_next)
            x = x + step_apply * 0.5 * (d + d_next)
        return x
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        score_sigma = batch.get('score_sigma')
        if score_sigma is not None:
            score_sigma = score_sigma.reshape(-1, 1)
        branch_label = batch.get('pc_branch_label')
        branch_valid = batch.get('pc_branch_valid')
        branch_normal = batch.get('pc_branch_normal')
        if branch_label is not None:
            branch_label = branch_label.reshape(-1, patch_size)
        if branch_valid is not None:
            branch_valid = branch_valid.reshape(-1, patch_size)
        if branch_normal is not None:
            branch_normal = branch_normal.reshape(-1, patch_size, 3)
        losses = self.get_supervised_losses(
            pc_noisy=pc_noisy,
            pc_clean=pc_clean,
            score_sigma=score_sigma,
            branch_label=branch_label,
            branch_valid=branch_valid,
            branch_normal=branch_normal,
        )
        return losses
    
    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)
    
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(self.predict_rounds):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=self.predict_patch_size,
                    seed_k=self.predict_seed_k,
                    seed_interval=self.predict_seed_interval,
                    seed_k_alpha=self.predict_seed_k_alpha,
                    fusion_mode=self.patch_fusion_mode,
                    fusion_tau=self.patch_fusion_tau,
                )
            pc_denoised = pc_next.detach().numpy()
            res.append({"pc_denoised": pc_denoised})
        return res
    
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                item = {
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                }
                if 'patch_seed' in b.meta:
                    item["patch_seed"] = b.meta['patch_seed']
                if 'score_sigma' in b.meta:
                    item["score_sigma"] = b.meta['score_sigma']
                if 'pc_branch_label' in b.meta:
                    item["pc_branch_label"] = b.meta['pc_branch_label']
                    item["pc_branch_valid"] = b.meta['pc_branch_valid']
                    item["pc_branch_normal"] = b.meta['pc_branch_normal']
                elif self.use_surface_branch_loss:
                    patch_shape = b.meta['pc_clean'].shape[:2]
                    item["pc_branch_label"] = np.zeros(
                        patch_shape,
                        dtype=np.int32,
                    )
                    item["pc_branch_valid"] = np.zeros(
                        patch_shape,
                        dtype=np.float32,
                    )
                    item["pc_branch_normal"] = np.zeros(
                        (*patch_shape, 3),
                        dtype=np.float32,
                    )
                if 'pc_branch_noise_fraction' in b.meta:
                    item["pc_branch_noise_fraction"] = b.meta['pc_branch_noise_fraction']
                elif self.use_surface_branch_loss:
                    item["pc_branch_noise_fraction"] = np.ones(
                        (b.meta['pc_clean'].shape[0], 1),
                        dtype=np.float32,
                    )
                res.append(item)
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def get_interval_seed_indices(n, interval):
    interval = max(1, int(interval))
    seed_idx = np.arange(0, n, interval, dtype=np.int32)
    if seed_idx.size == 0 or seed_idx[-1] != n - 1:
        seed_idx = np.concatenate([seed_idx, np.array([n - 1], dtype=np.int32)])
    return seed_idx

def patch_based_denoise(
    model: VelocityModule,
    pcl_noisy,
    patch_size=1000,
    seed_k=6,
    seed_interval=200,
    seed_k_alpha=1,
    fusion_mode="distance_weighted",
    fusion_tau=2.0,
) -> jt.Var:
    """
    pcl_noisy: (N, 3)
    """
    assert len(pcl_noisy.shape) == 2
    
    N, _ = pcl_noisy.shape
    patch_size = min(int(patch_size), N)
    num_patches = min(N, max(1, int(seed_k * N / patch_size)))
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    
    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)
    
    covered = np.zeros((N,), dtype=np.bool_)
    covered[point_idxs[0].numpy().reshape(-1)] = True
    missing_idx = np.flatnonzero(~covered).astype(np.int32)
    if missing_idx.size > 0:
        extra_seed_idx = jt.array(missing_idx).int32()
        extra_seed_pnts = pcl_noisy[:, extra_seed_idx, :]
        extra_patch_dists, extra_point_idxs, extra_patches = knn_points(
            extra_seed_pnts,
            pcl_noisy,
            patch_size,
        )
        seed_pnts = jt.concat([seed_pnts, extra_seed_pnts], dim=1)
        patch_dists = jt.concat([patch_dists, extra_patch_dists], dim=1)
        point_idxs = jt.concat([point_idxs, extra_point_idxs], dim=1)
        patches = jt.concat([patches, extra_patches], dim=1)
        num_patches += missing_idx.size
        print(
            f"Patch coverage: added {missing_idx.size} extra seed patches "
            "for points missed by FPS seeds."
        )
    
    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)
    
    seed_expand = seed_pnts[0].unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand
    
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    
    patches_denoised = []
    
    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step
    
    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand
    patches_denoised_np = patches_denoised.numpy()
    pcl_noisy_np = pcl_noisy[0].numpy()
    point_idxs_np = point_idxs.numpy()
    patch_dists_np = patch_dists.numpy()
    memberships = [[] for _ in range(N)]
    for patch_id in range(num_patches):
        for local_id, point_id in enumerate(point_idxs_np[patch_id]):
            memberships[int(point_id)].append(
                (patch_id, local_id, float(patch_dists_np[patch_id, local_id]))
            )

    pcl_out = pcl_noisy_np.copy()
    missing_count = 0
    for pidx in range(N):
        point_memberships = memberships[pidx]
        if fusion_mode == "nearest":
            if point_memberships:
                patch_id, local_id, _ = min(
                    point_memberships,
                    key=lambda item: item[2],
                )
                selected = patches_denoised_np[patch_id, local_id]
            else:
                selected = None
        elif fusion_mode == "distance_weighted":
            if point_memberships:
                pred_stack = np.stack(
                    [
                        patches_denoised_np[patch_id, local_id]
                        for patch_id, local_id, _ in point_memberships
                    ],
                    axis=0,
                )
                weight_stack = np.asarray(
                    [
                        np.exp(-float(fusion_tau) * dist)
                        for _, _, dist in point_memberships
                    ],
                    dtype=np.float32,
                )
                weight_stack /= max(float(weight_stack.sum()), 1e-8)
                selected = (pred_stack * weight_stack[:, None]).sum(axis=0)
            else:
                selected = None
        else:
            raise ValueError(f"unsupported patch fusion mode: {fusion_mode}")

        if selected is None:
            missing_count += 1
            continue
        pcl_out[pidx] = selected
    if missing_count > 0:
        print(
            f"Patch fusion warning: {missing_count}/{N} points were not covered "
            "by any denoised patch; kept their noisy coordinates."
        )
    return jt.array(pcl_out)
