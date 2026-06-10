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
        self.hard_chamfer_weight = float(cfg.get('hard_chamfer_weight', 0.05))
        self.hard_normal_weight = float(cfg.get('hard_normal_weight', 0.02))
        
        # patch-based prediction
        self.predict_rounds = cfg.get('predict_rounds', 1)
        self.denoise_num_steps = cfg.get('denoise_num_steps', 1)
        self.predict_patch_size = cfg.get('predict_patch_size', 1000)
        self.predict_seed_k = cfg.get('predict_seed_k', 6)
        self.predict_seed_interval = cfg.get('predict_seed_interval', 200)
        self.predict_seed_k_alpha = cfg.get('predict_seed_k_alpha', 1)
        
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

    def get_hard_patch_weight(self, pc_noisy, sigma):
        if not self.use_hard_aware_loss:
            return None
        patch_scale = self.get_patch_scale(pc_noisy)
        relative_sigma = self.clamp_edm_sigma(sigma) / jt.maximum(
            patch_scale,
            self.patch_scale_eps,
        )
        hard = jt.maximum(relative_sigma / self.hard_relative_sigma_ref - 1.0, 0.0)
        weight = 1.0 + self.hard_weight_scale * hard
        return jt.minimum(weight, self.hard_weight_max)

    def get_patch_chamfer_loss(self, pc_pred, pc_clean, sigma, hard_weight=None):
        dist = ((pc_pred.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        pred_to_clean = dist.min(dim=2)
        clean_to_pred = dist.min(dim=1)
        sigma2 = self.clamp_edm_sigma(sigma).reshape(pc_pred.shape[0]) ** 2.0
        loss = (pred_to_clean.mean(dim=1) + clean_to_pred.mean(dim=1)) / sigma2
        if hard_weight is not None:
            loss = loss * hard_weight.reshape(pc_pred.shape[0])
        return loss.mean()

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
    
    def get_normalized_surface_loss(self, pc_pred, pc_clean, pc_anchor):
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
        return (plane_dist / self.dsm_sigma).mean()

    def get_supervised_losses(self, pc_noisy, pc_clean, score_sigma=None):
        """
        pc_noisy: (B, N, 3)
        pc_clean: (B, N, 3)
        """
        B = pc_noisy.shape[0]
        score_sigma = self._expand_sigma(score_sigma, B)
        target = pc_clean - pc_noisy
        point_idx = get_random_indices(pc_noisy.shape[1], self.num_train_points)
        pc_noisy_for_loss = pc_noisy
        pc_clean_for_loss = pc_clean
        if point_idx is not None:
            target = target[:, point_idx, :]
            pc_noisy_for_loss = pc_noisy[:, point_idx, :]
            pc_clean_for_loss = pc_clean[:, point_idx, :]
        if self.use_edm:
            sigma_pred = self.predict_sigma(pc_noisy)
            pc_pred = self.predict_clean(
                pc_noisy,
                sigma=score_sigma,
                point_idx=point_idx,
            )
            mse = ((pc_pred - pc_clean_for_loss) ** 2.0).sum(dim=-1)
            hard_weight = self.get_hard_patch_weight(pc_noisy, score_sigma)
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
                )
                if hard_weight is not None:
                    normal_loss = normal_loss * hard_weight.mean()
                losses["hard_normal_loss"] = normal_loss
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
        
        return {
            "displacement_loss": displacement_loss,
            "normalized_surface_loss": normalized_surface_loss,
        }

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
        losses = self.get_supervised_losses(
            pc_noisy=pc_noisy,
            pc_clean=pc_clean,
            score_sigma=score_sigma,
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
    
    all_dists = jt.ones((num_patches, N)) * 1e10
    
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]
        
    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)
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
    pcl_out = []
    pcl_noisy_flat = pcl_noisy[0]
    missing_count = 0
    for pidx in range(N):
        patch_id = best_weights_idx[pidx].item()
        mask = (point_idxs[patch_id] == pidx)
        selected = patches_denoised[patch_id][mask]
        if selected.shape[0] == 0:
            missing_count += 1
            selected = pcl_noisy_flat[pidx:pidx+1]
        else:
            selected = selected[:1]
        pcl_out.append(selected)
    pcl_out = jt.concat(pcl_out, dim=0)
    if missing_count > 0:
        print(
            f"Patch fusion warning: {missing_count}/{N} points were not covered "
            "by any denoised patch; kept their noisy coordinates."
        )
    return pcl_out
