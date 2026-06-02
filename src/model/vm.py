from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

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
        self.geometry_token_knn = cfg.get('geometry_token_knn', 32)
        self.geometry_token_hidden_dim = cfg.get('geometry_token_hidden_dim', 64)
        self.decoder_hidden_dims = cfg.get(
            'decoder_hidden_dims',
            [cfg.get('decoder_hidden_dim', 64)],
        )
        
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
        
        # networks
        self.encoder = FeatureExtraction(
            knn_scales=self.attention_knn,
            input_dim=self.input_dim,
            input_expand_dim=self.input_expand_dim,
            embedding_dim=self.feat_embedding_dim,
            num_blocks=self.attention_blocks,
            attention_weight_init=self.attention_weight_init,
            ffn_hidden_dim=self.attention_ffn_hidden_dim,
            geometry_token_knn=self.geometry_token_knn,
            geometry_token_hidden_dim=self.geometry_token_hidden_dim,
        )
        
        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=3,
            hidden_dims=self.decoder_hidden_dims,
        )
    
    def predict_displacement(self, pc_noisy, point_idx=None):
        """
        pc_noisy: (B, N, 3)
        point_idx: optional point indices decoded after full-patch encoding
        return:   (B, N, 3) or (B, M, 3)
        """
        B, N, d = pc_noisy.shape
        feat = self.encoder(pc_noisy)
        if point_idx is not None:
            feat = feat[:, point_idx, :]
        N_out = feat.shape[1]
        F_dim = feat.shape[2]
        return self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, d)
    
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

    def get_supervised_losses(self, pc_noisy, pc_clean):
        """
        pc_noisy: (B, N, 3)
        pc_clean: (B, N, 3)
        """
        point_idx = get_random_indices(pc_noisy.shape[1], self.num_train_points)
        feat = self.encoder(pc_noisy)
        target = pc_clean - pc_noisy
        pc_noisy_for_loss = pc_noisy
        pc_clean_for_loss = pc_clean
        if point_idx is not None:
            target = target[:, point_idx, :]
            pc_noisy_for_loss = pc_noisy[:, point_idx, :]
            pc_clean_for_loss = pc_clean[:, point_idx, :]
        if point_idx is not None:
            feat_for_loss = feat[:, point_idx, :]
        else:
            feat_for_loss = feat
        B, N_out, F_dim = feat_for_loss.shape
        pred_dir = self.decoder(feat_for_loss.reshape(-1, F_dim)).reshape(B, N_out, 3)
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
            for it in range(num_steps):
                pred_dir = self.predict_displacement(pcl_next)
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None
    
    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        losses = self.get_supervised_losses(
            pc_noisy=pc_noisy,
            pc_clean=pc_clean,
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
