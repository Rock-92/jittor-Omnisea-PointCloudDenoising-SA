from typing import Dict, List, Optional

import jittor as jt
import numpy as np
from jittor import nn

from .spec import ModelSpec

from ..data.asset import Asset


def get_random_indices(n, m):
    if m is None or m <= 0 or m >= n:
        return None
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation: Optional[str] = "ReLU"):
        super().__init__()

        if activation == "ReLU":
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError(f"unsupported activation: {activation}")

    def execute(self, x, edge_index):
        """
        x:          (B*N, C)
        edge_index: (2, E), source and destination indices in the flattened batch
        """
        src = edge_index[0]
        dst = edge_index[1]

        x_i = x[dst]
        x_j = x[src]
        msg = self.mlp(jt.concat([x_i, x_j - x_i], dim=1))

        out = jt.full((x.shape[0], msg.shape[1]), 0)
        cnt = jt.full((x.shape[0], msg.shape[1]), 0)
        out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce="add")
        cnt = cnt.scatter_(
            0,
            dst.unsqueeze(1).broadcast(msg.shape),
            jt.ones_like(msg),
            reduce="add",
        )
        return out / (cnt + 1) + self.lin(x)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, activation: Optional[str] = "ReLU"):
        super().__init__(in_channels, out_channels, activation)

    def execute(self, x, edge_index):
        return super().execute(x, edge_index)


class EdgeConvFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=16,
        input_dim=3,
        embedding_dim=256,
        distance_estimation=False,
        noise_embedding_dim=None,
    ):
        super().__init__()
        self.k = int(k)
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.distance_estimation = bool(distance_estimation)
        self.noise_embedding_dim = noise_embedding_dim

        dim1 = self.embedding_dim // 8
        dim2 = self.embedding_dim // 4
        self.conv1 = DynamicEdgeConv(self.input_dim, dim1)
        self.conv2 = DynamicEdgeConv(dim1, dim2)
        self.conv3 = DynamicEdgeConv(
            dim1 + dim2,
            self.embedding_dim,
            activation=None,
        )
        if self.noise_embedding_dim is not None:
            self.noise_film_1 = nn.Linear(self.noise_embedding_dim, 2 * dim1)
            self.noise_film_2 = nn.Linear(self.noise_embedding_dim, 2 * dim2)
            self.noise_film_3 = nn.Linear(
                self.noise_embedding_dim,
                2 * self.embedding_dim,
            )

    def get_edge_index(self, x):
        """
        x: (B, N, C)
        return flattened batch edge index: (2, B*N*k)
        """
        B, N, _ = x.shape
        knn_idx = get_knn_idx(x, x, self.k + 1)
        knn_idx = knn_idx[:, :, 1:]
        base = (jt.arange(B) * N).reshape(B, 1, 1)
        knn_idx = knn_idx + base

        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base

        return jt.stack([knn_idx.reshape(-1), dst.reshape(-1)], dim=0)

    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8)

    def apply_noise_film(self, x, noise_emb, film):
        if noise_emb is None:
            return x
        scale_shift = 0.1 * film(noise_emb)
        split_dim = scale_shift.shape[-1] // 2
        scale = scale_shift[:, :split_dim]
        shift = scale_shift[:, split_dim:]
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def execute(self, x, noise_emb=None):
        """
        x: (B, N, 3)
        noise_emb: optional (B, noise_embedding_dim)
        return: (B, N, embedding_dim)
        """
        B, N, _ = x.shape
        if self.distance_estimation:
            x = self.normalize_patch(x)

        edge_index = self.get_edge_index(x)
        x1 = self.conv1(x.reshape(B * N, -1), edge_index)
        x1 = x1.reshape(B, N, -1)
        if self.noise_embedding_dim is not None:
            x1 = self.apply_noise_film(x1, noise_emb, self.noise_film_1)

        edge_index = self.get_edge_index(x1)
        x2 = self.conv2(x1.reshape(B * N, -1), edge_index)
        x2 = x2.reshape(B, N, -1)
        if self.noise_embedding_dim is not None:
            x2 = self.apply_noise_film(x2, noise_emb, self.noise_film_2)

        edge_index = self.get_edge_index(x2)
        x_combined = jt.concat([x1, x2], dim=-1)
        x3 = self.conv3(x_combined.reshape(B * N, -1), edge_index)
        x3 = x3.reshape(B, N, -1)
        if self.noise_embedding_dim is not None:
            x3 = self.apply_noise_film(x3, noise_emb, self.noise_film_3)
        return x3


class EdgeConvDecoder(nn.Module):
    def __init__(self, z_dim, out_dim=3, hidden_size=64, dropout=0.1):
        super().__init__()
        self.z_dim = int(z_dim)
        self.out_dim = int(out_dim)
        self.hidden_size = int(hidden_size)
        self.lin_1 = nn.Linear(self.z_dim, self.z_dim)
        self.bn_1_out = nn.BatchNorm1d(self.z_dim)
        self.lin_2 = nn.Linear(self.z_dim, self.hidden_size)
        self.bn_2_out = nn.BatchNorm1d(self.hidden_size)
        self.lin_3 = nn.Linear(self.hidden_size, self.out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

    def execute(self, c):
        """
        c: (B*N, F)
        return: (B*N, 3)
        """
        net = self.lin_1(c)
        net = self.bn_1_out(net)
        net = self.act(net)
        net = self.dropout(net)

        net = self.lin_2(net)
        net = self.bn_2_out(net)
        net = self.act(net)
        net = self.dropout(net)
        return self.lin_3(net)


class EdgeConvBaselineModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.frame_knn = int(cfg.get("frame_knn", 16))
        self.input_dim = int(cfg.get("input_dim", 3))
        self.feat_embedding_dim = int(cfg.get("feat_embedding_dim", 256))
        self.decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 64))
        self.decoder_dropout = float(cfg.get("decoder_dropout", 0.1))
        self.num_train_points = int(cfg.get("num_train_points", 128))
        self.dsm_sigma = float(cfg.get("dsm_sigma", 0.01))
        self.use_edm = bool(cfg.get("use_edm", True))
        self.sigma_data = float(cfg.get("sigma_data", 0.10))
        self.edm_default_sigma = float(
            cfg.get("edm_default_sigma", self.dsm_sigma)
        )
        self.edm_loss_weighting = bool(cfg.get("edm_loss_weighting", True))
        self.noise_embedding_dim = int(
            cfg.get("noise_embedding_dim", self.feat_embedding_dim)
        )
        self.use_patch_scale_condition = bool(
            cfg.get("use_patch_scale_condition", True)
        )
        self.patch_scale_eps = float(cfg.get("patch_scale_eps", 1e-4))
        self.edm_sampler = str(cfg.get("edm_sampler", "heun"))
        self.edm_sigma_min = float(cfg.get("edm_sigma_min", 1e-4))
        self.edm_inference_sigmas = [
            float(v)
            for v in cfg.get(
                "edm_inference_sigmas",
                [0.020, 0.010, 0.005, 0.0],
            )
        ]

        self.predict_rounds = int(cfg.get("predict_rounds", 1))
        self.predict_patch_size = int(cfg.get("predict_patch_size", 1000))
        self.predict_seed_k = int(cfg.get("predict_seed_k", 6))
        self.predict_seed_interval = int(cfg.get("predict_seed_interval", 200))
        self.predict_seed_k_alpha = int(cfg.get("predict_seed_k_alpha", 1))

        if self.use_edm:
            self.noise_embed_1 = nn.Linear(1, self.noise_embedding_dim)
            self.noise_embed_2 = nn.Linear(
                self.noise_embedding_dim,
                self.noise_embedding_dim,
            )
            self.noise_act = nn.ReLU()
        self.encoder = EdgeConvFeatureExtraction(
            k=self.frame_knn,
            input_dim=self.input_dim,
            embedding_dim=self.feat_embedding_dim,
            noise_embedding_dim=(
                self.noise_embedding_dim if self.use_edm else None
            ),
        )
        self.decoder = EdgeConvDecoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=3,
            hidden_size=self.decoder_hidden_dim,
            dropout=self.decoder_dropout,
        )

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
        centered = pc_noisy - pc_noisy.mean(dim=1).unsqueeze(1)
        scale2 = (centered ** 2.0).sum(dim=-1).mean(dim=1)
        scale = jt.sqrt(
            scale2 + self.patch_scale_eps ** 2.0
        ).reshape(pc_noisy.shape[0], 1)
        return jt.maximum(scale, self.patch_scale_eps)

    def get_noise_embedding(self, sigma, patch_scale=None):
        sigma = self.clamp_edm_sigma(sigma)
        sigma_condition = sigma
        if self.use_patch_scale_condition and patch_scale is not None:
            sigma_condition = sigma / jt.maximum(
                patch_scale,
                self.patch_scale_eps,
            )
            sigma_condition = self.clamp_edm_sigma(sigma_condition)
        c_noise = jt.log(sigma_condition) / 4.0
        return self.noise_embed_2(self.noise_act(self.noise_embed_1(c_noise)))

    def get_edm_coefficients(self, sigma):
        sigma = self.clamp_edm_sigma(sigma)
        sigma2 = sigma ** 2.0
        sigma_data2 = self.sigma_data ** 2.0
        denom = sigma2 + sigma_data2
        c_skip = sigma_data2 / denom
        c_out = sigma * self.sigma_data / jt.sqrt(denom)
        c_in = 1.0 / jt.sqrt(denom)
        return c_skip, c_out, c_in

    def encode_features(self, pc_noisy, sigma=None, point_idx=None):
        B = pc_noisy.shape[0]
        noise_emb = None
        encoder_input = pc_noisy
        if self.use_edm:
            sigma = self._expand_sigma(sigma, B)
            _, _, c_in = self.get_edm_coefficients(sigma)
            patch_scale = self.get_patch_scale(pc_noisy)
            noise_emb = self.get_noise_embedding(
                sigma,
                patch_scale=patch_scale,
            )
            encoder_input = pc_noisy * c_in.reshape(B, 1, 1)
        feat = self.encoder(encoder_input, noise_emb=noise_emb)
        if point_idx is not None:
            feat = feat[:, point_idx, :]
        return feat

    def predict_raw(self, pc_noisy, sigma=None, point_idx=None):
        B, _, d = pc_noisy.shape
        feat = self.encode_features(
            pc_noisy,
            sigma=sigma,
            point_idx=point_idx,
        )
        N_out = feat.shape[1]
        F_dim = feat.shape[2]
        return self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, d)

    def predict_clean(self, pc_noisy, sigma=None, point_idx=None):
        if not self.use_edm:
            pc_base = pc_noisy
            if point_idx is not None:
                pc_base = pc_noisy[:, point_idx, :]
            return pc_base + self.predict_raw(
                pc_noisy,
                point_idx=point_idx,
            )
        B = pc_noisy.shape[0]
        sigma = self._expand_sigma(sigma, B)
        raw = self.predict_raw(
            pc_noisy,
            sigma=sigma,
            point_idx=point_idx,
        )
        pc_base = pc_noisy
        if point_idx is not None:
            pc_base = pc_noisy[:, point_idx, :]
        c_skip, c_out, _ = self.get_edm_coefficients(sigma)
        return (
            c_skip.reshape(B, 1, 1) * pc_base
            + c_out.reshape(B, 1, 1) * raw
        )

    def predict_displacement(self, pc_noisy, sigma=None, point_idx=None):
        pc_base = pc_noisy
        if point_idx is not None:
            pc_base = pc_noisy[:, point_idx, :]
        return (
            self.predict_clean(
                pc_noisy,
                sigma=sigma,
                point_idx=point_idx,
            )
            - pc_base
        )

    def get_supervised_loss(self, pc_noisy, pc_clean, score_sigma=None):
        B = pc_noisy.shape[0]
        point_idx = get_random_indices(
            pc_noisy.shape[1],
            self.num_train_points,
        )
        target = pc_clean
        if point_idx is not None:
            target = target[:, point_idx, :]

        if self.use_edm:
            score_sigma = self._expand_sigma(score_sigma, B)
            pred_clean = self.predict_clean(
                pc_noisy,
                sigma=score_sigma,
                point_idx=point_idx,
            )
            loss = ((pred_clean - target) ** 2.0).sum(dim=-1)
            if self.edm_loss_weighting:
                sigma = self.clamp_edm_sigma(score_sigma)
                sigma2 = sigma ** 2.0
                sigma_data2 = self.sigma_data ** 2.0
                weight = (sigma2 + sigma_data2) / (
                    sigma2 * sigma_data2
                )
                loss = loss * weight.reshape(B, 1)
            return loss.mean()

        pred_dir = self.predict_displacement(
            pc_noisy,
            point_idx=point_idx,
        )
        target_dir = target
        pc_base = pc_noisy
        if point_idx is not None:
            pc_base = pc_noisy[:, point_idx, :]
        target_dir = target_dir - pc_base
        return (
            ((pred_dir - target_dir) ** 2.0) / self.dsm_sigma
        ).sum(dim=-1).mean()

    def edm_derivative(self, x, sigma):
        sigma_var = self._expand_sigma(sigma, x.shape[0])
        sigma_var = self.clamp_edm_sigma(sigma_var)
        denoised = self.predict_clean(x, sigma=sigma_var)
        return (
            (x - denoised)
            / sigma_var.reshape(x.shape[0], 1, 1)
        )

    def edm_euler_sampler(self, x):
        for sigma, sigma_next in zip(
            self.edm_inference_sigmas[:-1],
            self.edm_inference_sigmas[1:],
        ):
            step = sigma_next - sigma
            x = x + step * self.edm_derivative(x, sigma)
        return x

    def edm_heun_sampler(self, x):
        for sigma, sigma_next in zip(
            self.edm_inference_sigmas[:-1],
            self.edm_inference_sigmas[1:],
        ):
            step = sigma_next - sigma
            derivative = self.edm_derivative(x, sigma)
            x_euler = x + step * derivative
            if sigma_next <= 0:
                x = x_euler
                continue
            derivative_next = self.edm_derivative(x_euler, sigma_next)
            x = x + 0.5 * step * (derivative + derivative_next)
        return x

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=None):
        with jt.no_grad():
            if self.use_edm:
                if self.edm_sampler == "heun":
                    pcl_next = self.edm_heun_sampler(pcl_noisy.clone())
                elif self.edm_sampler == "euler":
                    pcl_next = self.edm_euler_sampler(pcl_noisy.clone())
                else:
                    raise ValueError(
                        f"unsupported edm_sampler: {self.edm_sampler}"
                    )
            else:
                pcl_next = self.predict_clean(pcl_noisy)
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        score_sigma = batch.get("score_sigma")
        if score_sigma is not None:
            score_sigma = score_sigma.reshape(-1, 1)
        return {
            "loss": self.get_supervised_loss(
                pc_noisy=pc_noisy,
                pc_clean=pc_clean,
                score_sigma=score_sigma,
            )
        }

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch["pc_noisy"]
        assert pc_noisy_batch.ndim == 3

        res = []
        for pc_noisy in pc_noisy_batch:
            pc_next = pc_noisy
            for _ in range(self.predict_rounds):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=self.predict_patch_size,
                    seed_k=self.predict_seed_k,
                    seed_k_alpha=self.predict_seed_k_alpha,
                )
            res.append({"pc_denoised": pc_next.detach().numpy()})
        return res

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                item = {
                    "pc_noisy": b.meta["pc_noisy"],
                    "pc_clean": b.meta["pc_clean"],
                }
                if "patch_seed" in b.meta:
                    item["patch_seed"] = b.meta["patch_seed"]
                if "score_sigma" in b.meta:
                    item["score_sigma"] = b.meta["score_sigma"]
                res.append(item)
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
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
        pts = pcls[b]
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for _ in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    return jt.concat(sampled, dim=0), jt.concat(indices, dim=0)


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
    nn = []
    for b in range(x.shape[0]):
        nn.append(y[b][idx[b]])
    return dist_k, idx, jt.stack(nn, dim=0)


def patch_based_denoise(model: EdgeConvBaselineModule, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1) -> jt.Var:
    """
    Starter-code patch denoise and hard patch fusion.
    pcl_noisy: (N, 3)
    """
    assert len(pcl_noisy.shape) == 2

    N, _ = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)

    seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)

    patches = patches[0]
    patch_dists = patch_dists[0]
    point_idxs = point_idxs[0]

    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand

    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)
    all_dists = jt.ones((num_patches, N)) * 1e10

    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]

    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)
    patches_denoised = []

    i = 0
    patch_step = int(np.ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    while i < num_patches:
        curr = patches[i:i + patch_step]
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
    for pidx in range(N):
        patch_id = best_weights_idx[pidx].item()
        mask = point_idxs[patch_id] == pidx
        pcl_out.append(patches_denoised[patch_id][mask])
    return jt.concat(pcl_out, dim=0)


def get_knn_idx(x, y, k, offset=0):
    """
    x: (B, N, d)
    y: (B, M, d)
    return: (B, N, k)
    """
    K = k + offset
    if x.shape[-1] == 3:
        _, idx = jt.misc.knn(x, y, K)
    else:
        dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
        _, idx = jt.topk(dist, k=K, dim=-1, largest=False)
    return idx[:, :, offset:]
