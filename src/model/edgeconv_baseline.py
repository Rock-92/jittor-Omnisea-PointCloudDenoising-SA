from typing import Dict, List, Optional

import jittor as jt
import numpy as np
from jittor import nn

from .feature import get_knn_idx
from .spec import ModelSpec
from .vm import patch_based_denoise

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
    def execute(self, x, edge_index):
        return super().execute(x, edge_index)


class EdgeConvFeatureExtraction(nn.Module):
    def __init__(self, k=16, input_dim=3, embedding_dim=256, distance_estimation=False):
        super().__init__()
        self.k = int(k)
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.distance_estimation = bool(distance_estimation)

        self.conv1 = DynamicEdgeConv(self.input_dim, self.embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(self.embedding_dim // 8, self.embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            self.embedding_dim // 8 + self.embedding_dim // 4,
            self.embedding_dim,
            activation=None,
        )

    def get_edge_index(self, x):
        """
        x: (B, N, C)
        return flattened batch edge index: (2, B*N*k)
        """
        B, N, _ = x.shape
        k = min(self.k, N - 1)
        knn_idx = get_knn_idx(x, x, k, offset=1)
        base = (jt.arange(B) * N).reshape(B, 1, 1)
        knn_idx = knn_idx + base

        dst = jt.arange(N)
        dst = dst.reshape(1, N, 1).broadcast((B, N, k))
        dst = dst + base

        return jt.stack([knn_idx.reshape(-1), dst.reshape(-1)], dim=0)

    def normalize_patch(self, pcl):
        scale = jt.sqrt((pcl ** 2).sum(-1, keepdims=True))
        scale = scale.max(dim=-2, keepdims=True)
        return pcl / (scale + 1e-8)

    def execute(self, x):
        """
        x: (B, N, 3)
        return: (B, N, embedding_dim)
        """
        B, N, _ = x.shape
        if self.distance_estimation:
            x = self.normalize_patch(x)

        edge_index = self.get_edge_index(x)
        x1 = self.conv1(x.reshape(B * N, -1), edge_index)
        x1 = x1.reshape(B, N, -1)

        edge_index = self.get_edge_index(x1)
        x2 = self.conv2(x1.reshape(B * N, -1), edge_index)
        x2 = x2.reshape(B, N, -1)

        edge_index = self.get_edge_index(x2)
        x_combined = jt.concat([x1, x2], dim=-1)
        x3 = self.conv3(x_combined.reshape(B * N, -1), edge_index)
        return x3.reshape(B, N, -1)


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

        self.predict_rounds = int(cfg.get("predict_rounds", 1))
        self.denoise_num_steps = int(cfg.get("denoise_num_steps", 4))
        self.predict_patch_size = int(cfg.get("predict_patch_size", 1000))
        self.predict_seed_k = int(cfg.get("predict_seed_k", 6))
        self.predict_seed_interval = int(cfg.get("predict_seed_interval", 200))
        self.predict_seed_k_alpha = int(cfg.get("predict_seed_k_alpha", 1))

        self.encoder = EdgeConvFeatureExtraction(
            k=self.frame_knn,
            input_dim=self.input_dim,
            embedding_dim=self.feat_embedding_dim,
        )
        self.decoder = EdgeConvDecoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=3,
            hidden_size=self.decoder_hidden_dim,
            dropout=self.decoder_dropout,
        )

    def predict_displacement(self, pc_noisy, point_idx=None):
        B, _, d = pc_noisy.shape
        feat = self.encoder(pc_noisy)
        if point_idx is not None:
            feat = feat[:, point_idx, :]
        N_out = feat.shape[1]
        F_dim = feat.shape[2]
        return self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, d)

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        point_idx = get_random_indices(pc_mix.shape[1], self.num_train_points)
        feat = self.encoder(pc_mix)
        target = pc_clean - pc_noisy
        if point_idx is not None:
            feat = feat[:, point_idx, :]
            target = target[:, point_idx, :]

        B, N_out, F_dim = feat.shape
        pred_dir = self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, 3)
        return (((pred_dir - target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=None):
        if num_steps is None:
            num_steps = self.denoise_num_steps
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                pred_dir = self.predict_displacement(pcl_next)
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        pc_noisy = batch["pc_noisy"].reshape(-1, patch_size, 3)
        pc_clean = batch["pc_clean"].reshape(-1, patch_size, 3)
        pc_mix = batch.get("pc_mix", batch["pc_noisy"]).reshape(-1, patch_size, 3)
        return {
            "loss": self.get_supervised_loss(
                pc_noisy=pc_noisy,
                pc_mix=pc_mix,
                pc_clean=pc_clean,
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
                    seed_interval=self.predict_seed_interval,
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
                    "pc_mix": b.meta.get("pc_mix", b.meta["pc_noisy"]),
                }
                if "patch_seed" in b.meta:
                    item["patch_seed"] = b.meta["patch_seed"]
                res.append(item)
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
