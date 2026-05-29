from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .feature import FeatureExtraction, apply_point_linear
from .spec import ModelSpec
from ..data.asset import Asset


def l2_normalize(x, eps=1e-12):
    return x / jt.sqrt((x * x).sum(dim=-1, keepdims=True) + eps)


def soft_cross_entropy(logits, target_prob):
    log_prob = nn.log_softmax(logits, dim=-1)
    return -(target_prob * log_prob).sum(dim=-1).mean()


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def execute(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        return self.lin_2(x)


class GlobalTokenSSLModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.patch_size = int(cfg.get("patch_size", 1000))
        self.noise_std_min = float(cfg.get("noise_std_min", 0.005))
        self.noise_std_max = float(cfg.get("noise_std_max", 0.020))
        self.dropout_min = float(cfg.get("dropout_min", 0.0))
        self.dropout_max = float(cfg.get("dropout_max", 0.08))
        self.jitter_std = float(cfg.get("jitter_std", 0.001))
        self.jitter_clip = float(cfg.get("jitter_clip", 0.003))
        self.num_geom_classes = int(cfg.get("num_geom_classes", 12))
        self.num_prototypes = int(cfg.get("num_prototypes", 24))
        self.projection_dim = int(cfg.get("projection_dim", 128))
        self.swav_temperature = float(cfg.get("swav_temperature", 0.1))
        self.sinkhorn_epsilon = float(cfg.get("sinkhorn_epsilon", 0.05))
        self.sinkhorn_iters = int(cfg.get("sinkhorn_iters", 3))

        self.encoder = FeatureExtraction(
            knn_scales=cfg.get("attention_knn", [8, 16, 32]),
            input_dim=cfg.get("input_dim", 3),
            input_expand_dim=cfg.get("input_expand_dim", 128),
            embedding_dim=cfg["feat_embedding_dim"],
            num_blocks=cfg.get("attention_blocks", 4),
            attention_weight_init=cfg.get("attention_weight_init", [1.0, 0.6, 0.3, 0.2]),
            relative_position_bias_hidden_dim=cfg.get("relative_position_bias_hidden_dim", 64),
            ffn_hidden_dim=cfg.get("attention_ffn_hidden_dim", 512),
            global_token_blocks=cfg.get("global_token_blocks", 4),
            global_token_ffn_hidden_dim=cfg.get("global_token_ffn_hidden_dim", 512),
            global_attn_bias_init=cfg.get("global_attn_bias_init", 0.0),
        )
        dim = self.encoder.embedding_dim
        self.geom_head = MLPHead(dim, cfg.get("ssl_hidden_dim", 256), self.num_geom_classes)
        self.projector = MLPHead(dim, cfg.get("ssl_hidden_dim", 256), self.projection_dim)
        self.prototype_head = nn.Linear(self.projection_dim, self.num_prototypes, bias=False)

    def encode_global(self, pc):
        feat = apply_point_linear(self.encoder.input_proj_1, pc)
        feat = self.encoder.act(feat)
        feat = apply_point_linear(self.encoder.input_proj_2, feat)
        feat = self.encoder.act(feat)
        token = self.encoder.global_token_generator(feat)
        return token[:, 0, :]

    def make_view(self, clean_patch):
        noise_std = jt.rand((clean_patch.shape[0], 1, 1)) * (
            self.noise_std_max - self.noise_std_min
        ) + self.noise_std_min
        # Jittor has no laplace helper here; inverse-CDF from uniform.
        u = jt.rand(clean_patch.shape) - 0.5
        laplace = -noise_std * jt.sign(u) * jt.log(1.0 - 2.0 * jt.abs(u) + 1e-12)
        jitter = jt.clamp(jt.randn(clean_patch.shape) * self.jitter_std, -self.jitter_clip, self.jitter_clip)
        return clean_patch + laplace + jitter

    def make_view_np(self, clean_patch):
        pc = clean_patch.astype(np.float32, copy=True)
        n = pc.shape[0]
        drop = np.random.uniform(self.dropout_min, self.dropout_max)
        keep = max(1, int(round(n * (1.0 - drop))))
        keep_idx = np.random.choice(n, keep, replace=False)
        pc = pc[keep_idx]
        if pc.shape[0] < n:
            add_idx = np.random.choice(pc.shape[0], n - pc.shape[0], replace=True)
            pc = np.concatenate([pc, pc[add_idx]], axis=0)
        pc = pc[np.random.permutation(n)]
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        pc = pc + np.random.laplace(0.0, noise_std, size=pc.shape).astype(np.float32)
        jitter = np.clip(
            np.random.normal(0.0, self.jitter_std, size=pc.shape),
            -self.jitter_clip,
            self.jitter_clip,
        ).astype(np.float32)
        return (pc + jitter).astype(np.float32, copy=False)

    def sinkhorn(self, scores):
        q = jt.exp(scores / self.sinkhorn_epsilon).transpose(0, 1)
        q = q / (q.sum() + 1e-12)
        k = q.shape[0]
        b = q.shape[1]
        for _ in range(self.sinkhorn_iters):
            q = q / (q.sum(dim=1, keepdims=True) + 1e-12)
            q = q / k
            q = q / (q.sum(dim=0, keepdims=True) + 1e-12)
            q = q / b
        q = q * b
        return q.transpose(0, 1)

    def forward_heads(self, pc):
        z = self.encode_global(pc)
        geom_logits = self.geom_head(z)
        proj = l2_normalize(self.projector(z))
        proto_logits = self.prototype_head(proj)
        return geom_logits, proto_logits

    def training_step(self, batch: Dict) -> Dict:
        clean_patch = batch["pc_clean"].reshape(-1, self.patch_size, 3)
        labels = batch["geom_label"].reshape(-1).int32()
        view_a = batch.get("view_a", None)
        view_b = batch.get("view_b", None)
        if view_a is None or view_b is None:
            view_a = self.make_view(clean_patch)
            view_b = self.make_view(clean_patch)
        else:
            view_a = view_a.reshape(-1, self.patch_size, 3)
            view_b = view_b.reshape(-1, self.patch_size, 3)

        geom_a, proto_a = self.forward_heads(view_a)
        geom_b, proto_b = self.forward_heads(view_b)
        loss_geom = (nn.cross_entropy_loss(geom_a, labels) + nn.cross_entropy_loss(geom_b, labels)) * 0.5

        scores_a = proto_a / self.swav_temperature
        scores_b = proto_b / self.swav_temperature
        with jt.no_grad():
            q_a = self.sinkhorn(proto_a)
            q_b = self.sinkhorn(proto_b)
        loss_swav = 0.5 * (
            soft_cross_entropy(scores_a, q_b) + soft_cross_entropy(scores_b, q_a)
        )
        pred = geom_a.argmax(dim=-1)[0]
        acc = (pred == labels).float().mean()
        return {
            "geom_loss": loss_geom,
            "swav_loss": loss_swav,
            "geom_acc": acc,
        }

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        out = []
        for b in batch:
            assert b.sampled_vertices is not None
            assert b.meta is not None and "geom_label" in b.meta
            out.append(
                {
                    "pc_clean": b.sampled_vertices,
                    "view_a": self.make_view_np(b.sampled_vertices),
                    "view_b": self.make_view_np(b.sampled_vertices),
                    "geom_label": b.meta["geom_label"],
                }
            )
        return out

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)
