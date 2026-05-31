from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .feature import GlobalTokenGenerator, apply_point_linear
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


class GlobalTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_expand_dim,
        embedding_dim,
        global_token_blocks,
        global_token_ffn_hidden_dim,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.input_proj_1 = nn.Linear(input_dim, input_expand_dim)
        self.input_proj_2 = nn.Linear(input_expand_dim, embedding_dim)
        self.act = nn.ReLU()
        self.global_token_generator = GlobalTokenGenerator(
            dim=embedding_dim,
            num_blocks=global_token_blocks,
            ffn_hidden_dim=global_token_ffn_hidden_dim,
        )

    def execute(self, pc):
        feat = apply_point_linear(self.input_proj_1, pc)
        feat = self.act(feat)
        feat = apply_point_linear(self.input_proj_2, feat)
        feat = self.act(feat)
        return self.global_token_generator(feat)


class GlobalTokenSSLModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.local_token_count = int(cfg.get("local_token_count", 16))
        self.local_token_patch_size = int(cfg.get("local_token_patch_size", 128))
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

        self.encoder = GlobalTokenEncoder(
            input_dim=cfg.get("input_dim", 3),
            input_expand_dim=cfg.get("input_expand_dim", 128),
            embedding_dim=cfg["feat_embedding_dim"],
            global_token_blocks=cfg.get("global_token_blocks", 4),
            global_token_ffn_hidden_dim=cfg.get("global_token_ffn_hidden_dim", 512),
        )
        dim = self.encoder.embedding_dim
        self.geom_head = MLPHead(dim, cfg.get("ssl_hidden_dim", 256), self.num_geom_classes)
        self.projector = MLPHead(dim, cfg.get("ssl_hidden_dim", 256), self.projection_dim)
        self.prototype_head = nn.Linear(self.projection_dim, self.num_prototypes, bias=False)

    def encode_local_tokens(self, local_patches):
        """
        local_patches: (B, M, K, 3)
        return:        (B, M, C)
        """
        B, M, K, _ = local_patches.shape
        tokens = self.encoder(local_patches.reshape(B * M, K, 3))[:, 0, :]
        return tokens.reshape(B, M, -1)

    def make_view(self, clean_patch):
        clean_patch = self.dropout_resample(clean_patch)
        noise_std = jt.rand((clean_patch.shape[0], 1, 1)) * (
            self.noise_std_max - self.noise_std_min
        ) + self.noise_std_min
        # Jittor has no laplace helper here; inverse-CDF from uniform.
        u = jt.rand(clean_patch.shape) - 0.5
        sign = (u > 0).float() * 2.0 - 1.0
        laplace = -noise_std * sign * jt.log(1.0 - 2.0 * jt.abs(u) + 1e-12)
        jitter = jt.clamp(jt.randn(clean_patch.shape) * self.jitter_std, -self.jitter_clip, self.jitter_clip)
        return clean_patch + laplace + jitter

    def dropout_resample(self, clean_patch):
        n = clean_patch.shape[1]
        if self.dropout_max <= 0.0 or n <= 1:
            return clean_patch
        pcs = []
        for i in range(clean_patch.shape[0]):
            drop = float(np.random.uniform(self.dropout_min, self.dropout_max))
            keep = max(1, int(round(n * (1.0 - drop))))
            keep_idx = np.random.choice(n, keep, replace=False)
            pc = clean_patch[i][jt.array(keep_idx).int32()]
            if keep < n:
                add_idx = np.random.choice(keep, n - keep, replace=True)
                pc = jt.concat([pc, pc[jt.array(add_idx).int32()]], dim=0)
            perm = np.random.permutation(n)
            pcs.append(pc[jt.array(perm).int32()][None, ...])
        return jt.concat(pcs, dim=0)

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

    def forward_local_heads(self, local_patches):
        z = self.encode_local_tokens(local_patches)
        B, M, C = z.shape
        z_flat = z.reshape(B * M, C)
        geom_logits = self.geom_head(z_flat).reshape(B, M, self.num_geom_classes)
        proj = l2_normalize(self.projector(z_flat))
        proto_logits = self.prototype_head(proj).reshape(B, M, self.num_prototypes)
        return geom_logits, proto_logits

    def training_step(self, batch: Dict) -> Dict:
        local_patches = batch["local_patches"].reshape(
            -1,
            self.local_token_count,
            self.local_token_patch_size,
            3,
        )
        local_labels = batch["local_geom_label"].reshape(-1, self.local_token_count).int32()
        local_view_a = self.make_view(
            local_patches.reshape(-1, self.local_token_patch_size, 3)
        ).reshape(
            -1,
            self.local_token_count,
            self.local_token_patch_size,
            3,
        )
        local_view_b = self.make_view(
            local_patches.reshape(-1, self.local_token_patch_size, 3)
        ).reshape(
            -1,
            self.local_token_count,
            self.local_token_patch_size,
            3,
        )
        geom_a, proto_a = self.forward_local_heads(local_view_a)
        geom_b, proto_b = self.forward_local_heads(local_view_b)
        labels_flat = local_labels.reshape(-1)
        geom_a_flat = geom_a.reshape(-1, self.num_geom_classes)
        geom_b_flat = geom_b.reshape(-1, self.num_geom_classes)
        loss_geom = (
            nn.cross_entropy_loss(geom_a_flat, labels_flat)
            + nn.cross_entropy_loss(geom_b_flat, labels_flat)
        ) * 0.5

        proto_a_flat = proto_a.reshape(-1, self.num_prototypes)
        proto_b_flat = proto_b.reshape(-1, self.num_prototypes)
        scores_a = proto_a_flat / self.swav_temperature
        scores_b = proto_b_flat / self.swav_temperature
        with jt.no_grad():
            q_a = self.sinkhorn(proto_a_flat)
            q_b = self.sinkhorn(proto_b_flat)
        loss_swav = 0.5 * (
            soft_cross_entropy(scores_a, q_b) + soft_cross_entropy(scores_b, q_a)
        )
        pred = geom_a_flat.argmax(dim=-1)[0]
        acc = (pred == labels_flat).float().mean()
        return {
            "geom_loss": loss_geom,
            "swav_loss": loss_swav,
            "geom_acc": acc,
        }

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        out = []
        for b in batch:
            assert b.meta is not None and "local_patches" in b.meta
            assert "local_geom_label" in b.meta
            out.append(
                {
                    "local_patches": b.meta["local_patches"],
                    "local_geom_label": b.meta["local_geom_label"],
                }
            )
        return out

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)
