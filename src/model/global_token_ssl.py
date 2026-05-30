from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from .feature import GlobalTokenGenerator, apply_point_linear
from .spec import ModelSpec
from ..data.asset import Asset


class MAEDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_points):
        super().__init__()
        self.num_points = int(num_points)
        self.lin_1 = nn.Linear(in_dim, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_3 = nn.Linear(hidden_dim, self.num_points * 3)
        self.act = nn.ReLU()

    def execute(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        x = self.act(x)
        return self.lin_3(x).reshape(x.shape[0], self.num_points, 3)


def chamfer_l2(pred, target):
    dist = ((pred.unsqueeze(2) - target.unsqueeze(1)) ** 2).sum(dim=-1)
    min_pred, _ = jt.min(dist, dim=2)
    min_target, _ = jt.min(dist, dim=1)
    return min_pred.mean() + min_target.mean()


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
        self.patch_size = int(cfg.get("patch_size", 1000))
        self.mask_ratio = float(cfg.get("mask_ratio", 0.6))
        self.mae_visible_points = int(
            cfg.get(
                "mae_visible_points",
                max(1, round(self.patch_size * (1.0 - self.mask_ratio))),
            )
        )
        self.mae_mask_points = int(
            cfg.get(
                "mae_mask_points",
                max(1, self.patch_size - self.mae_visible_points),
            )
        )
        self.mae_noise_std = float(cfg.get("mae_noise_std", 0.0))
        self.mae_jitter_std = float(cfg.get("mae_jitter_std", 0.0))
        self.mae_jitter_clip = float(cfg.get("mae_jitter_clip", 0.0))

        self.encoder = GlobalTokenEncoder(
            input_dim=cfg.get("input_dim", 3),
            input_expand_dim=cfg.get("input_expand_dim", 128),
            embedding_dim=cfg["feat_embedding_dim"],
            global_token_blocks=cfg.get("global_token_blocks", 4),
            global_token_ffn_hidden_dim=cfg.get("global_token_ffn_hidden_dim", 512),
        )
        dim = self.encoder.embedding_dim
        self.mae_decoder = MAEDecoder(
            in_dim=dim,
            hidden_dim=cfg.get("mae_decoder_hidden_dim", 512),
            num_points=self.mae_mask_points,
        )

    def encode_global(self, pc):
        token = self.encoder(pc)
        return token[:, 0, :]

    def training_step(self, batch: Dict) -> Dict:
        visible = batch.get("pc_visible", None)
        masked = batch.get("pc_masked", None)
        if visible is None or masked is None:
            clean_patch = batch["pc_clean"].reshape(-1, self.patch_size, 3)
            visible, masked = self.make_mae_batch_np(clean_patch.numpy())
            visible = jt.array(visible)
            masked = jt.array(masked)
        else:
            visible = visible.reshape(-1, self.mae_visible_points, 3)
            masked = masked.reshape(-1, self.mae_mask_points, 3)

        z = self.encode_global(visible)
        recon = self.mae_decoder(z)
        loss_mae = chamfer_l2(recon, masked)
        return {
            "mae_loss": loss_mae,
        }

    def make_mae_batch_np(self, clean_patch):
        visible = []
        masked = []
        for pc in clean_patch:
            v, m = self.make_mae_sample_np(pc)
            visible.append(v)
            masked.append(m)
        return (
            np.stack(visible, axis=0).astype(np.float32, copy=False),
            np.stack(masked, axis=0).astype(np.float32, copy=False),
        )

    def make_mae_sample_np(self, clean_patch):
        pc = clean_patch.astype(np.float32, copy=False)
        n = pc.shape[0]
        if self.mae_visible_points + self.mae_mask_points <= n:
            idx = np.random.permutation(n)
            visible_idx = idx[: self.mae_visible_points]
            masked_idx = idx[self.mae_visible_points : self.mae_visible_points + self.mae_mask_points]
        else:
            visible_idx = np.random.choice(n, self.mae_visible_points, replace=True)
            masked_idx = np.random.choice(n, self.mae_mask_points, replace=True)
        visible = pc[visible_idx].copy()
        masked = pc[masked_idx].copy()
        if self.mae_noise_std > 0.0:
            visible += np.random.laplace(0.0, self.mae_noise_std, size=visible.shape).astype(np.float32)
        if self.mae_jitter_std > 0.0:
            jitter = np.random.normal(0.0, self.mae_jitter_std, size=visible.shape).astype(np.float32)
            if self.mae_jitter_clip > 0.0:
                jitter = np.clip(jitter, -self.mae_jitter_clip, self.mae_jitter_clip)
            visible += jitter
        return visible.astype(np.float32, copy=False), masked.astype(np.float32, copy=False)

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        out = []
        for b in batch:
            assert b.sampled_vertices is not None
            visible, masked = self.make_mae_sample_np(b.sampled_vertices)
            out.append(
                {
                    "pc_clean": b.sampled_vertices,
                    "pc_visible": visible,
                    "pc_masked": masked,
                }
            )
        return out

    def execute(self, **kwargs) -> Dict:  # type: ignore
        return self.training_step(**kwargs)
