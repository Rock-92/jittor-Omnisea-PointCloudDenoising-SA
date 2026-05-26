import sys
from pathlib import Path

import jittor as jt
from jittor import nn
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.feature import (
    PointLayerNorm,
    RelativePositionBias,
    apply_point_linear,
    gather_neighbors,
    get_knn_idx,
)


class LegacyMultiScaleLocalSelfAttentionBlock(nn.Module):
    def __init__(self, dim, knn_scales, relative_position_bias_hidden_dim):
        super().__init__()
        self.dim = dim
        self.knn_scales = knn_scales
        self.scale = dim ** -0.5

        self.norm = PointLayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.rel_pos_bias = RelativePositionBias(relative_position_bias_hidden_dim)
        self.out_proj = nn.Linear(dim * len(knn_scales), dim)

    def execute(self, x, xyz, graph_knn_idx):
        x_norm = self.norm(x)
        q = apply_point_linear(self.q_proj, x_norm)
        k = apply_point_linear(self.k_proj, x_norm)
        v = apply_point_linear(self.v_proj, x_norm)

        scale_outputs = []
        for scale_k in self.knn_scales:
            idx = graph_knn_idx[:, :, :scale_k]
            k_neighbors = gather_neighbors(k, idx)
            v_neighbors = gather_neighbors(v, idx)
            xyz_neighbors = gather_neighbors(xyz, idx)
            rel_pos = xyz_neighbors - xyz.unsqueeze(2)

            attn_logits = (q.unsqueeze(2) * k_neighbors).sum(dim=-1) * self.scale
            attn_logits = attn_logits + self.rel_pos_bias(rel_pos)
            attn = nn.softmax(attn_logits, dim=-1)
            scale_outputs.append((attn.unsqueeze(-1) * v_neighbors).sum(dim=2))

        out = jt.concat(scale_outputs, dim=-1)
        out = apply_point_linear(self.out_proj, out)
        return x + out


class LegacyFeatureExtraction(nn.Module):
    def __init__(
        self,
        k=16,
        input_dim=3,
        input_expand_dim=128,
        embedding_dim=256,
        num_blocks=4,
        attention_weight_init=1.0,
        knn_scales=None,
        relative_position_bias_hidden_dim=32,
    ):
        super().__init__()
        if knn_scales is None:
            knn_scales = [k]
        elif isinstance(knn_scales, int):
            knn_scales = [knn_scales]
        self.knn_scales = [int(v) for v in knn_scales]
        self.max_knn = max(self.knn_scales)
        self.input_dim = input_dim
        self.input_expand_dim = input_expand_dim
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        self.input_proj_1 = nn.Linear(input_dim, input_expand_dim)
        self.input_proj_2 = nn.Linear(input_expand_dim, embedding_dim)
        self.act = nn.ReLU()

        if isinstance(attention_weight_init, (list, tuple)):
            assert len(attention_weight_init) == num_blocks
            block_weight_values = attention_weight_init
        else:
            block_weight_values = [attention_weight_init] * num_blocks

        self.blocks = []
        self.block_weights = []
        for i in range(num_blocks):
            block = LegacyMultiScaleLocalSelfAttentionBlock(
                dim=embedding_dim,
                knn_scales=self.knn_scales,
                relative_position_bias_hidden_dim=relative_position_bias_hidden_dim,
            )
            weight = jt.ones((1,)) * float(block_weight_values[i])
            setattr(self, f"block_{i}", block)
            setattr(self, f"block_weight_{i}", weight)
            self.blocks.append(block)
            self.block_weights.append(weight)

        self.fuse = nn.Linear(embedding_dim * num_blocks, embedding_dim)

    def execute(self, x):
        feat = apply_point_linear(self.input_proj_1, x)
        feat = self.act(feat)
        feat = apply_point_linear(self.input_proj_2, feat)
        feat = self.act(feat)

        block_outputs = []
        reuse_knn_idx = None
        for block_idx, (block, weight) in enumerate(zip(self.blocks, self.block_weights)):
            if block_idx == 0:
                graph_knn_idx = get_knn_idx(x, x, self.max_knn, offset=1)
            elif block_idx == 1:
                graph_knn_idx = get_knn_idx(feat, feat, self.max_knn, offset=1)
                reuse_knn_idx = graph_knn_idx
            else:
                graph_knn_idx = reuse_knn_idx
            feat = block(feat, x, graph_knn_idx)
            block_outputs.append(feat * weight)

        feat = jt.concat(block_outputs, dim=-1)
        return apply_point_linear(self.fuse, feat)


class LegacyDecoder(nn.Module):
    """Decoder used by checkpoints trained before decoder_hidden_dims was added."""

    def __init__(self, z_dim, out_dim, hidden_size):
        super().__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.lin_1 = nn.Linear(z_dim, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, out_dim)
        self.act = nn.ReLU()

    def execute(self, c):
        net = self.lin_1(c)
        net = self.act(net)
        return self.lin_2(net)


class LegacyVelocityModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_knn = cfg.get("attention_knn", cfg.get("frame_knn", 16))
        self.input_dim = cfg.get("input_dim", 3)
        self.input_expand_dim = cfg.get("input_expand_dim", 128)
        self.feat_embedding_dim = cfg["feat_embedding_dim"]
        self.attention_blocks = cfg.get("attention_blocks", 4)
        self.attention_weight_init = cfg.get("attention_weight_init", 1.0)
        self.relative_position_bias_hidden_dim = cfg.get(
            "relative_position_bias_hidden_dim",
            32,
        )
        self.decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 64))
        self.denoise_num_steps = cfg.get("denoise_num_steps", 1)

        self.encoder = LegacyFeatureExtraction(
            knn_scales=self.attention_knn,
            input_dim=self.input_dim,
            input_expand_dim=self.input_expand_dim,
            embedding_dim=self.feat_embedding_dim,
            num_blocks=self.attention_blocks,
            attention_weight_init=self.attention_weight_init,
            relative_position_bias_hidden_dim=self.relative_position_bias_hidden_dim,
        )
        self.decoder = LegacyDecoder(
            z_dim=self.encoder.embedding_dim,
            out_dim=3,
            hidden_size=self.decoder_hidden_dim,
        )

    def predict_displacement(self, pc_noisy):
        bsz, num_points, dim = pc_noisy.shape
        feat = self.encoder(pc_noisy)
        feat_dim = feat.shape[2]
        return self.decoder(feat.reshape(-1, feat_dim)).reshape(bsz, num_points, dim)

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=None):
        if num_steps is None:
            num_steps = self.denoise_num_steps
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for _ in range(num_steps):
                pred_dir = self.predict_displacement(pcl_next)
                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None


def load_legacy_model(checkpoint, model_config_path=None):
    model_config_path = model_config_path or PROJECT_ROOT / "configs/model/vm.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(model_config_path), resolve=True)
    if "decoder_hidden_dim" not in cfg:
        # Current configs may describe the newer 256-128-64-3 decoder. The
        # submitted best checkpoint was trained with the old 256-64-3 decoder.
        cfg["decoder_hidden_dim"] = 64
    model = LegacyVelocityModule(cfg)
    model.load(str(checkpoint))
    model.eval()
    return model
