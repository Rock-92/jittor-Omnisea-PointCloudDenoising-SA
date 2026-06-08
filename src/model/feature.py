from typing import List

from jittor import nn

import jittor as jt
import numpy as np


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


def gather_neighbors(x, idx):
    """
    x:   (B, N, C)
    idx: (B, N, K)
    return: (B, N, K, C)
    """
    neighbors: List[jt.Var] = []
    for b in range(x.shape[0]):
        neighbors.append(x[b][idx[b]])
    return jt.stack(neighbors, dim=0)


def gather_points(x, idx):
    """
    x:   (B, N, C)
    idx: (B, M)
    return: (B, M, C)
    """
    gathered: List[jt.Var] = []
    for b in range(x.shape[0]):
        gathered.append(x[b][idx[b]])
    return jt.stack(gathered, dim=0)


def farthest_point_sampling_idx(xyz, num_points):
    """
    xyz: (B, N, 3)
    return: (B, num_points)
    """
    B, N, _ = xyz.shape
    num_points = min(int(num_points), N)
    indices = []
    for b in range(B):
        pts = xyz[b]
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for _ in range(num_points):
            selected.append(farthest)
            centroid = pts[farthest]
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        indices.append(jt.array(selected).int32()[None, :])
    return jt.concat(indices, dim=0)


def nearest_center_idx(xyz, centers):
    """
    xyz:     (B, N, 3)
    centers: (B, M, 3)
    return:  (B, N)
    """
    dist = ((xyz.unsqueeze(2) - centers.unsqueeze(1)) ** 2).sum(dim=-1)
    _, idx = jt.argmin(dist, dim=-1)
    return idx.int32()


def apply_point_linear(linear, x):
    B, N, _ = x.shape
    out = linear(x.reshape(B * N, -1))
    return out.reshape(B, N, -1)


def apply_edge_linear(linear, x):
    B, N, K, _ = x.shape
    out = linear(x.reshape(B * N * K, -1))
    return out.reshape(B, N, K, -1)


class PointLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = jt.ones((dim,))
        self.bias = jt.zeros((dim,))

    def execute(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdims=True)
        x = (x - mean) / jt.sqrt(var + self.eps)
        return x * self.weight.reshape(1, 1, self.dim) + self.bias.reshape(1, 1, self.dim)


class MultiScaleLocalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        knn_scales,
        ffn_hidden_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.knn_scales = knn_scales
        self.scale = dim ** -0.5
        if ffn_hidden_dim is None:
            ffn_hidden_dim = dim * 2
        self.ffn_hidden_dim = int(ffn_hidden_dim)

        self.attn_norm = PointLayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim * len(knn_scales), dim)
        self.global_norm = PointLayerNorm(dim)
        self.scale_gate_proj = nn.Linear(dim, len(self.knn_scales))
        self.temperature_proj = nn.Linear(dim, len(self.knn_scales))
        self.bridge_proj = nn.Linear(dim, 2 * dim)
        self._zero_linear(self.scale_gate_proj)
        self._zero_linear(self.temperature_proj)
        self._zero_linear(self.bridge_proj)

        self.ffn_norm = PointLayerNorm(dim)
        self.ffn_lin_1 = nn.Linear(dim, self.ffn_hidden_dim)
        self.ffn_lin_2 = nn.Linear(self.ffn_hidden_dim, dim)
        self.act = nn.ReLU()

    def _zero_linear(self, linear):
        linear.weight.update(jt.zeros(linear.weight.shape))
        if linear.bias is not None:
            linear.bias.update(jt.zeros(linear.bias.shape))

    def get_gate_embedding(self, condition_feat):
        local_norm = self.global_norm(condition_feat)
        scale_gate = nn.softmax(
            apply_point_linear(self.scale_gate_proj, local_norm),
            dim=-1,
        ) * len(self.knn_scales)
        temperature = jt.exp(0.25 * jt.tanh(
            apply_point_linear(self.temperature_proj, local_norm)
        ))
        return jt.concat([scale_gate, temperature], dim=-1)

    def _apply_bridge_modulation(self, x, bridge_feat):
        if bridge_feat is None:
            return x
        bridge_params = self.bridge_proj(bridge_feat)
        scale = bridge_params[:, :self.dim].reshape(bridge_feat.shape[0], 1, self.dim)
        shift = bridge_params[:, self.dim:].reshape(bridge_feat.shape[0], 1, self.dim)
        return x * (1.0 + scale) + shift

    def execute(self, x, graph_knn_idx, condition_feat=None, bridge_feat=None):
        """
        x:       (B, N, C), point features.
        graph_knn_idx: (B, N, max(knn_scales)), neighbor indices used for
            attention neighbors.
        condition_feat: optional (B, N, C), point-wise geometry tokens
            used to modulate multi-scale attention.
        """
        x_norm = self.attn_norm(x)
        x_norm = self._apply_bridge_modulation(x_norm, bridge_feat)
        q = apply_point_linear(self.q_proj, x_norm)
        k = apply_point_linear(self.k_proj, x_norm)
        v = apply_point_linear(self.v_proj, x_norm)
        B, N, _ = x.shape
        if condition_feat is not None:
            gate_embedding = self.get_gate_embedding(condition_feat)
            num_scales = len(self.knn_scales)
            scale_gate = gate_embedding[:, :, :num_scales]
            temperature = gate_embedding[:, :, num_scales:]

        scale_outputs = []
        for scale_idx, scale_k in enumerate(self.knn_scales):
            idx = graph_knn_idx[:, :, :scale_k]
            k_neighbors = gather_neighbors(k, idx)
            v_neighbors = gather_neighbors(v, idx)

            dot_logits = (q.unsqueeze(2) * k_neighbors).sum(dim=-1) * self.scale
            if condition_feat is not None:
                dot_logits = dot_logits / temperature[:, :, scale_idx].reshape(B, N, 1)
            attn_logits = dot_logits
            attn = nn.softmax(attn_logits, dim=-1)
            scale_out = (attn.unsqueeze(-1) * v_neighbors).sum(dim=2)
            if condition_feat is not None:
                scale_out = scale_out * scale_gate[:, :, scale_idx].reshape(B, N, 1)
            scale_outputs.append(scale_out)

        out = jt.concat(scale_outputs, dim=-1)
        out = apply_point_linear(self.out_proj, out)
        x = x + out

        ffn = self.ffn_norm(x)
        ffn = self._apply_bridge_modulation(ffn, bridge_feat)
        ffn = apply_point_linear(self.ffn_lin_1, ffn)
        ffn = self.act(ffn)
        ffn = apply_point_linear(self.ffn_lin_2, ffn)
        return x + ffn


class BridgeTimeEncoder(nn.Module):
    def __init__(self, out_dim, hidden_dim=64):
        super().__init__()
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.lin_1 = nn.Linear(6, self.hidden_dim)
        self.lin_2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.act = nn.ReLU()

    def execute(self, t):
        """
        t: (B,), (B, 1), or (B, 1, 1). t=0 is clean, t=1 is noisy.
        return: bridge condition vector (B, C)
        """
        if len(t.shape) == 3:
            t = t.reshape(t.shape[0], -1)[:, :1]
        elif len(t.shape) == 1:
            t = t.reshape(-1, 1)
        else:
            t = t[:, :1]
        one_minus_t = 1.0 - t
        bridge_sigma = jt.sqrt(jt.maximum(t * one_minus_t, 0.0) + 1e-8)
        feat = jt.concat(
            [
                t,
                one_minus_t,
                t * one_minus_t,
                bridge_sigma,
                jt.sin(np.pi * t),
                jt.cos(np.pi * t),
            ],
            dim=-1,
        )
        feat = self.lin_1(feat)
        feat = self.act(feat)
        return self.lin_2(feat)


class GeometryTokenEncoder(nn.Module):
    def __init__(self, out_dim, hidden_dim=64, knn=32):
        super().__init__()
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.knn = int(knn)
        self.lin_1 = nn.Linear(8, self.hidden_dim)
        self.lin_2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.norm = PointLayerNorm(self.out_dim)
        self.act = nn.ReLU()

    def _pointwise_geometry(self, xyz):
        k = min(max(3, self.knn), xyz.shape[1] - 1)
        knn_idx = get_knn_idx(xyz, xyz, k, offset=1)
        xyz_neighbors = gather_neighbors(xyz, knn_idx)
        xyz_rel = xyz_neighbors - xyz.unsqueeze(2)
        centered = xyz_rel - xyz_rel.mean(dim=2, keepdims=True)
        cov = centered.transpose(2, 3) @ centered
        cov = cov / float(max(k - 1, 1))
        eigvals, eigvecs = jt.linalg.eigh(cov)
        eigvals = jt.maximum(eigvals, 0.0)

        l0 = eigvals[:, :, 0]
        l1 = eigvals[:, :, 1]
        l2 = eigvals[:, :, 2]
        eig_sum = l0 + l1 + l2 + 1e-8
        curvature = l0 / eig_sum
        linearity = (l2 - l1) / (l2 + 1e-8)
        planarity = (l1 - l0) / (l2 + 1e-8)
        scattering = l0 / (l2 + 1e-8)

        normals = eigvecs[:, :, :, 0]
        normal_neighbors = gather_neighbors(normals, knn_idx)
        normal_dot = jt.abs((normal_neighbors * normals.unsqueeze(2)).sum(dim=-1))
        normal_variation = 1.0 - normal_dot.mean(dim=2)

        radius = jt.sqrt((xyz_rel ** 2.0).sum(dim=-1) + 1e-8)
        radius_mean = radius.mean(dim=2)
        radius_var = ((radius - radius_mean.unsqueeze(-1)) ** 2.0).mean(dim=2)
        radius_std = jt.sqrt(radius_var + 1e-8)
        radius_max = radius.max(dim=2)
        patch_radius = jt.sqrt((xyz ** 2.0).sum(dim=-1) + 1e-8).max(dim=1).reshape(-1, 1)
        radius_mean = radius_mean / (patch_radius + 1e-8)
        radius_std = radius_std / (patch_radius + 1e-8)
        radius_max = radius_max / (patch_radius + 1e-8)

        return jt.stack(
            [
                curvature,
                linearity,
                planarity,
                scattering,
                normal_variation,
                radius_mean,
                radius_std,
                radius_max,
            ],
            dim=-1,
        )

    def execute(self, feat, xyz):
        """
        xyz:  (B, N, 3), noisy patch coordinates.
        return point-wise geometry modulation tokens: (B, N, C)
        """
        geom = self._pointwise_geometry(xyz)
        token = apply_point_linear(self.lin_1, geom)
        token = self.act(token)
        token = apply_point_linear(self.lin_2, token)
        return self.norm(token)


class FeatureExtraction(nn.Module):
    def __init__(
        self,
        k=16,
        input_dim=3,
        input_expand_dim=128,
        embedding_dim=256,
        num_blocks=4,
        attention_weight_init=1.0,
        knn_scales=None,
        ffn_hidden_dim=None,
        geometry_token_knn=32,
        geometry_token_hidden_dim=64,
        bridge_hidden_dim=64,
        use_geometry_tokens=True,
        use_bridge_condition=True,
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
        if ffn_hidden_dim is None:
            ffn_hidden_dim = embedding_dim * 2
        self.ffn_hidden_dim = int(ffn_hidden_dim)
        self.geometry_token_knn = int(geometry_token_knn)
        self.geometry_token_hidden_dim = int(geometry_token_hidden_dim)
        self.bridge_hidden_dim = int(bridge_hidden_dim)
        self.use_geometry_tokens = bool(use_geometry_tokens)
        self.use_bridge_condition = bool(use_bridge_condition)

        self.input_proj_1 = nn.Linear(input_dim, input_expand_dim)
        self.input_proj_2 = nn.Linear(input_expand_dim, embedding_dim)
        self.act = nn.ReLU()
        self.geometry_token_encoder = GeometryTokenEncoder(
            out_dim=embedding_dim,
            hidden_dim=self.geometry_token_hidden_dim,
            knn=self.geometry_token_knn,
        )
        self.bridge_time_encoder = BridgeTimeEncoder(
            out_dim=embedding_dim,
            hidden_dim=self.bridge_hidden_dim,
        )

        if isinstance(attention_weight_init, (list, tuple)):
            assert len(attention_weight_init) == num_blocks, (
                "attention_weight_init length must match num_blocks"
            )
            block_weight_values = attention_weight_init
        else:
            block_weight_values = [attention_weight_init] * num_blocks

        self.blocks = []
        self.block_weights = []
        for i in range(num_blocks):
            block = MultiScaleLocalSelfAttentionBlock(
                dim=embedding_dim,
                knn_scales=self.knn_scales,
                ffn_hidden_dim=self.ffn_hidden_dim,
            )
            weight = jt.ones((1,)) * float(block_weight_values[i])
            setattr(self, f"block_{i}", block)
            setattr(self, f"block_weight_{i}", weight)
            self.blocks.append(block)
            self.block_weights.append(weight)

    def get_gate_embedding(self, condition_feat):
        gate_parts = []
        for block in self.blocks:
            gate_parts.append(block.get_gate_embedding(condition_feat))
        return jt.concat(gate_parts, dim=-1)

    def execute(self, x, bridge_t=None, return_condition=False):
        """
        x: (B, N, 3)
        return: (B, N, 256)
        """
        feat = apply_point_linear(self.input_proj_1, x)
        feat = self.act(feat)
        feat = apply_point_linear(self.input_proj_2, feat)
        feat = self.act(feat)
        geometry_feat = None
        condition_feat = None
        gate_embedding = None
        if self.use_geometry_tokens or return_condition:
            geometry_feat = self.geometry_token_encoder._pointwise_geometry(x)
        if self.use_geometry_tokens:
            condition_feat = self.geometry_token_encoder(feat, x)
            gate_embedding = self.get_gate_embedding(condition_feat)
        bridge_feat = None
        if self.use_bridge_condition and bridge_t is not None:
            bridge_feat = self.bridge_time_encoder(bridge_t)

        for block_idx, block in enumerate(self.blocks):
            if block_idx == 0:
                block_knn_idx = get_knn_idx(x, x, self.max_knn, offset=1)
            else:
                block_knn_idx = get_knn_idx(feat, feat, self.max_knn, offset=1)
            feat = block(
                feat,
                block_knn_idx,
                condition_feat=condition_feat,
                bridge_feat=bridge_feat,
            )

        if return_condition:
            return feat, geometry_feat, gate_embedding
        return feat

class Decoder(nn.Module):
    def __init__(self, z_dim, out_dim, hidden_dims):
        super().__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.hidden_dims = [int(v) for v in hidden_dims]

        self.hidden_layer_names = []
        prev_dim = z_dim
        for i, dim in enumerate(self.hidden_dims, start=1):
            layer = nn.Linear(prev_dim, dim)
            setattr(self, f"lin_{i}", layer)
            self.hidden_layer_names.append(f"lin_{i}")
            prev_dim = dim
        self.output_layer_name = f"lin_{len(self.hidden_dims) + 1}"
        setattr(self, self.output_layer_name, nn.Linear(prev_dim, out_dim))
        self.act = nn.ReLU()

    def execute(self, c):
        """
        c: (B*N, F)
        return: (B*N, 3)
        """
        net = c
        for layer_name in self.hidden_layer_names:
            layer = getattr(self, layer_name)
            net = layer(net)
            net = self.act(net)
        return getattr(self, self.output_layer_name)(net)
