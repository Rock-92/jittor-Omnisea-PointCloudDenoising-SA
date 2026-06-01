from typing import List

from jittor import nn

import jittor as jt


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


class RelativePositionBias(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin_1 = nn.Linear(4, hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()

    def execute(self, rel_pos):
        """
        rel_pos: (B, N, K, 3), neighbor_xyz - center_xyz
        return:  (B, N, K), scalar attention bias for each local edge.
        """
        B, N, K, _ = rel_pos.shape
        dist = jt.sqrt((rel_pos ** 2).sum(dim=-1, keepdims=True) + 1e-8)
        rel_feat = jt.concat([rel_pos, dist], dim=-1)
        bias = apply_edge_linear(self.lin_1, rel_feat)
        bias = self.act(bias)
        bias = apply_edge_linear(self.lin_2, bias)
        return bias.reshape(B, N, K)


class TokenSelfAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        if ffn_hidden_dim is None:
            ffn_hidden_dim = dim * 2
        self.ffn_hidden_dim = int(ffn_hidden_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_norm = PointLayerNorm(dim)

        self.ffn_lin_1 = nn.Linear(dim, self.ffn_hidden_dim)
        self.ffn_lin_2 = nn.Linear(self.ffn_hidden_dim, dim)
        self.ffn_norm = PointLayerNorm(dim)
        self.act = nn.ReLU()

    def execute(self, x):
        """
        x: (B, N, C), including the leading global token.
        """
        q = apply_point_linear(self.q_proj, x)
        k = apply_point_linear(self.k_proj, x)
        v = apply_point_linear(self.v_proj, x)
        attn_logits = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) * self.scale
        attn = nn.softmax(attn_logits, dim=-1)
        out = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        out = apply_point_linear(self.out_proj, out)
        x = self.attn_norm(x + out)

        ffn = apply_point_linear(self.ffn_lin_1, x)
        ffn = self.act(ffn)
        ffn = apply_point_linear(self.ffn_lin_2, ffn)
        return self.ffn_norm(x + ffn)


class GlobalTokenGenerator(nn.Module):
    def __init__(self, dim, num_blocks=4, ffn_hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.num_blocks = int(num_blocks)
        if ffn_hidden_dim is None:
            ffn_hidden_dim = dim * 2
        self.ffn_hidden_dim = int(ffn_hidden_dim)
        self.global_token = jt.randn((1, 1, dim)) * 0.02

        self.blocks = []
        for i in range(self.num_blocks):
            block = TokenSelfAttentionBlock(
                dim=dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
            )
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)

    def execute(self, point_feat):
        """
        point_feat: (B, N, C)
        return:     (B, 1, C), generated global token.
        """
        B, _, C = point_feat.shape
        global_token = self.global_token.broadcast((B, 1, C))
        tokens = jt.concat([global_token, point_feat], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens[:, :1, :]


class MultiScaleLocalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        knn_scales,
        relative_position_bias_hidden_dim,
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
        self.rel_pos_bias = RelativePositionBias(relative_position_bias_hidden_dim)
        self.out_proj = nn.Linear(dim * len(knn_scales), dim)
        self.global_norm = PointLayerNorm(dim)
        self.scale_gate_proj = nn.Linear(dim, len(self.knn_scales))
        self.temperature_proj = nn.Linear(dim, len(self.knn_scales))
        self.rel_gate_proj = nn.Linear(dim, len(self.knn_scales))
        self._zero_linear(self.scale_gate_proj)
        self._zero_linear(self.temperature_proj)
        self._zero_linear(self.rel_gate_proj)

        self.ffn_norm = PointLayerNorm(dim)
        self.ffn_lin_1 = nn.Linear(dim, self.ffn_hidden_dim)
        self.ffn_lin_2 = nn.Linear(self.ffn_hidden_dim, dim)
        self.act = nn.ReLU()

    def _zero_linear(self, linear):
        linear.weight.update(jt.zeros(linear.weight.shape))
        if linear.bias is not None:
            linear.bias.update(jt.zeros(linear.bias.shape))

    def execute(self, x, xyz, graph_knn_idx, condition_feat=None):
        """
        x:       (B, N, C), point features.
        xyz:     (B, N, 3), noisy point coordinates for relative position bias.
        graph_knn_idx: (B, N, max(knn_scales)), neighbor indices used for
            attention neighbors and relative-position bias.
        condition_feat: optional (B, N, C), point-wise local geometry features
            used to modulate multi-scale attention.
        """
        x_norm = self.attn_norm(x)
        q = apply_point_linear(self.q_proj, x_norm)
        k = apply_point_linear(self.k_proj, x_norm)
        v = apply_point_linear(self.v_proj, x_norm)
        B, N, _ = x.shape
        if condition_feat is not None:
            local_norm = self.global_norm(condition_feat)
            scale_gate = nn.softmax(
                apply_point_linear(self.scale_gate_proj, local_norm),
                dim=-1,
            ) * len(self.knn_scales)
            temperature = jt.exp(0.25 * jt.tanh(
                apply_point_linear(self.temperature_proj, local_norm)
            ))
            rel_gate = 1.0 + 0.5 * jt.tanh(
                apply_point_linear(self.rel_gate_proj, local_norm)
            )

        scale_outputs = []
        for scale_idx, scale_k in enumerate(self.knn_scales):
            idx = graph_knn_idx[:, :, :scale_k]
            k_neighbors = gather_neighbors(k, idx)
            v_neighbors = gather_neighbors(v, idx)
            xyz_neighbors = gather_neighbors(xyz, idx)
            rel_pos = xyz_neighbors - xyz.unsqueeze(2)

            dot_logits = (q.unsqueeze(2) * k_neighbors).sum(dim=-1) * self.scale
            rel_bias = self.rel_pos_bias(rel_pos)
            if condition_feat is not None:
                dot_logits = dot_logits / temperature[:, :, scale_idx].reshape(B, N, 1)
                rel_bias = rel_bias * rel_gate[:, :, scale_idx].reshape(B, N, 1)
            attn_logits = dot_logits + rel_bias
            attn = nn.softmax(attn_logits, dim=-1)
            scale_out = (attn.unsqueeze(-1) * v_neighbors).sum(dim=2)
            if condition_feat is not None:
                scale_out = scale_out * scale_gate[:, :, scale_idx].reshape(B, N, 1)
            scale_outputs.append(scale_out)

        out = jt.concat(scale_outputs, dim=-1)
        out = apply_point_linear(self.out_proj, out)
        x = x + out

        ffn = self.ffn_norm(x)
        ffn = apply_point_linear(self.ffn_lin_1, ffn)
        ffn = self.act(ffn)
        ffn = apply_point_linear(self.ffn_lin_2, ffn)
        return x + ffn


class EdgeConvBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = int(dim)
        if hidden_dim is None:
            hidden_dim = dim
        self.hidden_dim = int(hidden_dim)
        self.edge_lin_1 = nn.Linear(self.dim * 2 + 3, self.hidden_dim)
        self.edge_lin_2 = nn.Linear(self.hidden_dim, self.dim)
        self.norm = PointLayerNorm(self.dim)
        self.act = nn.ReLU()

    def execute(self, feat, xyz, knn_idx):
        """
        feat:    (B, N, C)
        xyz:     (B, N, 3)
        knn_idx: (B, N, K), xyz-space neighbors.
        return:  (B, N, C)
        """
        feat_neighbors = gather_neighbors(feat, knn_idx)
        xyz_neighbors = gather_neighbors(xyz, knn_idx)
        feat_center = feat.unsqueeze(2).broadcast(feat_neighbors.shape)
        xyz_rel = xyz_neighbors - xyz.unsqueeze(2)
        edge = jt.concat([feat_center, feat_neighbors - feat_center, xyz_rel], dim=-1)
        edge_feat = apply_edge_linear(self.edge_lin_1, edge)
        edge_feat = self.act(edge_feat)
        edge_feat = apply_edge_linear(self.edge_lin_2, edge_feat)
        pooled = edge_feat.max(dim=2)
        return self.norm(feat + pooled)


class EdgeConvConditioner(nn.Module):
    def __init__(self, dim, knn=24, num_blocks=2, hidden_dim=None):
        super().__init__()
        self.dim = int(dim)
        self.knn = int(knn)
        self.num_blocks = int(num_blocks)
        if hidden_dim is None:
            hidden_dim = dim
        self.hidden_dim = int(hidden_dim)
        self.blocks = []
        for i in range(self.num_blocks):
            block = EdgeConvBlock(
                dim=self.dim,
                hidden_dim=self.hidden_dim,
            )
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)
        self.out_norm = PointLayerNorm(self.dim)

    def execute(self, feat, xyz):
        """
        feat: (B, N, C)
        xyz:  (B, N, 3)
        return point-wise local condition features: (B, N, C)
        """
        k = min(max(1, self.knn), xyz.shape[1] - 1)
        knn_idx = get_knn_idx(xyz, xyz, k, offset=1)
        out = feat
        for block in self.blocks:
            out = block(out, xyz, knn_idx)
        return self.out_norm(out)


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
        relative_position_bias_hidden_dim=32,
        ffn_hidden_dim=None,
        edgeconv_knn=24,
        edgeconv_blocks=2,
        edgeconv_hidden_dim=None,
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
        self.edgeconv_knn = int(edgeconv_knn)
        self.edgeconv_blocks = int(edgeconv_blocks)
        if edgeconv_hidden_dim is None:
            edgeconv_hidden_dim = embedding_dim
        self.edgeconv_hidden_dim = int(edgeconv_hidden_dim)

        self.input_proj_1 = nn.Linear(input_dim, input_expand_dim)
        self.input_proj_2 = nn.Linear(input_expand_dim, embedding_dim)
        self.act = nn.ReLU()
        self.edge_conditioner = EdgeConvConditioner(
            dim=embedding_dim,
            knn=self.edgeconv_knn,
            num_blocks=self.edgeconv_blocks,
            hidden_dim=self.edgeconv_hidden_dim,
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
                relative_position_bias_hidden_dim=relative_position_bias_hidden_dim,
                ffn_hidden_dim=self.ffn_hidden_dim,
            )
            weight = jt.ones((1,)) * float(block_weight_values[i])
            setattr(self, f"block_{i}", block)
            setattr(self, f"block_weight_{i}", weight)
            self.blocks.append(block)
            self.block_weights.append(weight)

        self.fuse = nn.Linear(embedding_dim * num_blocks, embedding_dim)

    def execute(self, x, return_condition=False):
        """
        x: (B, N, 3)
        return: (B, N, 256)
        """
        graph_knn_idx = get_knn_idx(x, x, self.max_knn, offset=1)
        reuse_knn_idx = None

        feat = apply_point_linear(self.input_proj_1, x)
        feat = self.act(feat)
        feat = apply_point_linear(self.input_proj_2, feat)
        feat = self.act(feat)
        condition_feat = self.edge_conditioner(feat, x)

        block_outputs = []
        for block_idx, (block, weight) in enumerate(zip(self.blocks, self.block_weights)):
            if block_idx == 0:
                block_knn_idx = graph_knn_idx
            elif block_idx == 1:
                reuse_knn_idx = get_knn_idx(feat, feat, self.max_knn, offset=1)
                block_knn_idx = reuse_knn_idx
            else:
                block_knn_idx = reuse_knn_idx
            feat = block(feat, x, block_knn_idx, condition_feat=condition_feat)
            block_outputs.append(feat * weight)

        feat = jt.concat(block_outputs, dim=-1)
        feat = apply_point_linear(self.fuse, feat)
        if return_condition:
            return feat, condition_feat
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
