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

        self.ffn_norm = PointLayerNorm(dim)
        self.ffn_lin_1 = nn.Linear(dim, self.ffn_hidden_dim)
        self.ffn_lin_2 = nn.Linear(self.ffn_hidden_dim, dim)
        self.act = nn.ReLU()

    def execute(self, x, graph_knn_idx):
        """
        x:       (B, N, C), point features.
        graph_knn_idx: (B, N, max(knn_scales)), neighbor indices used for
            attention neighbors.
        """
        x_norm = self.attn_norm(x)
        q = apply_point_linear(self.q_proj, x_norm)
        k = apply_point_linear(self.k_proj, x_norm)
        v = apply_point_linear(self.v_proj, x_norm)

        scale_outputs = []
        for scale_k in self.knn_scales:
            idx = graph_knn_idx[:, :, :scale_k]
            k_neighbors = gather_neighbors(k, idx)
            v_neighbors = gather_neighbors(v, idx)

            attn_logits = (q.unsqueeze(2) * k_neighbors).sum(dim=-1) * self.scale
            attn = nn.softmax(attn_logits, dim=-1)
            scale_outputs.append((attn.unsqueeze(-1) * v_neighbors).sum(dim=2))

        out = jt.concat(scale_outputs, dim=-1)
        out = apply_point_linear(self.out_proj, out)
        x = x + out

        ffn = self.ffn_norm(x)
        ffn = apply_point_linear(self.ffn_lin_1, ffn)
        ffn = self.act(ffn)
        ffn = apply_point_linear(self.ffn_lin_2, ffn)
        return x + ffn


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

        self.input_proj_1 = nn.Linear(input_dim, input_expand_dim)
        self.input_proj_2 = nn.Linear(input_expand_dim, embedding_dim)
        self.act = nn.ReLU()

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

        self.fuse = nn.Linear(embedding_dim * num_blocks, embedding_dim)

    def execute(self, x):
        """
        x: (B, N, 3)
        return: (B, N, 256)
        """
        feat = apply_point_linear(self.input_proj_1, x)
        feat = self.act(feat)
        feat = apply_point_linear(self.input_proj_2, feat)
        feat = self.act(feat)

        block_outputs = []
        for block_idx, (block, weight) in enumerate(zip(self.blocks, self.block_weights)):
            if block_idx == 0:
                block_knn_idx = get_knn_idx(x, x, self.max_knn, offset=1)
            else:
                block_knn_idx = get_knn_idx(feat, feat, self.max_knn, offset=1)
            feat = block(feat, block_knn_idx)
            block_outputs.append(feat * weight)

        feat = jt.concat(block_outputs, dim=-1)
        return apply_point_linear(self.fuse, feat)


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
