import jittor as jt
from jittor import nn

from .feature import (
    MultiScaleLocalSelfAttentionBlock,
    apply_edge_linear,
    apply_point_linear,
    gather_neighbors,
    get_knn_idx,
)


class CleanShapeRegionProcessor(nn.Module):
    """Encode a clean full shape into spatially anchored region tokens."""

    def __init__(
        self,
        token_dim=128,
        region_knn=16,
        num_blocks=3,
        relative_bias_dim=32,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.region_knn = int(region_knn)
        self.num_blocks = int(num_blocks)

        self.point_mlp_1 = nn.Linear(3, 64)
        self.point_mlp_2 = nn.Linear(64, self.token_dim)
        self.center_proj = nn.Linear(3, self.token_dim)
        self.act = nn.ReLU()
        self.blocks = []
        for index in range(self.num_blocks):
            block = MultiScaleLocalSelfAttentionBlock(
                dim=self.token_dim,
                knn_scales=[8, self.region_knn],
                ffn_hidden_dim=self.token_dim * 2,
                relative_position_bias_hidden_dim=relative_bias_dim,
                global_attn_bias_init=0.5,
            )
            setattr(self, f"block_{index}", block)
            self.blocks.append(block)
        self.global_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.ReLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )

    def execute(self, region_points, region_centers):
        batch_size, region_count, point_count, _ = region_points.shape
        feature = apply_edge_linear(self.point_mlp_1, region_points)
        feature = self.act(feature)
        feature = apply_edge_linear(self.point_mlp_2, feature)
        feature = self.act(feature).max(dim=2)
        feature = feature + apply_point_linear(
            self.center_proj,
            region_centers,
        )
        k = min(self.region_knn, region_count - 1)
        neighbor_idx = get_knn_idx(
            region_centers,
            region_centers,
            k=k,
            offset=1,
        )
        for block in self.blocks:
            global_token = feature.mean(dim=1, keepdims=True)
            feature = block(
                feature,
                neighbor_idx,
                global_token=global_token,
                xyz=region_centers,
            )
        global_token = self.global_proj(
            feature.mean(dim=1)
        ).reshape(batch_size, 1, self.token_dim)
        return feature, global_token


class ShapeContextVMAdapter(nn.Module):
    """Refine frozen pure-VM output using clean full-shape region context."""

    def __init__(
        self,
        token_dim=128,
        hidden_dim=192,
        point_knn=(8, 16, 32),
        context_knn=4,
        max_residual=0.008,
        relative_bias_dim=32,
        context_only_head=False,
        eps=1e-6,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.point_knn = [int(value) for value in point_knn]
        self.context_knn = int(context_knn)
        self.max_residual = float(max_residual)
        self.context_only_head = bool(context_only_head)
        self.eps = float(eps)

        self.point_input = nn.Linear(12, self.token_dim)
        self.local_blocks = []
        for index in range(2):
            block = MultiScaleLocalSelfAttentionBlock(
                dim=self.token_dim,
                knn_scales=self.point_knn,
                ffn_hidden_dim=self.token_dim * 2,
                relative_position_bias_hidden_dim=relative_bias_dim,
                global_attn_bias_init=0.5,
            )
            setattr(self, f"local_block_{index}", block)
            self.local_blocks.append(block)

        self.query_proj = nn.Linear(self.token_dim, self.token_dim)
        self.key_proj = nn.Linear(self.token_dim, self.token_dim)
        self.value_proj = nn.Linear(self.token_dim, self.token_dim)
        self.context_bias_1 = nn.Linear(4, relative_bias_dim)
        self.context_bias_2 = nn.Linear(relative_bias_dim, 1)
        if not self.context_only_head:
            self.film = nn.Sequential(
                nn.Linear(self.token_dim * 2, self.token_dim * 2),
                nn.ReLU(),
                nn.Linear(self.token_dim * 2, self.token_dim * 2),
            )
        fuse_dim = self.token_dim * (2 if self.context_only_head else 3)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.direction_head = nn.Linear(self.hidden_dim, 3)
        self.length_head = nn.Linear(self.hidden_dim, 1)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.act = nn.ReLU()

        self.direction_head.weight.update(
            jt.randn(self.direction_head.weight.shape) * 1e-2
        )
        self.direction_head.bias.update(
            jt.randn(self.direction_head.bias.shape) * 1e-3
        )
        self.length_head.bias.update(
            jt.ones_like(self.length_head.bias) * -2.5
        )
        self.gate_head.bias.update(
            jt.ones_like(self.gate_head.bias) * -2.0
        )

    def _cross_attention(
        self,
        point_feature,
        point_global,
        region_tokens,
        region_centers,
    ):
        batch_size = point_feature.shape[0]
        if region_tokens.shape[0] == 1 and batch_size > 1:
            region_tokens = region_tokens.broadcast(
                (batch_size, region_tokens.shape[1], region_tokens.shape[2])
            )
            region_centers = region_centers.broadcast(
                (batch_size, region_centers.shape[1], 3)
            )
        k = min(self.context_knn, region_tokens.shape[1])
        context_idx = get_knn_idx(
            point_global,
            region_centers,
            k=k,
            offset=0,
        )
        keys = gather_neighbors(
            apply_point_linear(self.key_proj, region_tokens),
            context_idx,
        )
        values = gather_neighbors(
            apply_point_linear(self.value_proj, region_tokens),
            context_idx,
        )
        centers = gather_neighbors(region_centers, context_idx)
        relative = centers - point_global.unsqueeze(2)
        distance = jt.sqrt(
            (relative ** 2.0).sum(dim=-1, keepdims=True) + self.eps
        )
        bias_input = jt.concat([relative, distance], dim=-1)
        bias = self.act(
            apply_edge_linear(self.context_bias_1, bias_input)
        )
        bias = apply_edge_linear(
            self.context_bias_2,
            bias,
        ).reshape(
            point_feature.shape[0],
            point_feature.shape[1],
            k,
        )
        query = apply_point_linear(
            self.query_proj,
            point_feature,
        ).unsqueeze(2)
        logits = (
            (query * keys).sum(dim=-1) * (self.token_dim ** -0.5)
            + bias
        )
        attention = nn.softmax(logits, dim=-1)
        return (attention.unsqueeze(-1) * values).sum(dim=2)

    def execute(
        self,
        noisy_local,
        coarse_local,
        point_global,
        region_tokens,
        region_centers,
        global_token,
    ):
        displacement = coarse_local - noisy_local
        point_input = jt.concat(
            [
                noisy_local,
                coarse_local,
                displacement,
                point_global,
            ],
            dim=-1,
        )
        feature = self.act(
            apply_point_linear(self.point_input, point_input)
        )
        max_knn = min(max(self.point_knn), feature.shape[1] - 1)
        point_idx = get_knn_idx(
            coarse_local,
            coarse_local,
            k=max_knn,
            offset=1,
        )
        batch_size = feature.shape[0]
        if global_token.shape[0] == 1 and batch_size > 1:
            global_token = global_token.broadcast(
                (batch_size, 1, self.token_dim)
            )
        for block in self.local_blocks:
            feature = block(
                feature,
                point_idx,
                global_token=global_token,
                xyz=coarse_local,
            )
        context = self._cross_attention(
            feature,
            point_global,
            region_tokens,
            region_centers,
        )
        global_broadcast = global_token.broadcast(
            (batch_size, feature.shape[1], self.token_dim)
        )
        if self.context_only_head:
            fused = jt.concat([context, global_broadcast], dim=-1)
        else:
            patch_context = context.mean(dim=1, keepdims=True)
            condition = jt.concat(
                [global_token, patch_context],
                dim=-1,
            ).reshape(batch_size, self.token_dim * 2)
            film = self.film(condition).reshape(
                batch_size,
                1,
                self.token_dim * 2,
            )
            scale = film[:, :, :self.token_dim]
            shift = film[:, :, self.token_dim:]
            modulated = feature * (1.0 + scale) + shift
            fused = jt.concat(
                [modulated, context, global_broadcast],
                dim=-1,
            )
        hidden = self.fuse(
            fused.reshape(-1, fused.shape[-1])
        ).reshape(batch_size, feature.shape[1], self.hidden_dim)

        direction = self.direction_head(
            hidden.reshape(-1, self.hidden_dim)
        ).reshape(batch_size, feature.shape[1], 3)
        direction = direction / (
            jt.sqrt((direction ** 2.0).sum(dim=-1, keepdims=True))
            + self.eps
        )
        length = jt.sigmoid(
            self.length_head(
                hidden.reshape(-1, self.hidden_dim)
            ).reshape(batch_size, feature.shape[1], 1)
        ) * self.max_residual
        gate = jt.sigmoid(
            self.gate_head(
                hidden.reshape(-1, self.hidden_dim)
            ).reshape(batch_size, feature.shape[1], 1)
        )
        residual = direction * length * gate
        return coarse_local + residual, {
            "residual": residual,
            "direction": direction,
            "length": length,
            "gate": gate,
            "context": context,
        }
