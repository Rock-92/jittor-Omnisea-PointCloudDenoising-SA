import jittor as jt
from jittor import nn

from .feature import gather_neighbors, get_knn_idx


def apply_mlp(module, x):
    shape = x.shape
    output = module(x.reshape(-1, shape[-1]))
    return output.reshape(*shape[:-1], output.shape[-1])


class PatchNoiseClassifier(nn.Module):
    """Classify patch noise from noisy points and a frozen VM prediction."""

    def __init__(
        self,
        k=24,
        local_dim=96,
        hidden_dim=192,
        num_classes=3,
        eps=1e-6,
    ):
        super().__init__()
        self.k = int(k)
        self.eps = float(eps)

        # Noisy/coarse offsets, center/neighbor VM displacement, and lengths.
        self.edge_mlp = nn.Sequential(
            nn.Linear(18, local_dim),
            nn.ReLU(),
            nn.Linear(local_dim, local_dim),
            nn.ReLU(),
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(9 + local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.patch_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.sigma_head = nn.Linear(hidden_dim // 2, 1)

    def execute(self, noisy, coarse):
        k = min(self.k, noisy.shape[1] - 1)
        neighbor_idx = get_knn_idx(noisy, noisy, k=k, offset=1)
        noisy_neighbors = gather_neighbors(noisy, neighbor_idx)
        coarse_neighbors = gather_neighbors(coarse, neighbor_idx)
        displacement = coarse - noisy
        neighbor_displacement = gather_neighbors(
            displacement,
            neighbor_idx,
        )

        noisy_offset = noisy_neighbors - noisy.unsqueeze(2)
        coarse_offset = coarse_neighbors - coarse.unsqueeze(2)
        center_displacement = displacement.unsqueeze(2).broadcast(
            neighbor_displacement.shape
        )
        noisy_length = jt.sqrt(
            (noisy_offset ** 2.0).sum(dim=-1, keepdims=True) + self.eps
        )
        coarse_length = jt.sqrt(
            (coarse_offset ** 2.0).sum(dim=-1, keepdims=True) + self.eps
        )
        displacement_delta = neighbor_displacement - center_displacement
        displacement_delta_length = jt.sqrt(
            (displacement_delta ** 2.0).sum(dim=-1, keepdims=True)
            + self.eps
        )
        edge_input = jt.concat(
            [
                noisy_offset,
                coarse_offset,
                center_displacement,
                neighbor_displacement,
                noisy_length,
                coarse_length,
                displacement_delta,
                displacement_delta_length,
            ],
            dim=-1,
        )
        edge_feature = apply_mlp(self.edge_mlp, edge_input).max(dim=2)

        point_input = jt.concat(
            [noisy, coarse, displacement, edge_feature],
            dim=-1,
        )
        point_feature = apply_mlp(self.point_mlp, point_input)
        pooled_mean = point_feature.mean(dim=1)
        pooled_max = point_feature.max(dim=1)

        displacement_length = jt.sqrt(
            (displacement ** 2.0).sum(dim=-1) + self.eps
        )
        noisy_radius_mean = noisy_length.mean(dim=2).mean(dim=1).reshape(-1)
        coarse_radius_mean = coarse_length.mean(dim=2).mean(dim=1).reshape(-1)
        displacement_delta_mean = (
            displacement_delta_length.mean(dim=2)
            .mean(dim=1)
            .reshape(-1)
        )
        displacement_delta_max = (
            displacement_delta_length.max(dim=2)
            .mean(dim=1)
            .reshape(-1)
        )
        patch_stats = jt.stack(
            [
                displacement_length.mean(dim=1),
                displacement_length.max(dim=1),
                noisy_radius_mean,
                coarse_radius_mean,
                displacement_delta_mean,
                displacement_delta_max,
            ],
            dim=-1,
        )
        patch_feature = self.patch_mlp(
            jt.concat([pooled_mean, pooled_max, patch_stats], dim=-1)
        )
        logits = self.classifier(patch_feature)
        sigma_normalized = jt.sigmoid(self.sigma_head(patch_feature))
        return logits, {
            "sigma_normalized": sigma_normalized,
            "patch_feature": patch_feature,
        }
