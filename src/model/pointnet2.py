from typing import Dict, List, Optional

import jittor as jt
from jittor import nn

from .feature import (
    Decoder,
    apply_edge_linear,
    apply_point_linear,
    farthest_point_sampling_idx,
    gather_neighbors,
    gather_points,
    get_knn_idx,
)
from .spec import ModelSpec
from .vm import get_random_indices, patch_based_denoise

from ..data.asset import Asset


class SharedEdgeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.layer_names = []
        prev_dim = int(in_dim)
        for idx, dim in enumerate(hidden_dims):
            layer = nn.Linear(prev_dim, int(dim))
            name = f"lin_{idx}"
            setattr(self, name, layer)
            self.layer_names.append(name)
            prev_dim = int(dim)
        self.out_dim = prev_dim
        self.act = nn.ReLU()

    def execute(self, x):
        """
        x: (B, N, K, C)
        """
        for name in self.layer_names:
            x = apply_edge_linear(getattr(self, name), x)
            x = self.act(x)
        return x


class SharedPointMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.layer_names = []
        prev_dim = int(in_dim)
        for idx, dim in enumerate(hidden_dims):
            layer = nn.Linear(prev_dim, int(dim))
            name = f"lin_{idx}"
            setattr(self, name, layer)
            self.layer_names.append(name)
            prev_dim = int(dim)
        self.out_dim = prev_dim
        self.act = nn.ReLU()

    def execute(self, x):
        """
        x: (B, N, C)
        """
        for name in self.layer_names:
            x = apply_point_linear(getattr(self, name), x)
            x = self.act(x)
        return x


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        k: int,
        in_dim: int,
        mlp_dims: List[int],
    ):
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)
        self.in_dim = int(in_dim)
        self.mlp = SharedEdgeMLP(3 + self.in_dim, mlp_dims)
        self.out_dim = self.mlp.out_dim

    def execute(self, xyz, features=None):
        """
        xyz:      (B, N, 3)
        features: optional (B, N, C)
        return:
            new_xyz:  (B, S, 3)
            new_feat: (B, S, C_out)
        """
        npoint = min(self.npoint, xyz.shape[1])
        center_idx = farthest_point_sampling_idx(xyz, npoint)
        new_xyz = gather_points(xyz, center_idx)

        group_k = min(self.k, xyz.shape[1])
        group_idx = get_knn_idx(new_xyz, xyz, group_k, offset=0)
        grouped_xyz = gather_neighbors(xyz, group_idx)
        grouped_rel = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_feat = gather_neighbors(features, group_idx)
            grouped = jt.concat([grouped_rel, grouped_feat], dim=-1)
        else:
            grouped = grouped_rel

        grouped = self.mlp(grouped)
        new_feat = grouped.max(dim=2)
        return new_xyz, new_feat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_dim: int, mlp_dims: List[int]):
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = SharedPointMLP(self.in_dim, mlp_dims)
        self.out_dim = self.mlp.out_dim

    def interpolate(self, target_xyz, source_xyz, source_feat):
        """
        Interpolate source features onto target coordinates by inverse-distance 3NN.
        """
        B, N, _ = target_xyz.shape
        _, S, C = source_feat.shape
        if S == 1:
            return source_feat.broadcast((B, N, C))

        k = min(3, S)
        idx = get_knn_idx(target_xyz, source_xyz, k, offset=0)
        grouped_xyz = gather_neighbors(source_xyz, idx)
        grouped_feat = gather_neighbors(source_feat, idx)
        dist = jt.sqrt(((grouped_xyz - target_xyz.unsqueeze(2)) ** 2.0).sum(dim=-1) + 1e-10)
        inv_dist = 1.0 / (dist + 1e-8)
        weight = inv_dist / inv_dist.sum(dim=2, keepdims=True)
        return (grouped_feat * weight.unsqueeze(-1)).sum(dim=2)

    def execute(self, target_xyz, source_xyz, target_skip_feat, source_feat):
        interpolated = self.interpolate(target_xyz, source_xyz, source_feat)
        if target_skip_feat is not None:
            feat = jt.concat([interpolated, target_skip_feat], dim=-1)
        else:
            feat = interpolated
        return self.mlp(feat)


class PointNet2Encoder(nn.Module):
    def __init__(
        self,
        k: int = 32,
        sa_npoints: Optional[List[int]] = None,
        sa_channels: Optional[List[int]] = None,
        fp_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        if sa_npoints is None:
            sa_npoints = [256, 64, 8]
        if sa_channels is None:
            sa_channels = [128, 256, 1024]
        if fp_channels is None:
            fp_channels = [256, 128, 128]
        assert len(sa_npoints) == 3
        assert len(sa_channels) == 3
        assert len(fp_channels) == 3

        self.k = int(k)
        self.sa_npoints = [int(v) for v in sa_npoints]
        self.sa_channels = [int(v) for v in sa_channels]
        self.fp_channels = [int(v) for v in fp_channels]
        self.out_dim = self.fp_channels[-1]

        self.sa1 = PointNetSetAbstraction(
            npoint=self.sa_npoints[0],
            k=self.k,
            in_dim=0,
            mlp_dims=[64, 64, self.sa_channels[0]],
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=self.sa_npoints[1],
            k=self.k,
            in_dim=self.sa_channels[0],
            mlp_dims=[128, 128, self.sa_channels[1]],
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=self.sa_npoints[2],
            k=self.k,
            in_dim=self.sa_channels[1],
            mlp_dims=[256, 512, self.sa_channels[2]],
        )

        self.fp3 = PointNetFeaturePropagation(
            in_dim=self.sa_channels[2] + self.sa_channels[1],
            mlp_dims=[512, self.fp_channels[0]],
        )
        self.fp2 = PointNetFeaturePropagation(
            in_dim=self.fp_channels[0] + self.sa_channels[0],
            mlp_dims=[256, self.fp_channels[1]],
        )
        self.fp1 = PointNetFeaturePropagation(
            in_dim=self.fp_channels[1] + 3,
            mlp_dims=[128, self.fp_channels[2]],
        )

    def execute(self, xyz):
        """
        xyz: (B, N, 3)
        return per-input-point features: (B, N, C)
        """
        l0_xyz = xyz
        l0_feat = xyz

        l1_xyz, l1_feat = self.sa1(l0_xyz, None)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)

        l2_up = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_up)
        l0_up = self.fp1(l0_xyz, l1_xyz, l0_feat, l1_up)
        return l0_up


class PointNet2VelocityModule(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.k = int(cfg.get("k", 32))
        self.sa_npoints = cfg.get("sa_npoints", [256, 64, 8])
        self.sa_channels = cfg.get("sa_channels", [128, 256, 1024])
        self.fp_channels = cfg.get("fp_channels", [256, 128, 128])
        self.decoder_hidden_dims = cfg.get("decoder_hidden_dims", [128, 64])

        self.predict_rounds = cfg.get("predict_rounds", 1)
        self.denoise_num_steps = cfg.get("denoise_num_steps", 1)
        self.predict_patch_size = cfg.get("predict_patch_size", 1024)
        self.predict_seed_k = cfg.get("predict_seed_k", 6)
        self.predict_seed_interval = cfg.get("predict_seed_interval", 200)
        self.predict_seed_k_alpha = cfg.get("predict_seed_k_alpha", 1)

        self.dsm_sigma = cfg["dsm_sigma"]
        self.num_train_points = cfg.get("num_train_points", 0)

        self.encoder = PointNet2Encoder(
            k=self.k,
            sa_npoints=self.sa_npoints,
            sa_channels=self.sa_channels,
            fp_channels=self.fp_channels,
        )
        self.decoder = Decoder(
            z_dim=self.encoder.out_dim,
            out_dim=3,
            hidden_dims=self.decoder_hidden_dims,
        )

    def predict_displacement(self, pc_noisy, point_idx=None):
        B, _, d = pc_noisy.shape
        feat = self.encoder(pc_noisy)
        if point_idx is not None:
            feat = feat[:, point_idx, :]
        N_out = feat.shape[1]
        F_dim = feat.shape[2]
        return self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, d)

    def get_normalized_surface_loss(self, pc_pred, pc_clean, pc_anchor):
        dist = ((pc_anchor.unsqueeze(2) - pc_clean.unsqueeze(1)) ** 2.0).sum(dim=-1)
        _, idx = jt.topk(dist, k=3, dim=-1, largest=False)
        neighbors = []
        for b in range(pc_clean.shape[0]):
            neighbors.append(pc_clean[b][idx[b]])
        neighbors = jt.stack(neighbors, dim=0)

        p0 = pc_anchor
        p1 = neighbors[:, :, 1, :]
        p2 = neighbors[:, :, 2, :]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = jt.stack(
            [
                v1[:, :, 1] * v2[:, :, 2] - v1[:, :, 2] * v2[:, :, 1],
                v1[:, :, 2] * v2[:, :, 0] - v1[:, :, 0] * v2[:, :, 2],
                v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0],
            ],
            dim=-1,
        )
        normal = normal / (((normal ** 2.0).sum(dim=-1) + 1e-8) ** 0.5).unsqueeze(-1)
        plane_dist = (((pc_pred - p0) * normal).sum(dim=-1) ** 2.0)
        return (plane_dist / self.dsm_sigma).mean()

    def get_supervised_losses(self, pc_noisy, pc_clean):
        point_idx = get_random_indices(pc_noisy.shape[1], self.num_train_points)
        feat = self.encoder(pc_noisy)
        target = pc_clean - pc_noisy
        pc_noisy_for_loss = pc_noisy
        pc_clean_for_loss = pc_clean
        if point_idx is not None:
            feat = feat[:, point_idx, :]
            target = target[:, point_idx, :]
            pc_noisy_for_loss = pc_noisy[:, point_idx, :]
            pc_clean_for_loss = pc_clean[:, point_idx, :]

        B, N_out, F_dim = feat.shape
        pred_dir = self.decoder(feat.reshape(-1, F_dim)).reshape(B, N_out, 3)
        displacement_loss = (((pred_dir - target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()
        normalized_surface_loss = self.get_normalized_surface_loss(
            pc_pred=pc_noisy_for_loss + pred_dir,
            pc_clean=pc_clean,
            pc_anchor=pc_clean_for_loss,
        )
        return {
            "displacement_loss": displacement_loss,
            "normalized_surface_loss": normalized_surface_loss,
        }

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
        return self.get_supervised_losses(pc_noisy=pc_noisy, pc_clean=pc_clean)

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
                }
                if "patch_seed" in b.meta:
                    item["patch_seed"] = b.meta["patch_seed"]
                res.append(item)
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res
