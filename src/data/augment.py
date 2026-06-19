from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import os

from .asset import Asset
from .spec import ConfigSpec
from .utils import random_euler_rotation, sample_vertex_groups

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):
    
    num_samples: int # total number of vertices on the face to be sampled
    
    num_vertex_samples: int=0 # number of vertices to be chosen
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        assert asset.vertices is not None
        assert asset.faces is not None
        sampled_vertices, sampled_normals, sampled_vertex_groups, hidden_states = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled_vertices.astype(np.float32, copy=False)

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        if asset.meta is None:
            asset.meta = {}
        asset.meta['normalize_center'] = center
        asset.meta['normalize_scale'] = scale
        asset.sampled_vertices = (pc / scale).astype(np.float32, copy=False)

@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float

    noise_std_distribution: str="uniform"

    noise_log_std: float=0.35

    noise_type: str="laplace"

    noise_std_floor: float=1e-5
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)

    def sample_noise_std(self):
        if self.noise_std_distribution == "uniform":
            return np.random.uniform(self.noise_std_min, self.noise_std_max)
        if self.noise_std_distribution == "log_normal":
            sample_min = max(float(self.noise_std_min), float(self.noise_std_floor))
            log_min = np.log(sample_min)
            log_max = np.log(self.noise_std_max)
            if self.noise_std_min <= 0:
                log_mean = np.log(0.5 * self.noise_std_max)
            else:
                log_mean = 0.5 * (log_min + log_max)
            log_sigma = float(self.noise_log_std)
            for _ in range(32):
                log_noise_std = np.random.normal(log_mean, log_sigma)
                if log_min <= log_noise_std <= log_max:
                    return float(np.exp(log_noise_std))
            log_noise_std = np.clip(log_noise_std, log_min, log_max)
            return float(np.exp(log_noise_std))
        raise ValueError(
            f"unsupported noise_std_distribution: {self.noise_std_distribution}"
        )
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        noise_std = self.sample_noise_std()
        if self.noise_type == "laplace":
            noise = np.random.laplace(0, noise_std, size=pc.shape)
        elif self.noise_type == "gaussian":
            noise = np.random.randn(*pc.shape) * noise_std
        else:
            raise ValueError(f"unsupported noise_type: {self.noise_type}")
        noise = noise.astype(np.float32, copy=False)
        if asset.meta is None:
            asset.meta = {}
        asset.meta['noise_std'] = np.float32(noise_std)
        asset.sampled_vertices_noisy = (pc + noise).astype(np.float32, copy=False)

@dataclass(frozen=True)
class AugmentLinear(Augment):
    
    scale: Tuple[float, float]=(1.0, 1.0)
    
    rotate_x_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_y_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_z_range: Tuple[float, float]=(0.0, 0.0)
    
    scale_p: float=0.0
    
    rotate_p: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)

@dataclass(frozen=True)
class AugmentPatch(Augment):
    
    patch_size: int
    
    num_patches: int

    mix_with_clean: bool=False

    bridge_sample_t: bool=False

    bridge_t_min: float=1e-3

    bridge_t_max: float=1.0

    bridge_noise_std: float=0.0

    refine_surface_branches: bool=True

    branch_refine_min_points: int=80

    branch_refine_min_side: int=20

    branch_refine_min_fraction: float=0.08

    branch_refine_min_gap: float=0.0075

    branch_refine_min_gap_score: float=0.35

    branch_refine_min_count_ratio: float=0.3

    branch_refine_neighbor_k: int=12

    branch_refine_min_same_neighbor_ratio: float=0.85
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)

    def _split_branch_by_patch_height(self, clean_patch, indices):
        if indices.size < max(2 * self.branch_refine_min_side, self.branch_refine_min_points):
            return None
        local = clean_patch[indices]
        centered = local - local.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(local.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        heights = local @ normal
        order = np.argsort(heights)
        sorted_heights = heights[order]
        gaps = np.diff(sorted_heights)
        if gaps.size == 0:
            return None

        min_side = max(
            int(self.branch_refine_min_side),
            int(round(indices.size * float(self.branch_refine_min_fraction))),
        )
        if indices.size < 2 * min_side:
            return None
        lo = min_side - 1
        hi = gaps.size - min_side + 1
        if hi <= lo:
            return None
        local_best = int(np.argmax(gaps[lo:hi]))
        best = lo + local_best
        best_gap = float(gaps[best])
        if best_gap < float(self.branch_refine_min_gap):
            return None

        left_heights = sorted_heights[: best + 1]
        right_heights = sorted_heights[best + 1:]
        left_iqr = float(np.percentile(left_heights, 75) - np.percentile(left_heights, 25))
        right_iqr = float(np.percentile(right_heights, 75) - np.percentile(right_heights, 25))
        score = best_gap / max(left_iqr + right_iqr, 1e-12)
        if score < float(self.branch_refine_min_gap_score):
            return None

        left = indices[order[: best + 1]]
        right = indices[order[best + 1:]]
        count_ratio = min(left.size, right.size) / max(left.size, right.size)
        if count_ratio < float(self.branch_refine_min_count_ratio):
            return None

        layer = np.zeros(indices.size, dtype=np.int8)
        layer[order[best + 1:]] = 1
        neighbor_k = min(int(self.branch_refine_neighbor_k) + 1, indices.size)
        if neighbor_k > 1:
            _, nn_idx = cKDTree(local).query(local, k=neighbor_k)
            nn_idx = np.asarray(nn_idx, dtype=np.int64)
            same_ratio = (layer[nn_idx[:, 1:]] == layer[:, None]).mean()
            if same_ratio < float(self.branch_refine_min_same_neighbor_ratio):
                return None
        return left, right

    def _refine_patch_branch_labels(self, clean_patch, labels, valid):
        if not self.refine_surface_branches:
            return labels
        refined = labels.copy()
        next_label = int(refined.max()) + 1 if refined.size else 0
        for label in np.unique(labels[valid > 0.5]):
            mask = (labels == label) & (valid > 0.5)
            indices = np.flatnonzero(mask).astype(np.int64)
            split = self._split_branch_by_patch_height(clean_patch, indices)
            if split is None:
                continue
            left, right = split
            if left.size < self.branch_refine_min_side or right.size < self.branch_refine_min_side:
                continue
            refined[left] = next_label
            next_label += 1
            refined[right] = next_label
            next_label += 1
        return refined
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy
        
        assert pc is not None
        assert pc_noisy is not None
        
        N = pc_noisy.shape[0]
        
        seed_idx = np.random.permutation(N)[:self.num_patches]   # (P,)
        seed_points = pc_noisy[seed_idx]                         # (P, 3)
        
        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed_points, k=self.patch_size)   # (P, M)

        pat_A = pc_noisy[nn_idx]  # (P, M, 3)
        pat_B = pc[nn_idx]        # (P, M, 3)
        seed_points = seed_points[:, None, :]
        if self.bridge_sample_t:
            t = np.random.uniform(
                self.bridge_t_min,
                self.bridge_t_max,
                size=(self.num_patches, 1, 1),
            ).astype(np.float32, copy=False)
            pat_t = (t * pat_A + (1.0 - t) * pat_B).astype(np.float32, copy=False)
            if self.bridge_noise_std > 0:
                bridge_noise_scale = (
                    self.bridge_noise_std * np.sqrt(np.maximum(t * (1.0 - t), 0.0))
                ).astype(np.float32, copy=False)
                bridge_noise = np.random.randn(*pat_t.shape).astype(np.float32, copy=False)
                pat_t = (pat_t + bridge_noise_scale * bridge_noise).astype(
                    np.float32,
                    copy=False,
                )
            pat_A = pat_A - seed_points
            pat_B = pat_B - seed_points
            pat_t = pat_t - seed_points
            patch_seed = seed_points
        elif self.mix_with_clean:
            t = np.random.rand(self.num_patches, self.patch_size, 1).astype(
                np.float32,
                copy=False,
            )
            t = (1.0 - 1e-8) * t + 1e-8
            seed_points_t = (
                t[:, 0:1, :] * pc[seed_idx][:, None, :]
                + (1.0 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
            ).astype(np.float32, copy=False)
            pat_t = (t * pat_B + (1.0 - t) * pat_A).astype(np.float32, copy=False)
            pat_A = pat_A - seed_points_t
            pat_B = pat_B - seed_points_t
            pat_t = pat_t - seed_points_t
            patch_seed = seed_points_t
        else:
            pat_A = pat_A - seed_points
            pat_B = pat_B - seed_points
            patch_seed = seed_points
        
        if asset.meta is None:
            asset.meta = {}
        asset.meta['pc_noisy'] = pat_A
        asset.meta['pc_clean'] = pat_B
        asset.meta['patch_seed'] = patch_seed
        if 'surface_branch_labels' in asset.meta:
            labels = asset.meta['surface_branch_labels'][nn_idx]
            valid = asset.meta['surface_branch_valid'][nn_idx]
            normals = asset.meta['surface_branch_normals'][nn_idx]
            if self.refine_surface_branches:
                refined_labels = []
                for patch_id in range(labels.shape[0]):
                    refined_labels.append(
                        self._refine_patch_branch_labels(
                            pat_B[patch_id],
                            labels[patch_id],
                            valid[patch_id],
                        )
                    )
                labels = np.stack(refined_labels, axis=0)
            noise_fraction = np.full(
                (self.num_patches, 1),
                asset.meta.get('surface_branch_noise_fraction', 1.0),
                dtype=np.float32,
            )
            asset.meta['pc_branch_label'] = labels.astype(np.int32, copy=False)
            asset.meta['pc_branch_valid'] = valid.astype(np.float32, copy=False)
            asset.meta['pc_branch_normal'] = normals.astype(np.float32, copy=False)
            asset.meta['pc_branch_noise_fraction'] = noise_fraction
        if 'noise_std' in asset.meta:
            asset.meta['score_sigma'] = np.full(
                (self.num_patches, 1),
                asset.meta['noise_std'],
                dtype=np.float32,
            )
        if self.bridge_sample_t:
            asset.meta['pc_bridge'] = pat_t
            asset.meta['bridge_t'] = t.reshape(self.num_patches, 1).astype(
                np.float32,
                copy=False,
            )
        if self.mix_with_clean:
            asset.meta['pc_mix'] = pat_t


@dataclass(frozen=True)
class AugmentSurfaceBranchCache(Augment):
    cache_root: str = "cache_surface_branches"
    cache_name: str = "surface_branches.npz"
    required: bool = False

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSurfaceBranchCache':
        cls.check_keys(kwargs)
        return AugmentSurfaceBranchCache(**kwargs)

    def _rel_path(self, asset: Asset):
        if asset.path is None:
            return None
        normalized = asset.path.replace("\\", "/")
        marker = "/shapenet/"
        if marker in normalized:
            rel = "shapenet/" + normalized.split(marker, maxsplit=1)[1]
            suffix = "/models/model_normalized.obj"
            if rel.endswith(suffix):
                rel = rel[:-len(suffix)]
            clean_suffix = "/clean.npy"
            if rel.endswith(clean_suffix):
                rel = rel[:-len(clean_suffix)]
            return rel
        marker = "shapenet/"
        if marker in normalized:
            rel = marker + normalized.split(marker, maxsplit=1)[1]
            suffix = "/models/model_normalized.obj"
            if rel.endswith(suffix):
                rel = rel[:-len(suffix)]
            clean_suffix = "/clean.npy"
            if rel.endswith(clean_suffix):
                rel = rel[:-len(clean_suffix)]
            return rel
        return None

    def apply(self, asset: Asset, **kwargs):
        if asset.meta is None:
            asset.meta = {}
        rel_path = self._rel_path(asset)
        if rel_path is None:
            if self.required:
                raise ValueError(f"cannot infer relative path from {asset.path}")
            return
        cache_path = os.path.join(self.cache_root, rel_path, self.cache_name)
        if not os.path.exists(cache_path):
            if self.required:
                raise FileNotFoundError(cache_path)
            return
        cache = np.load(cache_path)
        labels = cache["labels"].astype(np.int32, copy=False)
        normals = cache["normals"].astype(np.float32, copy=False)
        valid = cache["valid_mask"].astype(np.bool_, copy=False)
        if asset.sampled_vertices is None:
            raise ValueError("surface branch cache requires sampled clean points")
        if labels.shape[0] != asset.sampled_vertices.shape[0]:
            raise ValueError(
                "surface branch cache point count does not match sampled points: "
                f"{labels.shape[0]} vs {asset.sampled_vertices.shape[0]}"
            )
        asset.meta['surface_branch_labels'] = labels
        asset.meta['surface_branch_normals'] = normals
        asset.meta['surface_branch_valid'] = valid
        asset.meta['surface_branch_noise_fraction'] = np.float32(
            cache["noise_fraction"][0]
        )

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
        "surface_branch_cache": AugmentSurfaceBranchCache,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments
