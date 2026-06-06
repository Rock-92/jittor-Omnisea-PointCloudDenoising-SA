from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.laplace(0, noise_std, size=pc.shape).astype(np.float32, copy=False)
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
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)
    
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
        if self.mix_with_clean:
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
        if self.mix_with_clean:
            asset.meta['pc_mix'] = pat_t

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
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
