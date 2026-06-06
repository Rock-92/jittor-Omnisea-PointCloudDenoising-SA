from typing import List, Dict, Optional

import jittor as jt
import numpy as np
import os
import trimesh
import zipfile
from pathlib import Path

from evaluate import chamfer_distance, metric_to_score, point_to_surface_distance

from .spec import DummySystem, DummyWriter
from ..data.asset import Asset, Exporter

class VMWriter(DummyWriter):
    
    def __init__(
        self,
        save_dir: str="outputs/result/dataset_test_noisy",
        save_name: str="denoised",
        output_format: str="npy",
        result_zip: str="result.zip",
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.output_format = output_format
        self.result_zip = result_zip
        self.written_files = []
    
    def reset(self):
        self.written_files = []
    
    def _get_predict_roots(self, dataset_module=None):
        if dataset_module is None or dataset_module.predict_datapath is None:
            return []
        datapaths = dataset_module.predict_datapath
        if not isinstance(datapaths, dict):
            datapaths = {"predict": datapaths}
        roots = []
        for datapath in datapaths.values():
            roots.append(os.path.abspath(datapath.input_dataset_dir))
        return roots
    
    def _relative_output_dir(self, path: str, dataset_module=None):
        path_dir = os.path.abspath(os.path.dirname(path))
        for root in self._get_predict_roots(dataset_module):
            rel_dir = os.path.relpath(path_dir, root)
            if rel_dir == "." or not rel_dir.startswith(".."):
                return rel_dir
        return os.path.dirname(path)
    
    def write(self, batch, prediction: List[Dict], dataset_module=None):
        pc_noisy_batch = batch['pc_noisy']
        for i, asset in enumerate(batch['asset']):
            path = asset.path
            assert path is not None, "asset path is None"
            dirname = os.path.join(
                self.save_dir,
                self._relative_output_dir(path, dataset_module=dataset_module),
            )
            os.makedirs(dirname, exist_ok=True)
            denoised = prediction[i]['pc_denoised']
            if isinstance(denoised, np.ndarray):
                denoised_np = denoised
            else:
                denoised_np = denoised.numpy()
            expected_shape = tuple(pc_noisy_batch[i].shape)
            assert denoised_np.shape == expected_shape, (
                f"denoised shape {denoised_np.shape} != noisy shape {expected_shape}"
            )
            if self.output_format == 'npy':
                output_path = os.path.join(dirname, f"{self.save_name}.npy")
                np.save(output_path, denoised_np.astype(np.float32))
            else:
                output_path = os.path.join(dirname, f"{self.save_name}.obj")
                Exporter.export_obj(denoised_np, output_path)
            self.written_files.append(output_path)
    
    def package_result(self):
        result_files = self.written_files.copy()
        if len(result_files) == 0:
            filename = f"{self.save_name}.{self.output_format}"
            for root, _, files in os.walk(self.save_dir):
                if "outputs1" in Path(root).parts:
                    continue
                if filename in files:
                    result_files.append(os.path.join(root, filename))
        result_files = [
            path for path in result_files
            if "outputs1" not in Path(path).parts
        ]
        result_files.sort()
        
        zip_dir = os.path.dirname(self.result_zip)
        if zip_dir:
            os.makedirs(zip_dir, exist_ok=True)
        with zipfile.ZipFile(self.result_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in result_files:
                arcname = os.path.relpath(path, self.save_dir).replace(os.sep, "/")
                zf.write(path, arcname)
        print(f"Packaged {len(result_files)} files into {self.result_zip}")

class VMSystem(DummySystem):
    
    def __init__(
        self,
        dataset_module,
        model,
        loss_config=None,
        optimizer_config=None,
        scheduler_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        super().__init__(
            dataset_module=dataset_module,
            model=model,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            trainer_config=trainer_config,
            writer=writer,
            ckpt_save_dir=ckpt_save_dir,
            ckpt_save_name=ckpt_save_name,
        )

    def on_train_end(self):
        if self.writer is None or self.dataset_module.predict_dataset_config is None:
            return
        
        best_checkpoint_path = os.path.join(
            self.ckpt_save_dir,
            f"{self.ckpt_save_name}_best.pkl",
        )
        if os.path.exists(best_checkpoint_path):
            print(f"Loading best checkpoint for test prediction: {best_checkpoint_path}")
            self.model.load(best_checkpoint_path)
        else:
            print(
                f"Best checkpoint not found at {best_checkpoint_path}; "
                "using current model weights for test prediction."
            )
        self.predict()
    
    def on_predict_epoch_start(self):
        if isinstance(self.writer, VMWriter):
            self.writer.reset()
    
    def on_predict_epoch_end(self):
        if isinstance(self.writer, VMWriter):
            self.writer.package_result()
    
    def _normalized_mesh(self, asset: Asset):
        if asset.meta is None:
            return None, None
        vertices = asset.vertices
        faces = asset.faces
        if vertices is None or faces is None:
            mesh_path = self._source_mesh_path_from_cache(asset.path)
            if mesh_path is None or not mesh_path.exists():
                return None, None
            mesh = trimesh.load(mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
        center = asset.meta.get("normalize_center")
        scale = asset.meta.get("normalize_scale")
        if center is None or scale is None or float(scale) < 1e-12:
            return None, None
        vertices = (vertices - center) / float(scale)
        faces = faces.astype(np.int32)
        return vertices, faces

    def _source_mesh_path_from_cache(self, path: Optional[str]):
        if path is None:
            return None
        parts = Path(path).resolve().parts
        if "cache_clean_points" not in parts:
            return None
        cache_idx = parts.index("cache_clean_points")
        base = Path(*parts[:cache_idx]) if cache_idx > 0 else Path(".")
        rel_parts = parts[cache_idx + 1:-1]
        if not rel_parts:
            return None
        return base / "dataset_clean" / Path(*rel_parts) / "models" / "model_normalized.obj"
    
    def validation_metric_step(self, batch):
        if "pc_noisy" not in batch or "pc_clean" not in batch:
            return None
        
        pc_noisy = batch["pc_noisy"]
        pc_clean = batch["pc_clean"]
        if len(pc_noisy.shape) == 4:
            batch_size, num_patches, patch_size, dim = pc_noisy.shape
            pc_noisy = pc_noisy.reshape(batch_size * num_patches, patch_size, dim)
            pc_clean = pc_clean.reshape(batch_size * num_patches, patch_size, dim)
            asset_indices = [
                patch_idx // num_patches
                for patch_idx in range(batch_size * num_patches)
            ]
        elif len(pc_noisy.shape) == 3:
            batch_size, patch_size, dim = pc_noisy.shape
            pc_noisy = pc_noisy.reshape(batch_size, patch_size, dim)
            pc_clean = pc_clean.reshape(batch_size, patch_size, dim)
            asset_indices = list(range(batch_size))
        else:
            return None
        
        patch_seed = batch.get("patch_seed")
        if patch_seed is not None:
            patch_seed = patch_seed.reshape(pc_noisy.shape[0], 1, pc_noisy.shape[2])
            pc_noisy_abs = pc_noisy + patch_seed
            pc_clean_abs = pc_clean + patch_seed
        else:
            pc_noisy_abs = pc_noisy
            pc_clean_abs = pc_clean
        
        pc_pred, _ = self.model.denoise_langevin_dynamics(pc_noisy)
        if patch_seed is not None:
            pc_pred_abs = pc_pred + patch_seed
        else:
            pc_pred_abs = pc_pred
        
        pred_np = pc_pred_abs.detach().numpy()
        noisy_np = pc_noisy_abs.detach().numpy()
        clean_np = pc_clean_abs.detach().numpy()
        assets = list(batch.get("asset", []))
        
        metrics = []
        for i in range(pred_np.shape[0]):
            clean_patch = clean_np[i]
            pred_patch = pred_np[i]
            noisy_patch = noisy_np[i]

            cd_pred = chamfer_distance(pred_patch, clean_patch, normalize=True)
            cd_noisy = chamfer_distance(noisy_patch, clean_patch, normalize=True)
            item = {
                "cd_pred": cd_pred,
                "cd_noisy": cd_noisy,
                "cd_score": metric_to_score(cd_pred, cd_noisy),
            }
            
            asset_idx = asset_indices[i]
            if asset_idx < len(assets):
                mesh_v, mesh_f = self._normalized_mesh(assets[asset_idx])
                if mesh_v is not None and mesh_f is not None:
                    p2s_pred = point_to_surface_distance(
                        pred_patch,
                        mesh_v,
                        mesh_f,
                        normalize_ref_pc=clean_patch,
                    )
                    p2s_noisy = point_to_surface_distance(
                        noisy_patch,
                        mesh_v,
                        mesh_f,
                        normalize_ref_pc=clean_patch,
                    )
                    if p2s_pred is not None and p2s_noisy is not None:
                        item["p2s_pred"] = p2s_pred
                        item["p2s_noisy"] = p2s_noisy
                        item["p2s_score"] = metric_to_score(p2s_pred, p2s_noisy)
            
            metrics.append(item)
        return metrics


