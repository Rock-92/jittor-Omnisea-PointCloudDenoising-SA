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

    def estimate_edm_sigma_data(self):
        if not getattr(self.model, "use_edm", False):
            return
        train_dataloader = self.dataset_module.train_dataloader()
        if train_dataloader is None:
            return

        total = 0
        total_sum = 0.0
        total_sq_sum = 0.0
        from tqdm import tqdm

        pbar = tqdm(
            train_dataloader,
            total=len(train_dataloader) // train_dataloader.batch_size,
        )
        for batch in pbar:
            pc_clean = batch["pc_clean"]
            if isinstance(pc_clean, jt.Var):
                pc_clean = pc_clean.detach().numpy()
            pc_clean = np.asarray(pc_clean, dtype=np.float64)
            total += pc_clean.size
            total_sum += float(pc_clean.sum())
            total_sq_sum += float((pc_clean ** 2.0).sum())
            pbar.set_description("Estimating EDM sigma_data")

        if total == 0:
            return
        mean = total_sum / total
        variance = max(total_sq_sum / total - mean * mean, 1e-12)
        sigma_data = float(np.sqrt(variance))
        self.model.sigma_data = sigma_data
        print(f"Estimated EDM sigma_data from train patches: {sigma_data:.8f}")
    
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
        if (
            "dataset_clean" in parts
            and len(parts) >= 2
            and parts[-2] == "models"
            and parts[-1] == "model_normalized.obj"
        ):
            return Path(path)
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
        
        if hasattr(self.model, "validation_predict"):
            pc_pred = self.model.validation_predict(batch)
        elif getattr(self.model, "use_edm", False) and "score_sigma" in batch:
            score_sigma = batch["score_sigma"].reshape(pc_noisy.shape[0], 1)
            pc_pred = self.model.predict_clean(pc_noisy, sigma=score_sigma)
        else:
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


class VMSSLSystem(VMSystem):
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
        ssl_pretrained_ckpt: Optional[str]=None,
        freeze_global_epochs: int=8,
        global_lr_scale: float=0.2,
    ):
        self.ssl_pretrained_ckpt = ssl_pretrained_ckpt
        self.freeze_global_epochs = int(freeze_global_epochs)
        self.global_lr_scale = float(global_lr_scale)
        self._current_epoch = 0
        self._global_params = []
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
        self._global_params = self._collect_global_params()
        if self.ssl_pretrained_ckpt:
            self._load_ssl_pretrained(self.ssl_pretrained_ckpt)

    def _collect_global_params(self):
        params = []
        if not hasattr(self.model, "encoder"):
            return params
        enc = self.model.encoder
        for module in [
            getattr(enc, "input_proj_1", None),
            getattr(enc, "input_proj_2", None),
            getattr(enc, "global_token_generator", None),
        ]:
            if module is not None:
                params.extend(list(module.parameters()))
        return params

    def _load_ssl_pretrained(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"ssl_pretrained_ckpt not found: {path}")
        state = jt.load(path)
        if not isinstance(state, dict):
            raise ValueError(f"expected state_dict checkpoint, got {type(state)}: {path}")

        def sub_state(prefix: str):
            plen = len(prefix)
            return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}

        self.model.encoder.input_proj_1.load_state_dict(sub_state("encoder.input_proj_1."))
        self.model.encoder.input_proj_2.load_state_dict(sub_state("encoder.input_proj_2."))
        self.model.encoder.global_token_generator.load_state_dict(
            sub_state("encoder.global_token_generator.")
        )
        print(f"Loaded SSL global token weights: {path}")

    def _set_global_requires_grad(self, enabled: bool):
        for p in self._global_params:
            if enabled:
                p.start_grad()
            else:
                p.stop_grad()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._set_global_requires_grad(self._current_epoch >= self.freeze_global_epochs)

    def on_before_optimizer_step(self, optimizer):
        if self._current_epoch < self.freeze_global_epochs:
            return
        if abs(self.global_lr_scale - 1.0) < 1e-12:
            return
        for p in self._global_params:
            grad = p.opt_grad(optimizer)
            if grad is not None:
                grad.update(grad * self.global_lr_scale)

    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        self.estimate_edm_sigma_data()
        for epoch in range(self.epochs):
            self._current_epoch = epoch
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            from tqdm import tqdm

            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size)
            for batch in pbar:
                from .spec import _get_item, _to_jittor

                batch = _to_jittor(batch)
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.record_train_losses(self._last_train_loss_dict)
                self.on_train_batch_end()
            self.on_train_epoch_end()
            train_loss = self.get_train_loss_sum()

            self.model.eval()
            validate_dataloader = self.dataset_module.validate_dataloader()
            validation_loss = None
            score_summary = None
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        for batch in pbar:
                            from .spec import _get_item, _to_jittor

                            batch = _to_jittor(batch)
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            self.record_validation_scores(self.validation_metric_step(batch))
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    for batch in pbar:
                        from .spec import _get_item, _to_jittor

                        batch = _to_jittor(batch)
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        self.record_validation_scores(self.validation_metric_step(batch))
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()
                validation_loss = self.get_validation_loss_sum()
                score_summary = self.get_validation_score_summary()
                self.log_validation_epoch(epoch, validation_loss, score_summary)
                self.save_best_checkpoint(epoch, validation_loss, score_summary)
            else:
                self.save_best_train_checkpoint(epoch, train_loss)
            self.log_epoch_metrics(epoch, train_loss, validation_loss, score_summary)
            self.step_scheduler(epoch, train_loss, validation_loss, score_summary)

            checkpoint_path = os.path.join(self.ckpt_save_dir, f"{self.ckpt_save_name}_{epoch}.pkl")
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self.model.save(checkpoint_path)
        self._set_global_requires_grad(True)
        self.on_train_end()
