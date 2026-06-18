import csv
import json
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .spec import (
    DummySystem,
    _get_item,
    _to_jittor,
    get_optimizer_lr,
)
from .vm import VMSystem


def mean_metric(values, default=None):
    if not values:
        return default
    return float(sum(values) / len(values))


class ShapePretrainSystem(DummySystem):
    """Training loop with explicit masked reconstruction diagnostics."""

    metric_names = [
        "pretrain_loss",
        "masked_reconstruction_loss",
        "masked_chamfer",
        "masked_rmse",
        "masked_fscore",
        "masked_precision",
        "masked_recall",
        "all_reconstruction_loss",
        "all_chamfer",
        "all_rmse",
        "all_fscore",
        "all_precision",
        "all_recall",
        "center_displacement_loss",
        "center_rmse",
        "center_cosine",
        "geometry_loss",
        "token_distill_loss",
        "token_cosine",
        "normal_loss",
        "normal_cosine_abs",
        "crease_loss",
        "crease_pred_mean",
        "crease_target_mean",
        "consistency_loss",
        "mask_ratio",
        "noise_std",
    ]

    def _summary(self, storage, prefix):
        summary = {}
        for name in self.metric_names:
            values = storage.get(f"{prefix}/{name}", [])
            summary[name] = mean_metric(values)
        return summary

    def _write_epoch_log(self, epoch, train_summary, val_summary, seconds):
        if self.run_dir is None:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        record = {
            "epoch": epoch,
            "lr": get_optimizer_lr(self.optimizer),
            "seconds": seconds,
        }
        for prefix, summary in (
            ("train", train_summary),
            ("val", val_summary),
        ):
            for name, value in summary.items():
                record[f"{prefix}_{name}"] = value
        self._epoch_metric_records.append(record)
        fields = list(record.keys())
        path = os.path.join(self.run_dir, "epoch_log.csv")
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._epoch_metric_records)

    def _save_best(self, epoch, val_summary):
        value = val_summary.get("pretrain_loss")
        if value is None or value >= self.best_validation_loss:
            return
        self.best_validation_loss = value
        self.best_epoch = epoch
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        checkpoint = os.path.join(
            self.ckpt_save_dir,
            f"{self.ckpt_save_name}_best.pkl",
        )
        metadata = os.path.join(
            self.ckpt_save_dir,
            f"{self.ckpt_save_name}_best.json",
        )
        self.model.save(checkpoint)
        with open(metadata, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "best_epoch": epoch,
                    "selection_metric": "min_val_pretrain_loss",
                    "val_metrics": val_summary,
                    "checkpoint": checkpoint,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(
            "Saved best shape processor: "
            f"epoch={epoch}, pretrain_loss="
            f"{val_summary.get('pretrain_loss', float('nan')):.4f}, "
            f"masked_cd="
            f"{val_summary.get('masked_chamfer', float('nan')):.8f}, "
            f"fscore={val_summary.get('masked_fscore', float('nan')):.4f}, "
            f"center_rmse={val_summary.get('center_rmse', float('nan')):.5f}, "
            f"center_cos={val_summary.get('center_cosine', float('nan')):.3f}, "
            f"token_cos={val_summary.get('token_cosine', float('nan')):.3f}, "
            f"normal_cos={val_summary.get('normal_cosine_abs', float('nan')):.3f}, "
            f"crease={val_summary.get('crease_pred_mean', float('nan')):.3f}/"
            f"{val_summary.get('crease_target_mean', float('nan')):.3f}"
        )

    def train(self):
        import time

        if self.optimizer is None:
            raise ValueError("optimizer is required")
        self.model.set_predict(False)
        for epoch in range(self.epochs):
            self._current_epoch = epoch
            epoch_start = time.time()
            self.model.train()
            self.on_train_epoch_start()
            train_loader = self.dataset_module.train_dataloader()
            if train_loader is None:
                raise ValueError("train dataloader is required")
            total = len(train_loader) // train_loader.batch_size
            bar = tqdm(
                train_loader,
                total=total,
                desc=f"Pretrain epoch {epoch}",
                unit="batch",
            )
            for batch in bar:
                batch = _to_jittor(batch)
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.record_train_losses(self._last_train_loss_dict)
                values = {
                    name: float(_get_item(self._last_train_loss_dict[name]))
                    for name in self.metric_names
                }
                bar.set_postfix(
                    loss=f"{values['pretrain_loss']:.4f}",
                    mcd=f"{values['masked_chamfer']:.6f}",
                    acd=f"{values['all_chamfer']:.6f}",
                    crmse=f"{values['center_rmse']:.5f}",
                    ccos=f"{values['center_cosine']:.3f}",
                    fscore=f"{values['masked_fscore']:.3f}",
                    tok=f"{values['token_cosine']:.3f}",
                    ncos=f"{values['normal_cosine_abs']:.3f}",
                    crease=f"{values['crease_pred_mean']:.2f}/"
                    f"{values['crease_target_mean']:.2f}",
                    mask=f"{values['mask_ratio']:.2f}",
                    sigma=f"{values['noise_std']:.4f}",
                )
            train_summary = self._summary(self._train_loss, "train")

            self.model.eval()
            self.on_validation_epoch_start()
            val_loader = self.dataset_module.validate_dataloader()
            if val_loader is None:
                raise ValueError("validation dataloader is required")
            loaders = (
                val_loader.values()
                if isinstance(val_loader, dict)
                else [val_loader]
            )
            for loader in loaders:
                total = len(loader) // loader.batch_size
                bar = tqdm(
                    loader,
                    total=total,
                    desc=f"Validate epoch {epoch}",
                    unit="batch",
                )
                for batch in bar:
                    batch = _to_jittor(batch)
                    self.validation_step(batch)
                    values = {}
                    for name in self.metric_names:
                        candidates = [
                            values
                            for key, values in self._validation_loss.items()
                            if key.endswith(f"_{name}")
                        ]
                        flat = [
                            item
                            for candidate in candidates
                            for item in candidate
                        ]
                        values[name] = flat[-1] if flat else float("nan")
                    bar.set_postfix(
                        loss=f"{values['pretrain_loss']:.4f}",
                        mcd=f"{values['masked_chamfer']:.6f}",
                        acd=f"{values['all_chamfer']:.6f}",
                        crmse=f"{values['center_rmse']:.5f}",
                        ccos=f"{values['center_cosine']:.3f}",
                        fscore=f"{values['masked_fscore']:.3f}",
                        tok=f"{values['token_cosine']:.3f}",
                        ncos=f"{values['normal_cosine_abs']:.3f}",
                        crease=f"{values['crease_pred_mean']:.2f}/"
                        f"{values['crease_target_mean']:.2f}",
                        mask=f"{values['mask_ratio']:.2f}",
                        sigma=f"{values['noise_std']:.4f}",
                    )
            val_summary = {}
            for name in self.metric_names:
                candidates = [
                    values
                    for key, values in self._validation_loss.items()
                    if key.endswith(f"_{name}")
                ]
                val_summary[name] = mean_metric(
                    [
                        item
                        for candidate in candidates
                        for item in candidate
                    ]
                )
            seconds = time.time() - epoch_start
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_summary['pretrain_loss']:.4f}, "
                f"val_loss={val_summary['pretrain_loss']:.4f}, "
                f"val_cd={val_summary['masked_chamfer']:.8f}, "
                f"val_all_cd={val_summary['all_chamfer']:.8f}, "
                f"center_rmse={val_summary['center_rmse']:.6f}, "
                f"center_cos={val_summary['center_cosine']:.4f}, "
                f"geometry={val_summary['geometry_loss']:.6f}, "
                f"token_cos={val_summary['token_cosine']:.4f}, "
                f"normal_cos={val_summary['normal_cosine_abs']:.4f}, "
                f"crease={val_summary['crease_pred_mean']:.3f}/"
                f"{val_summary['crease_target_mean']:.3f}, "
                f"consistency={val_summary['consistency_loss']:.6f}, "
                f"val_fscore={val_summary['masked_fscore']:.4f}, "
                f"mask={val_summary['mask_ratio']:.3f}, "
                f"sigma={val_summary['noise_std']:.4f}, "
                f"time={seconds:.1f}s"
            )
            self._save_best(epoch, val_summary)
            self._write_epoch_log(
                epoch,
                train_summary,
                val_summary,
                seconds,
            )
            self.step_scheduler(
                epoch,
                train_loss=train_summary["pretrain_loss"],
                validation_loss=val_summary["pretrain_loss"],
            )
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self.model.save(
                os.path.join(
                    self.ckpt_save_dir,
                    f"{self.ckpt_save_name}_{epoch}.pkl",
                )
            )


class ShapeContextVMSystem(VMSystem):
    def __init__(
        self,
        *args,
        freeze_shape_epochs=5,
        shape_lr_scale=0.1,
        **kwargs,
    ):
        self.freeze_shape_epochs = int(freeze_shape_epochs)
        self.shape_lr_scale = float(shape_lr_scale)
        super().__init__(*args, **kwargs)
        self._shape_parameters = list(
            self.model.get_shape_train_parameters()
        )

    def _set_shape_grad(self, enabled):
        for parameter in self._shape_parameters:
            if enabled:
                parameter.start_grad()
            else:
                parameter.stop_grad()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._set_shape_grad(
            self._current_epoch >= self.freeze_shape_epochs
        )

    def on_before_optimizer_step(self, optimizer):
        if self._current_epoch < self.freeze_shape_epochs:
            return
        if abs(self.shape_lr_scale - 1.0) < 1e-12:
            return
        for parameter in self._shape_parameters:
            gradient = parameter.opt_grad(optimizer)
            if gradient is not None:
                gradient.update(gradient * self.shape_lr_scale)

    def train_progress_postfix(self):
        values = getattr(self, "_last_train_loss_dict", None)
        if not values:
            return {}

        def value(name, default=float("nan")):
            item = values.get(name)
            if item is None:
                return default
            return float(_get_item(item))

        return {
            "gate": f"{value('region_context_gate'):.3f}",
            "crease": f"{value('region_crease_mean'):.3f}",
            "prior": f"{value('region_prior_delta'):.3f}",
        }

    def log_epoch_metrics(
        self,
        epoch,
        train_loss=None,
        validation_loss=None,
        score_summary=None,
    ):
        if self.run_dir is None:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        if score_summary is None:
            score_summary = {}
        record = {
            "epoch": epoch,
            "lr": get_optimizer_lr(self.optimizer),
            "train_loss": train_loss,
            "region_context_gate": mean_metric(
                self._train_loss.get("train/region_context_gate", [])
            ),
            "region_crease_mean": mean_metric(
                self._train_loss.get("train/region_crease_mean", [])
            ),
            "region_prior_delta": mean_metric(
                self._train_loss.get("train/region_prior_delta", [])
            ),
            "val_loss": validation_loss,
            "cd_score": score_summary.get("cd_score"),
            "p2s_score": score_summary.get("p2s_score"),
            "final_score": score_summary.get("final_score"),
        }
        self._epoch_metric_records.append(record)
        path = os.path.join(self.run_dir, "epoch_log.csv")
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(record.keys()),
            )
            writer.writeheader()
            writer.writerows(self._epoch_metric_records)

        def fmt(name, digits=3):
            value = record.get(name)
            if value is None:
                return "nan"
            return f"{value:.{digits}f}"

        print(
            "Shape-context train diagnostics: "
            f"gate={fmt('region_context_gate')}, "
            f"crease={fmt('region_crease_mean')}, "
            f"prior_delta={fmt('region_prior_delta')}"
        )

    def on_train_end(self):
        self._set_shape_grad(True)
        super().on_train_end()
