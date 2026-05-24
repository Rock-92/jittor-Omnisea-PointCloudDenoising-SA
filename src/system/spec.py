from collections import defaultdict
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import json
import jittor as jt
import numpy as np
import os

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x

def _to_jittor(value):
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32, copy=False)
        return jt.array(value)
    if isinstance(value, dict):
        return {k: _to_jittor(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jittor(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_jittor(v) for v in value)
    return value

def get_optimizer(optimizer_config, model):
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        
        self._validation_loss = defaultdict(list)
        self._validation_scores = defaultdict(list)
        self.best_epoch = None
        self.best_validation_loss = float("inf")
        self.best_validation_score = float("-inf")
    
    def forward(self, batch, validate: bool=False): # return loss sum
        batch = _to_jittor(batch)
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_train_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
        self._validation_scores = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def validation_metric_step(self, batch):
        return None
    
    def record_validation_scores(self, metrics):
        if metrics is None:
            return
        if isinstance(metrics, dict):
            metrics = [metrics]
        for metric in metrics:
            for name, value in metric.items():
                if value is None:
                    continue
                self._validation_scores[name].append(float(_get_item(value)))
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def get_validation_loss_sum(self):
        loss_values = []
        for name, values in self._validation_loss.items():
            if name.endswith("_loss_sum"):
                loss_values.extend(values)
        if len(loss_values) == 0:
            return None
        return sum(loss_values) / len(loss_values)
    
    def get_validation_score_summary(self):
        if len(self._validation_scores) == 0:
            return None
        summary = {
            name: sum(values) / len(values)
            for name, values in self._validation_scores.items()
            if len(values) > 0
        }
        if "cd_score" in summary and "p2s_score" in summary:
            summary["final_score"] = (
                0.5 * summary["cd_score"] + 0.5 * summary["p2s_score"]
            )
        elif "cd_score" in summary:
            summary["final_score"] = summary["cd_score"]
        return summary
    
    def log_validation_epoch(self, epoch, validation_loss, score_summary):
        loss_text = (
            "nan" if validation_loss is None else f"{validation_loss:.8f}"
        )
        if score_summary is None or "final_score" not in score_summary:
            print(f"Epoch {epoch} validation loss={loss_text}")
            return
        print(
            f"Epoch {epoch} validation: "
            f"loss={loss_text}, "
            f"CD score={score_summary.get('cd_score', 0.0):.4f}, "
            f"P2S score={score_summary.get('p2s_score', 0.0):.4f}, "
            f"Final score={score_summary['final_score']:.4f}, "
            f"CD pred={score_summary.get('cd_pred', 0.0):.8f}, "
            f"CD noisy={score_summary.get('cd_noisy', 0.0):.8f}, "
            f"P2S pred={score_summary.get('p2s_pred', 0.0):.8f}, "
            f"P2S noisy={score_summary.get('p2s_noisy', 0.0):.8f}"
        )
    
    def save_best_checkpoint(self, epoch, validation_loss, score_summary=None):
        validation_score = None
        if score_summary is not None:
            validation_score = score_summary.get("final_score")
        if validation_score is not None:
            if validation_score < self.best_validation_score:
                return
            if (
                validation_score == self.best_validation_score
                and (
                    validation_loss is None
                    or validation_loss >= self.best_validation_loss
                )
            ):
                return
        elif validation_loss is not None:
            if validation_loss >= self.best_validation_loss:
                return
        else:
            return
        
        self.best_epoch = epoch
        if validation_loss is not None:
            self.best_validation_loss = validation_loss
        if validation_score is not None:
            self.best_validation_score = validation_score
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.ckpt_save_dir,
            f"{self.ckpt_save_name}_best.pkl",
        )
        metadata_path = os.path.join(
            self.ckpt_save_dir,
            f"{self.ckpt_save_name}_best.json",
        )
        self.model.save(checkpoint_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": self.best_epoch,
                    "validation_loss": self.best_validation_loss,
                    "validation_score": (
                        None
                        if self.best_validation_score == float("-inf")
                        else self.best_validation_score
                    ),
                    "selection_metric": (
                        "max_final_score"
                        if validation_score is not None
                        else "min_validation_loss"
                    ),
                    "checkpoint": checkpoint_path,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"Saved best checkpoint: epoch={self.best_epoch}, "
            f"validation_loss={self.best_validation_loss}, "
            f"validation_score={validation_score}"
        )
    
    def on_before_optimizer_step(self, optimizer):
        pass
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch = _to_jittor(batch)
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        for epoch in range(self.epochs):
            self.model.train()
            self.on_train_epoch_start()
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size) # type: ignore
            for batch in pbar:
                batch = _to_jittor(batch)
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                pbar.set_description(f"Epoch {epoch}, Loss: {_get_item(loss)}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            
            self.model.eval()
            validate_dataloader = self.dataset_module.validate_dataloader()
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size)
                        for batch in pbar:
                            batch = _to_jittor(batch)
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            self.record_validation_scores(
                                self.validation_metric_step(batch)
                            )
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size)
                    for batch in pbar:
                        batch = _to_jittor(batch)
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        self.record_validation_scores(
                            self.validation_metric_step(batch)
                        )
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()
                validation_loss = self.get_validation_loss_sum()
                score_summary = self.get_validation_score_summary()
                self.log_validation_epoch(epoch, validation_loss, score_summary)
                self.save_best_checkpoint(epoch, validation_loss, score_summary)
            
            checkpoint_path = os.path.join(self.ckpt_save_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
            os.makedirs(self.ckpt_save_dir, exist_ok=True)
            self.model.save(checkpoint_path)
        self.on_train_end()
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                batch = _to_jittor(batch)
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")
                self.on_predict_batch_end()
        self.on_predict_epoch_end()
