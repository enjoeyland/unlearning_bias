import torch
import pandas as pd

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class Callbacks:
    def __init__(self, cfg):
        self.output_dir = cfg.output_dir

        self.max_tolerance = cfg.callbacks.max_tolerance
        self.stop_step = cfg.callbacks.early_stop_step

        if cfg.method.name in ["negtaskvector"]:
            self.every_n_epochs = cfg.training.epochs    
        else:
            self.every_n_epochs = 1

        self.monitor = None
        self.mode = "min"
        self.filename = "best"


        if cfg.task.name in ["stereoset", "crows_pairs"] or ("combined" in cfg.task.name and ("stereoset" in cfg.task.targets or "crows_pairs" in cfg.task.targets)):
            if cfg.method.name == "dpo":
                ...
            elif cfg.method.fit_target == "forget":
                self.monitor = "valid/bias_score"
                self.mode = "max"
                self.filename = "ppl={valid/ppl/dataloader_idx_0:.2f}-bias_score={valid/bias_score:.4f}"
        
        elif cfg.task.name == "adult" or cfg.task.name == "compas":
            if cfg.method.name == "grad_ascent":
                self.monitor = "valid/equal_opportunity"
                self.mode = "min"
            else:
                self.monitor = "valid/accuracy"
                self.mode = "max"
            self.filename = "acc={valid/accuracy:.3f}-eo={valid/equal_opportunity:.4f}-spd={valid/spd:.4f}"
        else:
            print(f"Task {cfg.task.name} is not setup for callbacks.")  


        if cfg.method.name == "finetune":
            if cfg.method.fit_target == "forget":
                self.filename = f"forget_{self.filename}"
            elif cfg.method.fit_target == "retain":
                if hasattr(cfg.data, "retain_multiplier") and not cfg.data.retain_multiplier:
                    self.filename = f"retain{cfg.data.retain_multiplier}_{self.filename}"
                else:
                    self.filename = f"retain_{self.filename}"
        elif cfg.method.name == "grad_ascent":
            if cfg.method.fit_target == "without_retain":
                self.filename = f"without_retain_{self.filename}"
            elif cfg.method.fit_target == "with_retain":
                self.filename = f"with_retain_{self.filename}"

    def get_checkpoint_callback(self):
        return ModelCheckpoint(
            monitor=self.monitor,
            mode=self.mode,
            save_top_k=1,
            save_weights_only=True,
            save_last=False,
            dirpath=self.output_dir,
            filename=self.filename,
            verbose=True,
            auto_insert_metric_name=False,
            every_n_epochs=self.every_n_epochs,
        )
    
    def get_early_stopping(self):
        if self.max_tolerance == 0 or self.max_tolerance is None:
            return Callback()
        
        return EarlyStopping(
            monitor=self.monitor,
            mode=self.mode,
            patience=self.max_tolerance,
            verbose=True,
        )

    def get_early_stop_step(self):
        if self.stop_step == 0 or self.stop_step is None:
            return Callback()
        
        return EarlyStopStepCallback(
            stop_step=self.stop_step
        )

class EarlyStopStepCallback(Callback):
    def __init__(self, stop_step):
        super().__init__()
        self.stop_step = stop_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.stop_step:
            return
        
        if trainer.global_step >= self.stop_step:
            print(f"Stopping training at step {trainer.global_step}")
            trainer.should_stop = True
