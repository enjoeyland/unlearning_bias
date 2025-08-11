import torch
import pandas as pd

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from datamodules import DataModuleFactory

class Callbacks:
    def __init__(self, cfg):
        self.output_dir = cfg.output_dir

        self.max_tolerance = cfg.callbacks.max_tolerance
        self.stop_step = cfg.callbacks.early_stop_step

        if cfg.method.name in ["negtaskvector"]:
            self.every_n_epochs = cfg.training.epochs    
        else:
            self.every_n_epochs = 1

        self.monitor, self.mode, self.filename = DataModuleFactory.configure_callbacks_monitor(cfg)
        self.model_checkpoint = None

    def get_checkpoint_callback(self):
        self.model_checkpoint = ModelCheckpoint(
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
        return self.model_checkpoint
    
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

    def get_ckpt_path(self):
        if not self.model_checkpoint:
            return None
        return self.model_checkpoint._last_checkpoint_saved

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
