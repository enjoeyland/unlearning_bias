from torch.nn import ModuleDict, Module
from lightning import LightningDataModule
from typing import Literal

class MetricHandler:
    def on_step(self, split: Literal["train", "valid", "test"], outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        ...
    def on_epoch_end(self, split, *args, **kwargs):
        ...

class MetricDataModule(LightningDataModule, MetricHandler):
    def __init__(self):
        super().__init__()
        self.metrics: Module = ModuleDict({
            "_train": ModuleDict(),
            "_valid": ModuleDict(),
            "_test": ModuleDict(),
        })
        
    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        metrics = {}
        for name, metric in self.metrics[f"_{split}"].items():
            try:
                result = metric.on_step(split, outputs, batch, batch_idx, dataloader_idx, *args, **kwargs)
            except Exception as e:
                print(f"Error in metric {name}")
                raise e

            if result is not None:
                metrics[f"{split}/{name}"] = result
        return metrics
    
    def on_epoch_end(self, split, *args, **kwargs):
        metrics = {}
        for name, metric in self.metrics[f"_{split}"].items():
            try:
                result = metric.on_epoch_end(split, *args, **kwargs)
            except Exception as e:
                print(f"Error in metric {name}")
                raise e

            if result is not None:
                metrics[f"{split}/{name}"] = result
        return metrics
