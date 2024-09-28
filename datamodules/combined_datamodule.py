from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from metric_logging import MetricLogger, MetricDataModule


class CombinedDataModule(MetricDataModule):
    def __init__(self, cfg, tokenizer, data_modules=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = Path(__file__).parent.parent / cfg.task.data_path
        self.data_modules = data_modules
        self.metric_logger = CombinedMetricLogger([dm.metric_logger for dm in self.data_modules])

    def prepare_data(self) -> None:
        for dm in self.data_modules:
            dm.prepare_data()

    def setup(self, stage=None):
        for dm in self.data_modules:
            dm.setup(stage)

    def train_dataloader(self):
        return DataLoader(
            ConcatDataset([dm.datasets["train"] for dm in self.data_modules if "train" in dm.datasets]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
    
    def val_dataloader(self):
        valid_data = []
        for dm in self.data_modules:
            if "valid" in dm.datasets:
                valid_data.extend(dm.datasets["valid"])
                
        return [DataLoader(
            vd,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        ) for vd in valid_data]

    def test_dataloader(self):
        test_data = []
        for dm in self.data_modules:
            if "test" in dm.datasets:
                test_data.extend(dm.datasets["test"])

        return [DataLoader(
            td,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        ) for td in test_data]

class CombinedMetricLogger(MetricLogger):
    def __init__(self, callbacks=[]):
        super().__init__(None)
        self.callbacks = callbacks

    def on_training_step(self, pl_module, outputs, batch, batch_idx):
        for cb in self.callbacks:
            cb.on_training_step(pl_module, outputs, batch, batch_idx)
    
    def on_validation_step(self, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for cb in self.callbacks:
            cb.on_validation_step(pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_step(self, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for cb in self.callbacks:
            cb.on_test_step(pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_fit_start(self, trainer, pl_module):
        for cb in self.callbacks:
            cb.on_fit_start(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        for cb in self.callbacks:
            cb.on_validation_epoch_end(trainer, pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        for cb in self.callbacks:
            cb.on_validation_end(trainer, pl_module)
    
    def on_test_epoch_end(self, trainer, pl_module):
        for cb in self.callbacks:
            cb.on_test_epoch_end(trainer, pl_module)
        
