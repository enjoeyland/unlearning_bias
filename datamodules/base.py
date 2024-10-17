from pathlib import Path
from collections import defaultdict
from lightning import LightningDataModule
from torch.utils.data import  DataLoader, ConcatDataset

from metrics.metric_base import MetricDataModule

class DatasetLoaderModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.datasets = defaultdict(list)

    def train_dataloader(self):
        if not self.datasets["train"]:
            print("No training data found")
            return None
        
        return DataLoader(
            ConcatDataset(self.datasets["train"]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        if not self.datasets["valid"]:
            print("No validation data found")
            return None
        
        dataloaders = []
        for dataset in self.datasets["valid"]:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False
            )
            dataloaders.append(dataloader)
        return dataloaders

    def test_dataloader(self):
        if not self.datasets["test"]:
            print("No test data found")
            return None

        dataloaders = []
        for dataset in self.datasets["test"]:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False
            )
            dataloaders.append(dataloader)
        return dataloaders

class BaseDataModule(DatasetLoaderModule, MetricDataModule):
    ...

class CombinedDataModule(BaseDataModule):
    def __init__(self, cfg, tokenizer, data_modules=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = Path(__file__).parent.parent / cfg.task.data_path
        self.data_modules = data_modules
        for dm in self.data_modules:
            for split, metircs in dm.metrics.items():
                self.metrics[split].update(metircs)

    def prepare_data(self) -> None:
        for dm in self.data_modules:
            dm.prepare_data()

    def setup(self, stage=None):
        for dm in self.data_modules:
            dm.setup(stage)
            for split in dm.datasets:
                self.datasets[split].extend(dm.datasets[split])