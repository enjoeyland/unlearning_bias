from pathlib import Path
from collections import defaultdict
from lightning import LightningDataModule
from torch.utils.data import  DataLoader, ConcatDataset, Dataset

from metrics.metric_base import MetricDataModule

class DatasetLoaderModule(LightningDataModule):
    def __init__(self, module, cfg):
        super().__init__()
        self._module = module
        self.datasets: dict[str,list[Dataset]] = defaultdict(list)
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.reload_dataloaders_every_epoch = cfg.training.reload_dataloaders_every_epoch
        self.limit_train_batches = cfg.training.limit_train_batches
        self.collate_fn = None

    def train_dataloader(self) -> DataLoader:
        if not self.datasets["train"]:
            print("No training data found")
            return None
        
        if self.reload_dataloaders_every_epoch:
            current_idx = self._module.current_epoch  % len(self.datasets["train"])
            dataset = self.datasets["train"][current_idx]
        else:
            dataset = ConcatDataset(self.datasets["train"])
        
        return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=self.collate_fn
            )

    def val_dataloader(self) -> list[DataLoader]:
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
                shuffle=False,
                collate_fn=self.collate_fn
            )
            dataloaders.append(dataloader)
        return dataloaders

    def test_dataloader(self) -> list[DataLoader]:
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
                shuffle=False,
                collate_fn=self.collate_fn
            )
            dataloaders.append(dataloader)
        return dataloaders

class BaseDataModule(DatasetLoaderModule, MetricDataModule):
    ...

class CombinedDataModule(BaseDataModule):
    def __init__(self, module, cfg, tokenizer, data_modules=[]):
        super().__init__(module, cfg)
        self.tokenizer = tokenizer
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