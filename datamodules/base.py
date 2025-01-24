from collections import defaultdict
from lightning import LightningDataModule
from torch.utils.data import  DataLoader, ConcatDataset, Dataset, Sampler
from itertools import chain

from metrics.metric_base import MetricDataModule
from utils import get_absolute_path

class DatasetLoaderModule(LightningDataModule):
    def __init__(self, module, cfg):
        super().__init__()
        self._module = module
        self.datasets: dict[str,list[Dataset]] = defaultdict(list)
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.reload_dataloaders_every_epoch = cfg.training.reload_dataloaders_every_epoch
        self.limit_train_batches = cfg.training.limit_train_batches
        self.sampler = None
        self.collate_fn = None

    def setup(self, stage):
        self.datasets: dict[str,list[Dataset]] = defaultdict(list)

    def train_dataloader(self) -> DataLoader:
        if not self.datasets["train"]:
            print("No training data found")
            return None
        
        if self.reload_dataloaders_every_epoch:
            current_idx = self._module.current_epoch % len(self.datasets["train"])
            dataset = self.datasets["train"][current_idx]
        else:
            dataset = ConcatDataset(self.datasets["train"])
        
        if self.sampler is not None and isinstance(self.sampler, BaseSampler):
            if hasattr(dataset, "data"):
                self.sampler.set_data(dataset.data)
            else:
                self.sampler.set_data(list(chain.from_iterable(d.data for d in self.datasets["train"])))

        return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True if self.sampler is None else False,
                sampler=self.sampler,
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

class BaseSampler(Sampler):
    def set_data(self, data):
        self.data = data

class BaseDataModule(DatasetLoaderModule, MetricDataModule):
    ...

class CombinedDataModule(BaseDataModule):
    def __init__(self, module, cfg, tokenizer, data_modules=[]):
        super().__init__(module, cfg)
        self.tokenizer = tokenizer
        self.cache_dir = cfg.cache_dir
        self.data_path = get_absolute_path(cfg.task.data_path)
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

def retain_forget_ratio_hook(datamodule, retain_forget_ratio):
    from types import MethodType
    def train_dataloader(self):
        if len(self.datasets["train"]) == 2:
            train_dataset = self.datasets["train"]
            self.datasets["train"] = [train_dataset[0]] + [train_dataset[1]] * retain_forget_ratio
        assert len(self.datasets["train"]) == retain_forget_ratio + 1
        return super(datamodule.__class__, self).train_dataloader()
    datamodule.train_dataloader = MethodType(train_dataloader, datamodule)