from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from metric_logging import MetricDataModule


class CombinedDataModule(MetricDataModule):
    def __init__(self, cfg, tokenizer, data_modules=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = cfg.training.per_device_batch_size
        self.num_workers = cfg.data.num_workers
        self.cache_dir = cfg.cache_dir
        self.data_path = Path(__file__).parent.parent / cfg.task.data_path
        self.data_modules = data_modules
        self.metrics = [dm.metrics for dm in self.data_modules]

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

        
