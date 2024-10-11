from collections import defaultdict
from torch.utils.data import  DataLoader, ConcatDataset

from metrics.metric_base import MetricDataModule

class BaseDataModule(MetricDataModule):
    def __init__(self):
        super().__init__()
        self.datasets = defaultdict(list)

    def train_dataloader(self):
        if not self.datasets["train"]:
            raise ValueError("No training data found")
        
        return DataLoader(
            ConcatDataset(self.datasets["train"]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        if not self.datasets["valid"]:
            raise ValueError("No validation data found")
        
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
            raise ValueError("No test data found")

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