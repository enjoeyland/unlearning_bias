from torch.nn import ModuleDict, Module
from lightning import LightningDataModule

class MetricDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.metrics: Module = ModuleDict()