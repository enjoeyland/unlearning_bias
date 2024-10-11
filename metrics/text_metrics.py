import torch
from torchmetrics import Metric
from metrics.metric_base import MetricHandler


class PPL(Metric, MetricHandler):
    def __init__(self):
        super().__init__()
    
    def update(self, loss):
        return torch.exp(loss)

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        loss = outputs.loss
        return self(loss)        