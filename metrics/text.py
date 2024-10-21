import torch
from torchmetrics import Metric, text

from metrics.metric_base import MetricHandler

class Perplexity(text.Perplexity, MetricHandler):
    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        return self(outputs.logits[:, :-1], batch["labels"][:, 1:])