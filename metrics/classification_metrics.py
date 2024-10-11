from torchmetrics import classification
from metrics.metric_base import MetricHandler

class BinaryAccuracy(classification.BinaryAccuracy, MetricHandler):
    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = outputs.logits.argmax(dim=1)
        target = batch["labels"]
        return self(preds, target)
