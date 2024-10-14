from torchmetrics import Metric, classification

from metrics.metric_base import MetricHandler


class BinaryAccuracy(classification.BinaryAccuracy, MetricHandler):
    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = outputs.logits.argmax(dim=1)
        target = batch["labels"]
        return self(preds, target)

class EqulityOfOpportunity(Metric, MetricHandler):
    """TPR of the privileged group  - TPR of the unprivileged group"""
    def __init__(self, feature_name):
        super().__init__()
        self.feature_name = feature_name
        self.privileged_recall = classification.BinaryRecall()
        self.unprivileged_recall = classification.BinaryRecall()
    
    def update(self, preds, target, feature):
        if len(preds[feature]) != 0:
            self.privileged_recall.update(preds[feature], target[feature])
        if len(preds[~feature]) != 0:
            self.unprivileged_recall.update(preds[~feature], target[~feature]) # <-- seed42 batch16으로 할때 ~feature가 없으면 멈춰버림 아직 원인 모름..
    
    def compute(self):
        result = self.privileged_recall.compute() - self.unprivileged_recall.compute()
        return result

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = outputs.logits.argmax(dim=1)
        target = batch["labels"]
        feature = batch[self.feature_name]
        return self(preds, target, feature)

    def on_epoch_end(self):
        return self.compute()