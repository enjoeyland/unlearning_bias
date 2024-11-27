import torch
from torchmetrics import Metric, classification
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.functional.classification.group_fairness import _binary_groups_stat_scores
from metrics.metric_base import MetricHandler


class BinaryAccuracy(classification.BinaryAccuracy, MetricHandler):
    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = outputs.logits.argmax(dim=1)
        target = batch["labels"]
        print("device:",preds.device, target.device)
        return self(preds.to("cuda:0"), target.to("cuda:0"))

class EqulityOfOpportunity(classification.BinaryFairness, MetricHandler):
    def __init__(self, group_name, num_groups, **kwargs):
        super().__init__(num_groups=num_groups, task="equal_opportunity", **kwargs)
        self.group_name = group_name

    def compute(self):
        true_pos_rates = _safe_divide(self.tp, self.tp + self.fn)
        min_pos_rate_id = torch.argmin(true_pos_rates)
        max_pos_rate_id = torch.argmax(true_pos_rates)

        return true_pos_rates[max_pos_rate_id] - true_pos_rates[min_pos_rate_id]

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = outputs.logits.argmax(dim=1)
        target = batch["labels"]
        groups = batch[self.group_name].long()
        return self(preds, target, groups)

class StatisticalParityDifference(classification.BinaryFairness, MetricHandler):
    def __init__(self, group_name, num_groups, **kwargs):
        super().__init__(num_groups=num_groups, task="demographic_parity", **kwargs)
        self.group_name = group_name

    def update(self, preds, target, groups):
        target = torch.zeros(preds.shape, device=preds.device)

        group_stats = _binary_groups_stat_scores(
            preds, target, groups, self.num_groups, self.threshold, self.ignore_index, self.validate_args
        )

        self._update_states(group_stats)

    def compute(self):
        pos_rates = _safe_divide(self.tp + self.fp, self.tp + self.fp + self.tn + self.fn)
        min_pos_rate_id = torch.argmin(pos_rates)
        max_pos_rate_id = torch.argmax(pos_rates)

        return pos_rates[max_pos_rate_id] - pos_rates[min_pos_rate_id]

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = outputs.logits.argmax(dim=1)
        groups = batch[self.group_name].long()
        return self(preds, None, groups)