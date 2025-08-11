import torch

from torch import Tensor
from torchmetrics import Metric, classification
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.functional.classification.group_fairness import _binary_groups_stat_scores

from .metric_base import MetricHandler

class BinaryAccuracy(classification.BinaryAccuracy, MetricHandler):
    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = kwargs.get("preds", outputs.logits.argmax(dim=1))
        target = kwargs.get("target", batch["labels"])
        self(preds, target)
        return self

class GroupAccuracy(classification.BinaryAccuracy, MetricHandler):
    def __init__(self, group_name, group, **kwargs):
        super().__init__(**kwargs)
        self.group_name = group_name
        self.group = group

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = kwargs.get("preds", outputs.logits.argmax(dim=1))
        target = kwargs.get("target", batch["labels"])
        groups = batch[self.group_name]

        group_mask = groups == self.group
        if group_mask.sum() == 0:
            return None
        self(preds[group_mask], target[group_mask])
        return self

class EqulityOfOpportunity(classification.BinaryFairness, MetricHandler):
    def __init__(self, group_name, num_groups, **kwargs):
        super().__init__(num_groups=num_groups, task="equal_opportunity", **kwargs)
        self.group_name = group_name

    def compute(self):
        true_pos_rates = _safe_divide(self.tp, self.tp + self.fn)
        min_pos_rate_id = torch.argmin(true_pos_rates)
        max_pos_rate_id = torch.argmax(true_pos_rates)

        if (self.tp[max_pos_rate_id] + self.fn[max_pos_rate_id] + self.tn[max_pos_rate_id] + self.fp[max_pos_rate_id]) >= 128:
            print(f"EoO: tp: {self.tp}, fn: {self.fn}, tn: {self.tn}, fp: {self.fp}")

        return true_pos_rates[max_pos_rate_id] - true_pos_rates[min_pos_rate_id]

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = kwargs.get("preds", outputs.logits.argmax(dim=1))
        target = kwargs.get("target", batch["labels"])
        groups = batch[self.group_name]
        # print(f"EoO: preds: {preds}, target: {target}, groups: {groups}")
        self(preds, target, groups)
        return self

class StatisticalParityDifference(classification.BinaryFairness, MetricHandler):
    def __init__(self, group_name, num_groups, **kwargs):
        super().__init__(num_groups=num_groups, task="demographic_parity", **kwargs)
        self.group_name = group_name

    def compute(self):
        pos_rates = _safe_divide(self.tp + self.fp, self.tp + self.fp + self.tn + self.fn)
        min_pos_rate_id = torch.argmin(pos_rates)
        max_pos_rate_id = torch.argmax(pos_rates)

        if (self.tp[max_pos_rate_id] + self.fn[max_pos_rate_id] + self.tn[max_pos_rate_id] + self.fp[max_pos_rate_id]) >= 128:
            print(f"SPD: tp: {self.tp}, fn: {self.fn}, tn: {self.tn}, fp: {self.fp}")

        return pos_rates[max_pos_rate_id] - pos_rates[min_pos_rate_id]

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = kwargs.get("preds", outputs.logits.argmax(dim=1))
        groups = batch[self.group_name]
        # print(f"SPD: preds: {preds}, groups: {groups}")
        self(preds, None, groups)
        return self


class BalancedAccuracy(Metric, MetricHandler):
    tp: Tensor
    fp: Tensor
    tn: Tensor
    fn: Tensor

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        - preds: 모델의 예측값 (Binary: 0 또는 1)
        - target: 실제 정답 (Binary: 0 또는 1)
        """
        preds = preds.int()
        target = target.int()

        self.tp += torch.sum((preds == 1) & (target == 1))
        self.fn += torch.sum((preds == 0) & (target == 1))
        self.tn += torch.sum((preds == 0) & (target == 0))
        self.fp += torch.sum((preds == 1) & (target == 0))

    def compute(self):
        true_pos_rates = _safe_divide(self.tp, self.tp + self.fn)
        true_neg_rates = _safe_divide(self.tn, self.tn + self.fp)
        balanced_accuracies = (true_pos_rates + true_neg_rates) / 2
        return balanced_accuracies

    def on_step(self, split, outputs, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        preds = kwargs.get("preds", outputs.logits.argmax(dim=1))
        target = kwargs.get("target", batch["labels"])
        self(preds, target)
        return self