import torch
from typing import List, Optional, Tuple
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
)
from torchmetrics.functional.classification.group_fairness import _groups_validation, _groups_format
from torchmetrics.functional.classification import group_fairness

def _binary_groups_stat_scores(
    preds: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute the true/false positives and true/false negatives rates for binary classification by group.

    Related to `Type I and Type II errors`_.

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, "global", ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, "global", ignore_index)
        _groups_validation(groups, num_groups)

    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    groups = _groups_format(groups).squeeze(1)

    group_preds = [preds[groups == g] if (groups == g).any() else torch.empty((0, 0), device=preds.device) for g in range(num_groups)]
    group_target = [target[groups == g] if (groups == g).any() else torch.empty((0, 0), device=target.device) for g in range(num_groups)]

    return [_binary_stat_scores_update(group_p, group_t) for group_p, group_t in zip(group_preds, group_target)]

group_fairness._binary_groups_stat_scores = _binary_groups_stat_scores


from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
from torchmetrics import classification

def update(self, preds: Tensor, target: Tensor, groups: Tensor) -> None:
    """Update state with predictions, groups, and target.

    Args:
        preds: Tensor with predictions.
        target: Tensor with true labels.
        groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

    """
    if self.task == "demographic_parity":
        if target is not None:
            rank_zero_warn("The task demographic_parity does not require a target.", UserWarning)
        target = torch.zeros(preds.shape, device=preds.device)

    group_stats = _binary_groups_stat_scores(
        preds, target, groups, self.num_groups, self.threshold, self.ignore_index, self.validate_args
    )

    self._update_states(group_stats)

classification.BinaryFairness.update = update