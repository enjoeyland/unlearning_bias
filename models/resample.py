import torch
import numpy as np

from .base import BaseModel
from datamodules import BaseSampler

class ResampleModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.datamodule.sampler = EOWeightedSampler(self.datamodule.batch_size, self.hparams.training.gradient_accumulation_steps, self.datamodule.sensitive_attribute)

    def _get_loss_and_metrics(self, outputs, batch, batch_idx):
        loss = outputs.loss
        metrics = {"train/loss": loss}

        self.datamodule.sampler.update_weights(outputs, batch)
        
        return loss, metrics


class EOWeightedSampler(BaseSampler):
    def __init__(self, batch_size, gradient_accumulation_steps, sensitive_attribute):
        """
        Args:
            batch_size: Number of samples per batch
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights
            sensitive_attribute: name of the sensitive attribute in the batch
        """
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.sensitive_attribute = sensitive_attribute
        
        self.group_labels = None
        self.groups = None
        self.num_samples = 0

        self.weights = None
        self.accumulated_probs = {0: [], 1: []}

    def set_data(self, data):
        self.group_labels = torch.tensor([item[self.sensitive_attribute] for item in data])
        self.groups = torch.unique(self.group_labels).tolist()
        self.num_samples = len(self.group_labels)
        print(f"number of sample: {self.num_samples}")
        self.weights = self.initialize_uniform_weights()
        print(self.weights)

    def initialize_uniform_weights(self):
        """
        Initialize weights uniformly for all groups.
        """
        num_groups = len(self.groups)
        return {g: 1 / num_groups for g in self.groups}
    
    def accumulate_probabilities(self, prob, batch):
        """
        Accumulate probabilities for each group.
        Args:
            prob: Tensor of probabilities for the current batch.
            batch: Batch dictionary containing 'labels' and 'group_labels'.
        """
        group_0 = (batch[self.sensitive_attribute] == 0)
        group_1 = (batch[self.sensitive_attribute] == 1)

        group_0_y1 = (batch["labels"][group_0] == 1)
        group_1_y1 = (batch["labels"][group_1] == 1)

        self.accumulated_probs[0].extend(prob[group_0][group_0_y1].tolist())
        self.accumulated_probs[1].extend(prob[group_1][group_1_y1].tolist())

    def calculate_group_eod(self):
        """
        Calculate EO deviations and update weights based on accumulated probabilities.
        """
        # Calculate EO deviations
        p_y1_s0 = torch.tensor(self.accumulated_probs[0]).mean()
        p_y1_s1 = torch.tensor(self.accumulated_probs[1]).mean()
        p_y1 = (torch.tensor(self.accumulated_probs[0] + self.accumulated_probs[1]).mean())

        self.accumulated_probs = {0: [], 1: []}

        return {
            0: p_y1_s0 - p_y1,
            1: p_y1_s1 - p_y1,
        }
    
    def update_weights(self, outputs, batch):
        """
        Accumulate probabilities and update weights when gradient_accumulation_steps is reached.
        Args:
            outputs: Model predictions (logits or probabilities).
            batch: Batch dictionary containing 'labels' and 'group_labels'.
        """
        prob = torch.softmax(outputs.logits, dim=1)[:, 1]  # P(ŷ=1)
        
        # Accumulate probabilities
        self.accumulate_probabilities(prob, batch)

        # Check if accumulation steps are reached
        if len(self.accumulated_probs[0]) >= self.gradient_accumulation_steps:
            eo_deviation = self.calculate_group_eod()

            # Compute EO-based weights
            total_deviation = sum(abs(dev) for dev in eo_deviation.values()) + 1e-6
            self.weights = {g: abs(eo_deviation[g]) / total_deviation for g in self.groups}
            print("\n", self.weights)

    def __iter__(self):
        """
        Generate indices dynamically based on updated weights.
        Returns indices, not precomputed batches.
        """
        group_indices = {g: (self.group_labels == g).nonzero(as_tuple=True)[0] for g in self.groups}
        for _ in range(self.num_samples // self.batch_size):
            indices = []
            for g in self.groups:
                group_count = int(self.weights[g] * 2 * self.batch_size)
                indices.extend(np.random.choice(group_indices[g].tolist(), group_count, replace=True).astype(int).tolist())

            np.random.shuffle(indices)
            yield from indices[:self.batch_size]

    def __len__(self):
        return self.num_samples // self.batch_size

# def calculate_group_eod(outputs, batch, sensitive_attribute):
#     """
#     Utility function to calculate probability deviations for equal opportunity(EOD) fairness by each group.
#     P(ŷ=1 | Y=1, s=g) - P(ŷ=1 | Y=1).

#     Args:
#         outputs: Model predictions (logits or probabilities).
#         batch: Batch dictionary containing 'labels' and 'group_labels'.
#         sensitive_attribute: Name of the sensitive attribute in the batch.

#     Returns:
#         Dictionary containing EO deviations for each group.
#     """
#     # P(ŷ=1)
#     prob = torch.softmax(outputs.logits, dim=1)[:, 1]  
        
#     # 민감한 속성 그룹 분리
#     group_0 = (batch[sensitive_attribute] == False)
#     group_1 = (batch[sensitive_attribute] == True)

#     # P(ŷ=1 | Y=1, S=0)
#     group_0_y1 = (batch["labels"][group_0] == 1)
#     p_y1_s0 = prob[group_0][group_0_y1].mean() if group_0_y1.sum() > 0 else torch.tensor(0.0).to(prob.device)

#     # P(ŷ=1 | Y=1, S=1)
#     group_1_y1 = (batch["labels"][group_1] == 1)
#     p_y1_s1 = prob[group_1][group_1_y1].mean() if group_1_y1.sum() > 0 else torch.tensor(0.0).to(prob.device)

#     # P(ŷ=1 | Y=1)
#     p_y1 = prob[batch["labels"] == 1].mean()

#     return {
#         0: p_y1_s0 - p_y1,
#         1: p_y1_s1 - p_y1,
#     }
