import torch

from .base import BaseModel

class RegularizationModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.regularization_coeff = self.hparams.method.regularization_weight
        self.balance_coeff = 1.0 / self.datamodule.rho * self.hparams.method.balance_weight

    def _get_loss_and_metrics(self, outputs, batch, batch_idx):
        ce_loss, _ = super()._get_loss_and_metrics(outputs, batch, batch_idx)
        regular_loss, regular_metrics = self.equal_opportunity_loss_and_metric(outputs, batch, batch_idx)
        loss = ce_loss + regular_loss
        metrics = {
            "train/loss": loss,
            "train/ce_loss": ce_loss,
            **regular_metrics
        }
        return loss, metrics

    def equal_opportunity_loss_and_metric(self, outputs, batch, batch_idx):
        prob = torch.softmax(outputs.logits, dim=1)[:, 1]  # P(ŷ=1)
        
        # 민감한 속성 그룹 분리
        sensitive_attribute = self.datamodule.sensitive_attribute
        group_0 = (batch[sensitive_attribute] == 0)
        group_1 = (batch[sensitive_attribute] == 1)

        # P(ŷ=1 | Y=1, A=0)
        group_0_y1 = (batch["labels"][group_0] == 1)
        p_y1_a0 = prob[group_0][group_0_y1].mean() if group_0_y1.sum() > 0 else torch.tensor(0.0).to(prob.device)

        # P(ŷ=1 | Y=1, A=1)
        group_1_y1 = (batch["labels"][group_1] == 1)
        p_y1_a1 = prob[group_1][group_1_y1].mean() if group_1_y1.sum() > 0 else torch.tensor(0.0).to(prob.device)

        p_y1 = prob[batch["labels"] == 1].mean()

        # Equal Opportunity Difference Penalty
        eo_penalty = self.regularization_coeff * torch.abs(p_y1_a0 - p_y1)
        eo_penalty += self.regularization_coeff * torch.abs(p_y1_a1 - p_y1)

        # Class balance penalty
        rho = self.datamodule.rho
        overall_p_y1 = prob.mean()
        balance_penalty = self.balance_coeff * torch.clamp(rho - overall_p_y1, min=0)

        return eo_penalty + balance_penalty, {"train/eod_loss": eo_penalty, "train/balance_loss": balance_penalty}

class AdversarialRepresentationModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def configure_model(self):
        super().configure_model()
        self.model = DualSequenceClassifierModel(self.model, self.datamodule.num_sensitive_classes)
    
    def forward(self, input_ids, attention_mask=None, labels=None, sensitive_labels=None, **inputs):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels, sensitive_labels=sensitive_labels)
    
    def _get_loss_and_metrics(self, outputs, batch, batch_idx):
        loss = outputs.loss - outputs.second_loss
        metrics = {"train/loss": loss, "train/y_loss": outputs.loss, "train/a_loss": outputs.second_loss}
        return loss, metrics
    
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput

class DualSequenceClassifierModel(nn.Module):
    """Adversarial Representation Model"""
    def __init__(self, model, num_sensitive_classes):
        super().__init__()
        self.base_model = model
        
        # Define custom heads
        hidden_size = self.base_model.config.hidden_size
                
        # Adversarial predictor for A
        self.adversary = nn.Linear(hidden_size, num_sensitive_classes)

    def forward(self, input_ids, attention_mask, labels, sensitive_labels, output_attentions=False, **kwargs):
        # Forward pass through base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True, output_attentions=output_attentions)

        # Get hidden states (last layer)
        hidden_states = outputs.hidden_states[-2]
        print(outputs.hidden_states[-1].shape)
        print(hidden_states.shape)

        # Average pooling to reduce sequence to a single vector
        pooled_output = hidden_states.mean(dim=1)
        
        # Predictions
        second_logits = self.adversary(pooled_output)  # For adversarial task
        second_loss = nn.CrossEntropyLoss()(second_logits.view(-1, 2), sensitive_labels.view(-1))

        return DualSequenceClassifierOutputWithPast(
            loss=outputs.loss,
            second_loss=second_loss,
            logits=outputs.logits,
            second_logits=second_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if output_attentions else None
        )

@dataclass
class DualSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    second_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    second_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
