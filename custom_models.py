import torch.nn as nn


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
        hidden_states = outputs.hidden_states[-1]
        
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

import torch

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput


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
