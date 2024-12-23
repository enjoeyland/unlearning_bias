import torch
import torch.nn.functional as F
from typing import Tuple

def dpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        beta: float,
        label_smoothing: float = 0.0,
        reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    
    logits, chosen_rewards, rejected_rewards = preference_logit(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta, reference_free)

    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    return losses, chosen_rewards, rejected_rewards

def preference_logit(policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        beta: float,
        reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return logits, chosen_rewards, rejected_rewards
    
def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """
    Compute log probabilities for the given labels under the given logits.

    Args:
        logits: Model logits. Shape: (batch_size, sequence_length, vocab_size).
        labels: Target labels. Tokens with -100 are ignored. Shape: (batch_size, sequence_length).
        average_log_prob: If True, return average log probability per token; otherwise, return the sum.

    Returns:
        Tensor of shape (batch_size,) with log probabilities.
    """
    # Align logits and labels for causal modeling
    logits, labels = logits[:, :-1], labels[:, 1:]
    loss_mask = labels != -100
    labels = labels.masked_fill(~loss_mask, 0)  # Replace -100 with dummy value 0

    # Compute log probabilities
    per_token_logps = torch.gather(logits.log_softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_probs = (per_token_logps * loss_mask).sum(dim=-1)

    # Return average or sum of log probabilities
    return log_probs / loss_mask.sum(dim=-1) if average_log_prob else log_probs