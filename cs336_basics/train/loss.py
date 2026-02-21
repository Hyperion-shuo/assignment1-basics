import torch
import einops
import torch.nn as nn
import torch.nn.functional as f
from typing import Union
from torch import Tensor

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    logits -= torch.max(logits, dim=-1, keepdim=True)[0]
    log_probs = f.log_softmax(logits, dim=-1)
    index = targets
    while index.ndim < log_probs.ndim:
        index = index.unsqueeze(-1)
    select_log_probs = torch.gather(log_probs, dim=-1, index=index)
    return -select_log_probs.mean()

def perplexity(logits: Tensor, targets: Tensor) -> Tensor:
    return torch.exp(cross_entropy_loss(logits, targets))