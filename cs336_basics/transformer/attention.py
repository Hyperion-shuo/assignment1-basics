import torch
import torch.nn as nn
import torch.nn.functional as f
import einops
import math

from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.transformer.core import softmax, Linear, SwiGLU, RMSNorm
from cs336_basics.transformer.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor|None=None):
    attention_product = einops.einsum(Q, K, 'b ... n d_k, b ... m d_k -> b ... n m')
    attention_score = attention_product / math.sqrt(Q.shape[-1])
    min_value = torch.finfo(attention_score.dtype).min
    if mask is not None:
        attention_score = attention_score.masked_fill(mask, min_value)
    attention_score = softmax(attention_score, dim=-1)
    return einops.einsum(attention_score, V, 'b ... n m, b ... m d_v -> b ... n d_v')

class MultiheadSelfAttention(nn.Module):
    # TO use Muon optimizer, we seperate Q\K\V weights
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.n_head = num_heads
        self.d_head = int(d_model / num_heads)
        
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
    
    def forward(
        self, 
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: torch.Tensor | None = None, 
        rope: RotaryPositionalEmbedding | None = None,
        ) -> Float[Tensor, "... seq d_model"]:
        
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        Q = einops.rearrange(Q, '... seq (h d) -> ... h seq d', h=self.n_head, d=self.d_head)
        K = einops.rearrange(K, '... seq (h d) -> ... h seq d', h=self.n_head, d=self.d_head)
        V = einops.rearrange(V, '... seq (h d) -> ... h seq d', h=self.n_head, d=self.d_head)
        causal_mask = torch.ones((Q.shape[-2], K.shape[-2]), dtype=torch.bool, device=Q.device)
        # https://docs.pytorch.org/docs/stable/generated/torch.triu.html
        # torch.triu(a, diagonal=1)
        # tensor([[ 0.0000,  0.5207,  2.0049],
        #         [ 0.0000,  0.0000,  0.6602],
        #         [ 0.0000,  0.0000,  0.0000]])
        causal_mask = torch.triu(causal_mask, diagonal=1) 
        # causal_mask = torch.tril(causal_mask)   
               
        if token_positions is not None and rope is not None:
            Q, K = rope(Q, token_positions), rope(K, token_positions)
        output = scaled_dot_product_attention(Q, K, V, causal_mask)
        output = einops.rearrange(output, '... h seq d -> ... seq (h d)', h=self.n_head, d=self.d_head)
        return self.W_O(output)