import torch
import torch.nn as nn
import torch.nn.functional as f
import einops
import math

from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.transformer.attention import MultiheadSelfAttention
from cs336_basics.transformer.core import SwiGLU, RMSNorm, Embedding, Linear, softmax
from cs336_basics.transformer.rope import RotaryPositionalEmbedding

class TransformerBlock(nn.Module):
    # add QK norm afterwards
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(
        self,                
        x: Float[Tensor, " ... seq d_in"],
        token_positions: torch.Tensor | None = None, 
        rope: RotaryPositionalEmbedding | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        
        x = x + self.attn(self.attn_norm(x), token_positions, rope)
        x = x + self.ffn(self.ffn_norm(x))
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int,
                 context_length: int, num_layers: int, rope_theta: Float):
        super().__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(rope_theta, int(d_model/num_heads), context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(
        self, 
        token_ids: Int[Tensor, " batch_size sequence_length"], 
        token_positions: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length"]:
        
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions, self.rope)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        # probs = softmax(logits)
        
        return logits