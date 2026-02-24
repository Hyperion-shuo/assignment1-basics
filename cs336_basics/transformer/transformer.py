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
    def __init__(self, d_model: int, num_heads: int, d_ff: int, with_rms_norm: bool = True, pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        self.attn_norm = RMSNorm(d_model) if with_rms_norm else nn.Identity()
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.ffn_norm = RMSNorm(d_model) if with_rms_norm else nn.Identity()
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(
        self,                
        x: Float[Tensor, " ... seq d_in"],
        token_positions: torch.Tensor | None = None, 
        rope: RotaryPositionalEmbedding | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        
        if self.pre_norm:
            x = x + self.attn(self.attn_norm(x), token_positions, rope)
            x = x + self.ffn(self.ffn_norm(x))
        else:
            x = self.attn_norm(x + self.attn(x, token_positions, rope))
            x = self.ffn_norm(x + self.ffn(x))
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int,
                 context_length: int, num_layers: int, rope_theta: Float,
                 use_rope: bool = True, pre_norm: bool = True, with_rms_norm: bool = True):
        super().__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(rope_theta, int(d_model/num_heads), context_length) if use_rope else None
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, with_rms_norm=with_rms_norm, pre_norm=pre_norm) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model) if with_rms_norm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(
        self, 
        token_ids: Int[Tensor, " batch_size sequence_length"], 
        token_positions: Int[Tensor, " batch_size sequence_length"] | None = None,
        return_layer_norms: bool = False
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        
        if token_positions is None:
            batch_size, sequence_length = token_ids.shape
            token_positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(token_ids)
        layer_norms = [] if return_layer_norms else None
        for layer in self.layers:
            x = layer(x, token_positions, self.rope)
            if return_layer_norms:
                # compute mean L2 norm per token, then average; detach to avoid graph overhead
                layer_norms.append(x.detach().norm(dim=-1).mean().item())
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        if return_layer_norms:
            return logits, layer_norms
        return logits