import torch
import einops
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        cos, sin = self._pre_compute()
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    
    def _pre_compute(self):
        channel_range = torch.arange(0, self.d_k, 2, device=self.device, dtype=torch.float32)
        base = 1 / (self.theta**(channel_range / self.d_k)) 
        t = torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32)
        angle = torch.outer(t, base)
        cos, sin = angle.cos(), angle.sin()
        # cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        return cos, sin
    
    # to remember
    # 1. complex view
    # e^(i theta) = cos + i sin
    # (a + bi) (cos + i sin) = (a cos - b sin) + i (a sin + b cos)
    # cos -sin  a
    # sin cos.  b     
    # 2. base vector view
    # cos -sin  
    # sin cos.  
    # dot (1, 0) = (cos, sin)
    # dot (0, 1) = (-sin, cos)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        q_0, q_1 = x[..., ::2], x[..., 1::2]
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        # If x has shape (B, num_heads, seq, d_head), cos/sin need an extra dim for heads
        if x.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        # torch.cat([q_0 * cos - q_1 * sin, q_0 * sin + q_1 * cos], dim=-1) is also right, llama style implementation
        # but to pass the test case, we must get the same order back
        return torch.stack([q_0 * cos - q_1 * sin, q_0 * sin + q_1 * cos], dim=-1).flatten(-2)