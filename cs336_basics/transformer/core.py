import torch
import einops
import torch.nn as nn
import torch.nn.functional as f
from typing import Union
from torch import Tensor

DeviceType = Union[torch.device, str]

# 1600 * 6400 = 10240000
class Linear(nn.Module):
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                device: DeviceType | None = None, 
                dtype:torch.dtype | None = None):
        # call the super class constructor
        super().__init__()
        weight = torch.empty((out_features, in_features), dtype=dtype, device=device)
        std = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y

# 50257 * 1600 * 2 = 160822400
class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int,
                 device: DeviceType | None = None, 
                 dtype: torch.dtype | None = None):
        # call the super class constructor
        super().__init__()
        weight = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

# 1600
class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: DeviceType | None = None, 
                 dtype: torch.dtype | None = None):
        # call the super class constructor
        super().__init__()
        weight = torch.ones(d_model, dtype=dtype, device=device)
        self.weight = nn.Parameter(weight)
        # use python float, do not register_buffer, aviod model.half auto cast to bf16 or fp16
        self.eps = eps
        # register_buffer will auto move with model.to and auto save\load state dict with model
        # self.register_buffer('eps', torch.tensor(eps, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm do not divid d_model, can not use
        # https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
        # https://docs.pytorch.org/docs/main/generated/torch.linalg.norm.html#torch.linalg.norm
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        # must cast to fp32 for rms 
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps)
        # cast back after rms and element-wise multiplication
        return  (x_fp32 * rms).to(input_dtype) * self.weight.to(input_dtype)

# TODO: maybe must write raw matrix for zero implementation
# 
class SwiGLU(nn.Module):
    def __init__(self,
                d_model: int, 
                d_ff: int | None, 
                device: DeviceType | None = None, 
                dtype:torch.dtype | None = None):
        super().__init__()
        if not d_ff:
            d_ff = (int((8/3) * d_model + 63) // 64) * 64
        else:
            d_ff = (d_ff + 63) // 64 * 64
        self.linear_1 = Linear(d_model, d_ff, device, dtype)
        self.linear_3 = Linear(d_model, d_ff, device, dtype)
        self.linear_2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_1_out= self.linear_1(x)
        swish = torch.sigmoid(linear_1_out) * linear_1_out
        gate = self.linear_3(x)
        GLU = swish * gate
        y = self.linear_2(GLU)
        return y
    
def softmax(x: Tensor, dim: int = -1):
    x -= torch.max(x, dim=dim, keepdim=True)[0]
    return f.softmax(x, dim)