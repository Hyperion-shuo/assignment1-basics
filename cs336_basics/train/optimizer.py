import torch
import math

from collections.abc import Callable, Iterable
from typing import Optional, Union, Tuple
from jaxtyping import Bool, Float, Int
from torch import linalg as LA

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: Float=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr: Float=1e-3, 
                 betas: Tuple[Float, Float]= (0.9, 0.999), # try (0.9, 0.95) for larger model
                 eps: Float=1e-8,
                 weight_decay: Float=0.01
        ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                if not state:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                t, m, v = state['t'], state['m'], state['v']

                if weight_decay != 0:
                    eff_weight_decay = lr * weight_decay
                    p.data.mul_(1 - eff_weight_decay)
                
                grad = p.grad.data # Get the gradient of loss with respect to p.
                # m = beta_1 * m + (1 - beta_1) * grad
                # v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value=1 - beta_2)
                alpha = lr * (math.sqrt(1 - beta_2 ** t)/(1 - beta_1 ** t))
                # p.data -= alpha * (m / (torch.sqrt(v) + eps)) # Update weight tensor in-place.
                denom = torch.sqrt(v).add_(eps)
                # update = m.div(denom).mul_(alpha)
                # p.data.add_(update, alpha=-1.0)
                p.data.addcdiv_(m, denom, value=-alpha)
                state["t"] = t + 1 # Increment iteration number.

def lr_cos_schedule(t: Int, alpha_min: Float, alpha_max: Float, T_w: Int, T_c: Int) -> Float:
    assert T_w < T_c, "T_w should smaller than T_c"

    if T_w > 0 and t <= T_w:
        return alpha_max * (t / T_w)
    
    if t < T_c:
        if T_c == T_w:
            return alpha_max
        progress = (t - T_w)/(T_c - T_w) 
        return alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + math.cos(progress * math.pi))

    return alpha_min
    

def gradient_clipping(
        params: Iterable[torch.nn.Parameter], 
        max_l2_norm: Float,
        eps: Float = 1e-6
    ) -> None:
    
    # one old implementation
    # total_norm = 0
    # for p in params:
    #     if p.grad is not None:
    #         total_norm += torch.linalg.vector_norm(p.grad, ord=2).item() ** 2

    # norm = total_norm**0.5

    # if norm > max_l2_norm:
    #     with torch.no_grad():
    #         scale = max_l2_norm /(norm + eps)
    #         for p in params:
    #             if p.grad is not None:
    #                 p.grad.mul_(scale)


    params_with_grad = [p for p in params if p.grad is not None]
    if not params_with_grad:
        return 
    
    grad_norms = torch.stack([LA.vector_norm(p.grad, ord=2) for p in params_with_grad])
    norm = LA.vector_norm(grad_norms)

    if norm > max_l2_norm:
        with torch.no_grad():
            scale = max_l2_norm / (norm + eps)
            for p in params_with_grad:
                p.grad.mul_(scale)