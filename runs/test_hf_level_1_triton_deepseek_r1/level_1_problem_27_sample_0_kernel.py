import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    alpha: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SELU computation: scale * (x if x > 0 else alpha * (exp(x) - 1))
    zero = tl.full(x.shape, 0.0, tl.float32)
    safe_x = tl.where(x > 0, x, zero)
    neg_x = tl.where(x <= 0, x, zero)
    result = scale * (safe_x + alpha * (tl.exp(neg_x) - 1))
    
    tl.store(out_ptr + offsets, result, mask=mask)

def triton_selu(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Hyperparameters from original SELU paper
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    BLOCK_SIZE = 1024  # Optimized for modern GPU architectures
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    selu_kernel[grid](x, out, n_elements, alpha, scale, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_selu(x)