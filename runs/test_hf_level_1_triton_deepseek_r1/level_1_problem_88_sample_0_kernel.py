import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU computation
    sqrt_2_over_pi = 0.7978845608028654
    approx = 0.044715
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + approx * x_cubed)
    tanh = tl.tanh(inner)
    output = 0.5 * x * (1.0 + tanh)
    
    tl.store(out_ptr + offsets, output, mask=mask)


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Tuned for good occupancy across various sizes
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return triton_gelu(x)