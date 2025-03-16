import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def frobenius_norm_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    norm,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    normalized = x / norm
    tl.store(output_ptr + offsets, normalized, mask=mask)

@triton.jit
def sum_squares_kernel(
    x_ptr,
    output_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_squared = x * x
    sum_x = tl.sum(x_squared)
    tl.atomic_add(output_sum_ptr, sum_x)

def compute_frobenius_norm(x: torch.Tensor):
    x_contig = x.contiguous().view(-1)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return torch.zeros_like(x_contig)
    
    # Compute sum of squares
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    sum_sq = torch.zeros(1, device=x.device, dtype=x.dtype)
    sum_squares_kernel[grid](x_contig, sum_sq, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute norm with epsilon for numerical stability
    norm = torch.sqrt(sum_sq) + 1e-8
    
    # Normalize using Triton kernel
    output = torch.empty_like(x_contig)
    frobenius_norm_kernel[grid](x_contig, output, n_elements, norm, BLOCK_SIZE=BLOCK_SIZE)
    return output.view_as(x)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return compute_frobenius_norm(x)