import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardtanh_kernel(
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
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_hardtanh(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Heuristic for block size based on tensor size
    max_block_size = 1024
    BLOCK_SIZE = min(max_block_size, triton.next_power_of_2(n_elements))
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hardtanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x)