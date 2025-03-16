import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    sigmoid_x = tl.sigmoid(x)
    output = x * sigmoid_x
    tl.store(out_ptr + offsets, output, mask=mask)

def triton_swish(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Heuristic for block size based on tensor size
    max_block_size = 1024
    BLOCK_SIZE = min(max_block_size, triton.next_power_of_2(n_elements // 128))
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    swish_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)