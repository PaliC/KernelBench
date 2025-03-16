import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * row_stride

    # Compute max
    max_val = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
        curr_max = tl.max(x, 0)
        max_val = tl.maximum(max_val, curr_max)
    
    # Compute sum of exponentials
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x, 0)
    
    # Normalize and store
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < n_cols
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        softmax_out = tl.exp(x - max_val) / sum_exp
        tl.store(output_ptr + offsets, softmax_out, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    row_stride = x.stride(0)
    BLOCK_SIZE = 1024  # Tunable parameter

    grid = (n_rows,)
    softmax_kernel[grid](x, output, n_cols, row_stride, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)