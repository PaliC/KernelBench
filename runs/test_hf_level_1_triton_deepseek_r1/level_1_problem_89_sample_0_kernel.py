import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,
    out_ptr,
    row_length,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * row_length
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_start + row_length)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Perform inclusive scan using associative_scan
    prefix_sum = tl.associative_scan(x, 0, tl.math.add)
    tl.store(out_ptr + offsets, prefix_sum, mask=mask)


def triton_cumsum(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, row_length = x.shape[0], x.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(row_length)
    
    grid = (n_rows,)
    cumsum_kernel[grid](x, out, row_length, n_rows, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)