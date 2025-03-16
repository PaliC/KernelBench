import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    dim_size,
    n_rows,
    x_row_stride,
    out_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    x_row = x_ptr + row_idx * x_row_stride
    out_row = out_ptr + row_idx * out_row_stride
    
    acc = 0.0
    for i in range(dim_size - 1, -1, -1):
        offset = i
        x_val = tl.load(x_row + offset)
        acc += x_val
        tl.store(out_row + offset, acc)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert x.is_cuda, "Input must be on CUDA"
        n_rows, dim_size = x.shape[0], x.shape[self.dim]
        x = x.contiguous()
        out = torch.empty_like(x)
        
        grid = (n_rows,)
        reverse_cumsum_kernel[grid](
            x, out,
            dim_size,
            n_rows,
            x.stride(0),
            out.stride(0),
            BLOCK_SIZE=1024,
        )
        return out