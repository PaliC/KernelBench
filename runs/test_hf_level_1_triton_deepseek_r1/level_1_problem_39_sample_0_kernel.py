import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_start = x_ptr + row_idx * x_row_stride
    out_row_start = out_ptr + row_idx * out_row_stride

    sum_squares = 0.0
    num_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_idx in range(num_blocks):
        col_offset = block_idx * BLOCK_SIZE
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_start + cols, mask=mask, other=0.0)
        sum_squares += tl.sum(x * x, axis=0)

    eps = 1e-12
    scale = 1.0 / tl.sqrt(sum_squares + eps)

    for block_idx in range(num_blocks):
        col_offset = block_idx * BLOCK_SIZE
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_start + cols, mask=mask, other=0.0)
        scaled = x * scale
        tl.store(out_row_start + cols, scaled, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, n_cols = x.shape
    BLOCK_SIZE = 1024

    grid = (batch_size,)
    l2_norm_kernel[grid](
        x,
        out,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)