import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    reduce_size,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = input_ptr + row_idx * input_row_stride
    acc = tl.zeros((1,), dtype=tl.float32)

    for offset in range(0, reduce_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < reduce_size
        vec = tl.load(row_start + cols, mask=mask, other=0.0)
        acc += tl.sum(vec, axis=0)

    tl.store(output_ptr + row_idx * output_row_stride, acc)


def triton_sum_reduce(x: torch.Tensor):
    assert x.is_cuda and x.is_contiguous(), "Input must be CUDA contiguous tensor"
    n_rows, reduce_size = x.shape
    out = torch.empty((n_rows, 1), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = (n_rows,)
    
    sum_reduce_kernel[grid](
        x,
        out,
        n_rows,
        reduce_size,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        reduce_dim = self.dim if self.dim >= 0 else self.dim + x.ndim

        x_permuted = x.transpose(reduce_dim, -1).contiguous()
        permuted_shape = x_permuted.shape
        x_reshaped = x_permuted.view(-1, permuted_shape[-1])

        out_reshaped = triton_sum_reduce(x_reshaped)

        out_permuted = out_reshaped.view(permuted_shape[:-1] + (1,))
        return out_permuted.transpose(reduce_dim, -1).contiguous()