import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= n_rows:
        return

    row_start = pid * row_stride
    tid = tl.arange(0, BLOCK_SIZE)
    col_offsets = tid[:, None] * ELEMENTS_PER_THREAD + tl.arange(0, ELEMENTS_PER_THREAD)[None, :]
    global_col = col_offsets
    mask = global_col < n_cols
    offsets = row_start + global_col
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    thread_max = tl.max(x, axis=1)
    max_val = tl.max(thread_max, axis=0)

    x_minus_max = x - max_val
    exp_x = tl.exp(x_minus_max)
    thread_sum = tl.sum(exp_x, axis=1)
    sum_exp = tl.sum(thread_sum, axis=0)
    log_sum_exp = tl.log(sum_exp)

    log_softmax_val = x_minus_max - log_sum_exp
    tl.store(out_ptr + offsets, log_softmax_val, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    assert x.is_cuda, "Input tensor must be on CUDA."
    if dim != 1:
        raise NotImplementedError("Only dim=1 is supported.")
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, n_cols = x.shape
    row_stride = x.stride(0)
    BLOCK_SIZE = 1024
    ELEMENTS_PER_THREAD = 16
    grid = (batch_size,)
    log_softmax_kernel[grid](
        x, out, row_stride, batch_size, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x, dim=self.dim)