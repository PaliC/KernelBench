import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    n_rows,
    row_size,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    row_offset = row_idx * stride_row
    x_row_ptr = x_ptr + row_offset
    out_row_ptr = out_ptr + row_offset

    pid = tl.program_id(1)
    offsets = pid * ELEMENTS_PER_THREAD + tl.arange(0, ELEMENTS_PER_THREAD)
    mask = offsets < row_size
    
    x = tl.load(x_row_ptr + offsets, mask=mask, other=1.0)
    
    cumprod = tl.zeros((ELEMENTS_PER_THREAD,), dtype=tl.float32)
    current = 1.0
    for i in range(ELEMENTS_PER_THREAD):
        current *= tl.where(mask[i], x[i], 1.0)
        cumprod = tl.where(mask[i], cumprod, cumprod.at[i].set(current))
    
    chunk_product = tl.where(mask[-1], cumprod[-1], 1.0)
    
    shmem = tl.static_array(shape=(BLOCK_SIZE,), dtype=tl.float32)
    shmem[pid] = chunk_product
    tl.debug_barrier()

    stride = 1
    while stride < BLOCK_SIZE:
        if pid >= stride:
            shmem[pid] *= shmem[pid - stride]
        stride *= 2
        tl.debug_barrier()

    prefix = tl.where(pid == 0, 1.0, shmem[pid - 1])
    adjusted = cumprod * prefix
    
    tl.store(out_row_ptr + offsets, adjusted, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int):
    assert x.is_cuda and dim == 1, "Requires CUDA tensor and dim=1"
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, row_size = x.shape[0], x.shape[1]
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 16
    
    grid = (n_rows, BLOCK_SIZE)
    cumprod_kernel[grid](
        x, out, x.stride(0), n_rows, row_size,
        BLOCK_SIZE=BLOCK_SIZE, ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)