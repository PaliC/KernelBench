import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    row_stride_x,
    row_stride_out,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * row_stride_x
    out_row_ptr = out_ptr + row_idx * row_stride_out

    sum_acc = tl.zeros((1,), dtype=tl.float32)
    for elem_offset in range(0, dim, BLOCK_SIZE):
        offsets = elem_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        sum_acc += tl.sum(tl.abs(x), 0

    total_sum = tl.sum(sum_acc, 0)
    tid = tl.arange(0, BLOCK_SIZE)
    shm = tl.static_local_array(BLOCK_SIZE, tl.float32)
    shm[tid] = sum_acc
    tl.barrier()

    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shm[tid] += shm[tid + stride]
        stride //= 2
        tl.barrier()

    total_sum = shm[0]

    for elem_offset in range(0, dim, BLOCK_SIZE):
        offsets = elem_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        out = x / total_sum
        tl.store(out_row_ptr + offsets, out, mask=mask)


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    batch_size, dim = x.shape
    BLOCK_SIZE = 512

    grid = (batch_size,)
    l1_norm_kernel[grid](
        x, out,
        x.stride(0), out.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)