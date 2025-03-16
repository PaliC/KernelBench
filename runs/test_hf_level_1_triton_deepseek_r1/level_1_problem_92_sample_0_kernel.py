import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    n_elements_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x
    out_row_ptr = out_ptr + row_idx * stride_out

    accumulator = 0.0
    for i in range(n_elements_dim):
        curr_val = tl.load(x_row_ptr + i)
        tl.store(out_row_ptr + i, accumulator)
        accumulator += curr_val


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Tensor must be on CUDA"
    original_shape = x.shape
    
    # Move target dim to last dimension and flatten others
    x_moved = x.movedim(dim, -1)
    moved_shape = x_moved.shape
    x_flat = x_moved.flatten(0, -2)
    x_flat = x_flat.contiguous()
    
    # Create output tensor
    out_flat = torch.empty_like(x_flat)
    num_rows, row_size = x_flat.shape
    
    # Launch kernel
    grid = (num_rows,)
    exclusive_cumsum_kernel[grid](
        x_flat, out_flat,
        x_flat.stride(0),
        out_flat.stride(0),
        row_size,
        BLOCK_SIZE=128,
    )
    
    # Restore original dimensions
    out_moved = out_flat.reshape(moved_shape)
    return out_moved.movedim(-1, dim)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)