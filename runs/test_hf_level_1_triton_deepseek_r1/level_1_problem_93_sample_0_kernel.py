import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    rows,
    cols,
    x_row_stride,
    x_col_stride,
    mask_row_stride,
    mask_col_stride,
    out_row_stride,
    out_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= rows:
        return
    
    cum_sum = 0.0
    for col_offset in range(0, cols):
        x_offset = row_idx * x_row_stride + col_offset * x_col_stride
        mask_offset = row_idx * mask_row_stride + col_offset * mask_col_stride
        out_offset = row_idx * out_row_stride + col_offset * out_col_stride
        
        x_val = tl.load(x_ptr + x_offset)
        mask_val = tl.load(mask_ptr + mask_offset)
        mask_active = mask_val != 0
        
        contribution = x_val * tl.where(mask_active, 1.0, 0.0)
        cum_sum += contribution
        tl.store(out_ptr + out_offset, cum_sum)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    assert x.shape == mask.shape, "Shapes must match."
    
    # Permute target dim to last position for kernel simplicity
    dim_orig = dim
    perm = list(range(x.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(perm).contiguous()
    mask_perm = mask.permute(perm).contiguous()
    
    # Prepare output
    out_perm = torch.empty_like(x_perm)
    
    # Reshape to 2D view (flattened_dims, target_dim)
    rows = x_perm.numel() // x_perm.shape[-1]
    cols = x_perm.shape[-1]
    
    # Launch kernel
    grid = (rows,)
    masked_cumsum_kernel[grid](
        x_perm, mask_perm, out_perm,
        rows, cols,
        x_perm.stride(0), x_perm.stride(1),
        mask_perm.stride(0), mask_perm.stride(1),
        out_perm.stride(0), out_perm.stride(1),
        BLOCK_SIZE=1
    )
    
    # Permute back to original dimension order
    return out_perm.permute(perm)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        return triton_masked_cumsum(x, mask, self.dim)