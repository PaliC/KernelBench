import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    stride_batch,
    stride_dim1,
    stride_dim2,
    dim1,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    batch = row_idx // dim2
    d2 = row_idx % dim2
    base = batch * stride_batch + d2 * stride_dim2
    
    min_val = tl.load(x_ptr + base + 0 * stride_dim1)
    min_idx = 0
    for i in tl.range(1, dim1):
        val = tl.load(x_ptr + base + i * stride_dim1)
        if val < min_val:
            min_val = val
            min_idx = i
    tl.store(out_ptr + row_idx, min_idx)


def triton_argmin(x: torch.Tensor, dim: int):
    if dim != 1 or x.dim() != 3:
        return torch.argmin(x, dim=dim)
    
    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape
    n_rows = batch_size * dim2
    out = torch.empty(n_rows, device=x.device, dtype=torch.long)
    
    BLOCK_SIZE = 256
    grid = (n_rows,)
    argmin_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2),
        dim1, n_rows,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(batch_size, dim2)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)