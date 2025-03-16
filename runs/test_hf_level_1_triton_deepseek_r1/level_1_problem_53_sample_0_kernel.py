import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    reduction_dim,
    dim_size,
    num_output_elements,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_output_elements
    
    if reduction_dim == 1:
        b = offsets // dim2
        other_dim_idx = offsets % dim2
    else:
        b = offsets // dim1
        other_dim_idx = offsets % dim1

    inf = -tl.math.inf(DTYPE)
    min_vals = tl.full((BLOCK_SIZE,), inf, dtype=DTYPE)

    for i in range(dim_size):
        if reduction_dim == 1:
            input_idx = b * dim1 * dim2 + i * dim2 + other_dim_idx
        else:
            input_idx = b * dim1 * dim2 + other_dim_idx * dim2 + i
        
        current_val = tl.load(input_ptr + input_idx, mask=mask, other=inf)
        min_vals = tl.minimum(min_vals, current_val)

    tl.store(output_ptr + offsets, min_vals, mask=mask)


def triton_min_reduction(x: torch.Tensor, reduction_dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim1, dim2 = x.shape
    if reduction_dim == 1:
        output_shape = (batch_size, dim2)
        dim_size = dim1
    else:
        output_shape = (batch_size, dim1)
        dim_size = dim2

    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    num_output_elements = output.numel()
    
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(num_output_elements, meta['BLOCK_SIZE']),)
    
    if x.dtype == torch.float32:
        dtype = tl.float32
    elif x.dtype == torch.float16:
        dtype = tl.float16
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    min_reduction_kernel[grid](
        x, output, batch_size, dim1, dim2, reduction_dim, 
        dim_size, num_output_elements, DTYPE=dtype, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min_reduction(x, self.dim)