import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    inner_size,
    non_reduction_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < non_reduction_elements

    base_outer = (offsets // inner_size) * dim_size * inner_size
    base_inner = offsets % inner_size
    base_offset = base_outer + base_inner

    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(dim_size):
        input_offsets = base_offset + i * inner_size
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        sum += x
    mean = sum / dim_size
    tl.store(output_ptr + offsets, mean, mask=mask)


def triton_mean(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    dim_size = x.size(dim)
    inner_size = x.stride(dim)
    outer_size = x.numel() // (dim_size * inner_size)
    non_reduction_elements = outer_size * inner_size

    output = torch.empty((outer_size, inner_size), dtype=x.dtype, device=x.device)
    x_flat = x.view(-1)
    output_flat = output.view(-1)

    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(non_reduction_elements, meta['BLOCK_SIZE']),)
    
    mean_kernel[grid](x_flat, output_flat, dim_size, inner_size, non_reduction_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    new_shape = x.shape[:dim] + x.shape[dim+1:]
    return output_flat.view(new_shape)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean(x, self.dim)