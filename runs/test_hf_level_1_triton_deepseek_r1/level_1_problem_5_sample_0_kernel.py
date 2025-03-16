import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mul_kernel(
    input_ptr,
    output_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Perform scalar multiplication
    output_data = input_data * scalar
    
    # Store result
    tl.store(output_ptr + offsets, output_data, mask=mask)


def triton_scalar_mul(A: torch.Tensor, s: float):
    assert A.is_cuda, "Input tensor must be on CUDA"
    A = A.contiguous()
    
    output = torch.empty_like(A)
    n_elements = A.numel()
    
    # Tuned for modern GPUs - can be adjusted based on specific hardware
    BLOCK_SIZE = 1024  
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    scalar_mul_kernel[grid](A, output, s, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_scalar_mul(A, s)