import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to first matrix
    B_ptr,  # Pointer to second matrix
    out_ptr,  # Pointer to output
    rows_A,  # Number of rows in first matrix
    cols_B,  # Number of columns in second matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < BLOCK_SIZE
    # Load input values
    A = tl.load(A_ptr + offsets * cols_B, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * cols_B, mask=mask, other=0.0)
    # Perform the matrix multiplication
    out = tl.sum(A * B, axis=1)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Prepare output tensor
    out = torch.empty((A.shape[0], B.shape[1]))

    # Number of elements in the tensor
    rows_A = A.shape[0]
    cols_B = B.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((rows_A + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, out, rows_A, cols_B, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(16384, 16384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instead of "return torch.nn.functional.softplus(self.linear(x))", call our Triton-based Softplus
        # and Matmul
        x = triton_matmul(x, self.linear.weight.t())
        x = triton_softplus(x)
        return x