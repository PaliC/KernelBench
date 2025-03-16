import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to input matrix A
    B_ptr,  # Pointer to input vector B
    out_ptr,  # Pointer to output vector
    M: tl.constexpr,  # Number of rows in matrix A
    K: tl.constexpr,  # Number of columns in matrix A and rows in vector B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < M
    # Load input values
    A = tl.load(A_ptr + offsets * K, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets * K, mask=mask, other=0.0)
    # Perform the elementwise multiplication and sum
    out = tl.sum(A * B, axis=0)
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
    out = torch.empty_like(B)

    # Number of elements in the tensor
    M = A.shape[0]
    K = A.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, out, M, K, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Instead of "return torch.matmul(A, B)", call our Triton-based matrix-vector multiplication
        return triton_matmul(A, B)