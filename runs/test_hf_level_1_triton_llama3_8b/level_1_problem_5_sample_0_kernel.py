import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to input matrix A
    s,  # Scalar value
    C_ptr,  # Pointer to output matrix C
    M,  # Number of rows in A
    N,  # Number of columns in A
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < N
    # Load input values
    A = tl.load(A_ptr + offsets * N, mask=mask, other=0.0)
    # Perform the elementwise multiplication
    C = A * s
    # Store the result
    tl.store(C_ptr + offsets * N, C, mask=mask)


def triton_matmul(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda, "Tensor must be on CUDA."
    A = A.contiguous()

    # Prepare output tensor
    C = torch.empty_like(A)

    # Number of elements in the tensor
    M, N = A.shape
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](A, s, C, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        # Instead of "return A * s", call our Triton-based matrix-scalar multiplication
        return triton_matmul(A, s)