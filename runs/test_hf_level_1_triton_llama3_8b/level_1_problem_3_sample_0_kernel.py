import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def batch_matmul_kernel(
    A_ptr,  # Pointer to first input
    B_ptr,  # Pointer to second input
    C_ptr,  # Pointer to output
    batch_size,  # Batch size
    m,  # Number of rows in A
    k,  # Number of columns in A and rows in B
    n,  # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = BLOCK_ID * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < m
    # Load input values
    A = tl.load(A_ptr + offsets * k + tl.arange(0, k) * batch_size, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets * n + tl.arange(0, n) * batch_size, mask=mask, other=0.0)
    # Perform the elementwise multiplication and sum
    C = tl.sum(A * B, axis=1)
    # Store the result
    tl.store(C_ptr + offsets * n + BLOCK_ID * BLOCK_SIZE * n, C, mask=mask)


def triton_batch_matmul(A: torch.Tensor, B: torch.Tensor):
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
    C = torch.empty_like(A)

    # Number of elements in the tensor
    batch_size = A.shape[0]
    m = A.shape[1]
    k = A.shape[2]
    n = B.shape[2]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch the Triton kernel
    batch_matmul_kernel[grid](A, B, C, batch_size, m, k, n, BLOCK_SIZE=BLOCK_SIZE, BLOCK_ID=tl.program_id(0))
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Instead of "return torch.bmm(A, B)", call our Triton-based addition
        return triton_batch_matmul(A, B)