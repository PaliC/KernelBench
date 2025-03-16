import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to first input
    B_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_rows,  # Number of rows in A
    n_cols,  # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    ROW_ID: tl.constexpr,
    COL_ID: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = BLOCK_ID * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_cols
    # Load input values
    A = tl.load(A_ptr + ROW_ID * n_cols + offsets, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets * n_rows + COL_ID, mask=mask, other=0.0)
    # Perform the elementwise multiplication and sum
    out = tl.sum(A * B, axis=0)
    # Store the result
    tl.store(out_ptr + ROW_ID * n_cols + COL_ID, out, mask=mask)


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
    out = torch.empty_like(A)

    # Number of rows and columns in the tensor
    n_rows = A.shape[0]
    n_cols = B.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_rows + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (n_cols + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch the Triton kernel
    matmul_kernel[grid, BLOCK_SIZE, BLOCK_SIZE](A, B, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)