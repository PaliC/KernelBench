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
    n_rows_A,  # Number of rows in A
    n_cols_A,  # Number of columns in A
    n_rows_B,  # Number of rows in B
    n_cols_B,  # Number of columns in B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_cols_A
    # Load input values
    A = tl.load(A_ptr + offsets * n_rows_A, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets * n_rows_B, mask=mask, other=0.0)
    # Perform the elementwise multiplication
    out = A * B
    # Store the result
    tl.store(out_ptr + offsets * n_rows_A, out, mask=mask)


@triton.jit
def tril_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_rows,  # Number of rows in the matrix
    n_cols,  # Number of columns in the matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_cols
    # Load input values
    x = tl.load(x_ptr + offsets * n_rows, mask=mask, other=0.0)
    # Perform the lower triangular operation
    out = tl.where(offsets >= tl.arange(0, n_cols), 0.0, x)
    # Store the result
    tl.store(out_ptr + offsets * n_rows, out, mask=mask)


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

    # Number of elements in the tensor
    n_rows_A = A.shape[0]
    n_cols_A = A.shape[1]
    n_rows_B = B.shape[0]
    n_cols_B = B.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_cols_A + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, out, n_rows_A, n_cols_A, n_rows_B, n_cols_B, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_tril(x: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_cols + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    tril_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Instead of "return torch.tril(torch.matmul(A, B))", call our Triton-based operations
        return triton_tril(triton_matmul(A, B))