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
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    GLOBAL_ID: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = BLOCK_ID * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < BLOCK_SIZE
    # Load input values
    A = tl.load(A_ptr + GLOBAL_ID * K * BLOCK_SIZE + offsets * K, mask=mask, other=0.0)
    B = tl.load(B_ptr + (GLOBAL_ID * BLOCK_SIZE + offsets) * N, mask=mask, other=0.0)
    # Perform the elementwise multiplication and accumulation
    out = tl.dot(A, B)
    # Store the result
    tl.store(out_ptr + GLOBAL_ID * N + offsets, out, mask=mask)


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
    out = torch.empty(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)

    # Number of elements in the tensor
    M, K, N = A.shape[0], A.shape[1], B.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, out, M, K, N, BLOCK_SIZE, tl.program_id(0), tl.program_id(1))
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Instead of "return torch.matmul(A, B)", call our Triton-based matrix multiplication
        return triton_matmul(A, B)