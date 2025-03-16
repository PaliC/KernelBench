import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    kernel_size,  # Size of the kernel
    stride,  # Stride of the pooling operation
    padding,  # Padding to apply before pooling
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    GLOBAL_ID: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Calculate the start and end indices for the current block
    block_start = BLOCK_ID * BLOCK_SIZE
    block_end = (BLOCK_ID + 1) * BLOCK_SIZE
    # Calculate the start and end indices for the current global thread
    thread_start = GLOBAL_ID
    thread_end = GLOBAL_ID + 1
    # Calculate the number of elements in the current block
    block_size = block_end - block_start
    # Calculate the number of elements in the current thread
    thread_size = thread_end - thread_start
    # Create a mask to ensure we don't go out of bounds
    mask = tl.program_id(0) < NUM_BLOCKS
    # Load input values
    x = tl.load(x_ptr + block_start + thread_start * kernel_size * kernel_size, mask=mask, other=0.0)
    # Calculate the output value
    out = (x + tl.load(x_ptr + block_start + (thread_start + 1) * kernel_size * kernel_size, mask=mask, other=0.0) +
           tl.load(x_ptr + block_start + (thread_start + 1) * kernel_size * (kernel_size + 1), mask=mask, other=0.0) +
           tl.load(x_ptr + block_start + (thread_start + 1) * (kernel_size + 1) * (kernel_size + 1), mask=mask, other=0.0) +
           tl.load(x_ptr + block_start + thread_start * kernel_size * (kernel_size + 1), mask=mask, other=0.0) +
           tl.load(x_ptr + block_start + thread_start * (kernel_size + 1) * (kernel_size + 1), mask=mask, other=0.0)) / (kernel_size * kernel_size)
    # Store the result
    tl.store(out_ptr + block_start + thread_start * kernel_size * kernel_size, out, mask=mask)


def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
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
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    avg_pool_kernel[grid](x, out, kernel_size, stride or kernel_size, padding, BLOCK_SIZE, tl.program_id(0), tl.program_id(0), (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = triton_avg_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return self.avg_pool(x, kernel_size=self.avg_pool.kernel_size, stride=self.avg_pool.stride, padding=self.avg_pool.padding)