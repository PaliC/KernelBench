import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    kernel_size,  # Size of the pooling window
    stride,  # Stride of the pooling operation
    padding,  # Padding applied to the input tensor
    height,  # Height of the input tensor
    width,  # Width of the input tensor
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    THREAD_ID: tl.constexpr,
):
    # Calculate the start and end indices for the current block
    block_start_y = BLOCK_ID * (height // BLOCK_SIZE)
    block_end_y = min(block_start_y + (height // BLOCK_SIZE), height)
    block_start_x = THREAD_ID * (width // BLOCK_SIZE)
    block_end_x = min(block_start_x + (width // BLOCK_SIZE), width)

    # Calculate the start and end indices for the current thread
    thread_start_y = block_start_y + (THREAD_ID // (width // BLOCK_SIZE)) * (height // BLOCK_SIZE)
    thread_end_y = min(thread_start_y + (height // BLOCK_SIZE), block_end_y)
    thread_start_x = block_start_x + (THREAD_ID % (width // BLOCK_SIZE)) * (width // BLOCK_SIZE)
    thread_end_x = min(thread_start_x + (width // BLOCK_SIZE), block_end_x)

    # Create a range of offsets for the current thread
    offsets_y = thread_start_y + tl.arange(0, thread_end_y - thread_start_y + 1)
    offsets_x = thread_start_x + tl.arange(0, thread_end_x - thread_start_x + 1)

    # Mask to ensure we don't go out of bounds
    mask_y = offsets_y < height
    mask_x = offsets_x < width

    # Load input values
    x = tl.load(x_ptr + (offsets_y * width + offsets_x), mask=mask_y & mask_x, other=0.0)

    # Calculate the output value
    out = tl.sum(x, axis=0) / ((kernel_size * kernel_size) - (kernel_size - 1) * (kernel_size - 1))

    # Store the result
    tl.store(out_ptr + (offsets_y * width + offsets_x), out, mask=mask_y & mask_x)


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
    height, width = x.shape[2:]
    n_elements = height * width
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((height // (meta["BLOCK_SIZE"] // (width // BLOCK_SIZE)) + 1,), (width // (meta["BLOCK_SIZE"] // (width // BLOCK_SIZE)) + 1,),)

    # Launch the Triton kernel
    avg_pool_kernel[grid](x, out, kernel_size, stride or kernel_size, padding, height, width, BLOCK_SIZE=BLOCK_SIZE, BLOCK_ID=tl.program_id(0), THREAD_ID=tl.thread_id())
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = triton_avg_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)