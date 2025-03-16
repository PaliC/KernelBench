import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_pool_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    kernel_size,  # Size of the pooling window
    stride,  # Stride of the pooling window
    padding,  # Padding to be applied before pooling
    dilation,  # Spacing between kernel elements
    height,  # Height of the input tensor
    width,  # Width of the input tensor
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    THREAD_ID: tl.constexpr,
):
    # Calculate the start and end indices for the current block
    block_start_x = (BLOCK_ID * stride) - padding
    block_end_x = block_start_x + (width // stride)
    block_start_y = (BLOCK_ID * stride) - padding
    block_end_y = block_start_y + (height // stride)

    # Calculate the start and end indices for the current thread
    thread_start_x = (THREAD_ID * stride) - padding
    thread_end_x = thread_start_x + (width // stride)
    thread_start_y = (THREAD_ID * stride) - padding
    thread_end_y = thread_start_y + (height // stride)

    # Mask to ensure we don't go out of bounds
    mask_x = tl.all(thread_start_x <= block_end_x)
    mask_y = tl.all(thread_start_y <= block_end_y)

    # Load input values
    x = tl.load(x_ptr + (thread_start_y * width + thread_start_x) * 4, mask=mask_x & mask_y, other=-float('inf'))

    # Perform the max pooling
    out = tl.max(x)

    # Store the result
    tl.store(out_ptr + (BLOCK_ID * (height // stride) * (width // stride) + (thread_start_y * (width // stride) + thread_start_x)) * 4, out, mask=mask_x & mask_y)


def triton_max_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
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
    grid = lambda meta: ((height // stride + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (width // stride + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    max_pool_kernel[grid](x, out, kernel_size, stride, padding, dilation, height, width, BLOCK_SIZE=BLOCK_SIZE, BLOCK_ID=tl.program_id(0), THREAD_ID=tl.thread_id(0))
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.maxpool = triton_max_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)


batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]