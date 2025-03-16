import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    kernel_size,  # Size of the pooling window
    stride,  # Stride of the pooling operation
    padding,  # Padding applied to the input tensor
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = BLOCK_ID * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < x.shape[2]
    # Load input values
    x_block = tl.load(x_ptr + offsets * x.shape[1], mask=mask, other=0.0)
    # Perform the 1D Average Pooling
    out_block = tl.sum(x_block, axis=0) / (kernel_size + 2 * padding - stride + 1)
    # Store the result
    tl.store(out_ptr + offsets * x.shape[1], out_block, mask=mask)


def triton_avg_pool(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
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
    n_elements = x.shape[2]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    avg_pool_kernel[grid, BLOCK_SIZE, BLOCK_SIZE](x, out, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE, BLOCK_ID=tl.program_id(0), NUM_BLOCKS=grid(None))
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = triton_avg_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool(x, kernel_size=self.avg_pool.kernel_size, stride=self.avg_pool.stride, padding=self.avg_pool.padding)