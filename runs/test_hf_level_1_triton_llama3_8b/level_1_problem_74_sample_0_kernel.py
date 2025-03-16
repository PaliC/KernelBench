import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Pointer to input
    weight_ptr,  # Pointer to weight
    bias_ptr,  # Pointer to bias
    out_ptr,  # Pointer to output
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    BLOCK_SIZE: tl.constexpr,
    grid_size: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < grid_size
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels, mask=mask, other=0.0)
    # Load weight values
    weight = tl.load(weight_ptr + offsets * out_channels, mask=mask, other=0.0)
    # Perform the convolution
    out = tl.dot(x, weight)
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out += bias
    # Store the result
    tl.store(out_ptr + offsets * out_channels, out, mask=mask)


def triton_conv1d_transpose(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
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
    in_channels = x.shape[1]
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    stride = weight.shape[3]
    padding = weight.shape[4]
    dilation = weight.shape[5]
    BLOCK_SIZE = 128  # Tunable parameter for block size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    conv1d_transpose_kernel[grid_size, BLOCK_SIZE](x, weight, bias, out, in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE, grid_size=grid_size)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, stride, padding, dilation))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d_transpose(x, self.weight, self.bias)