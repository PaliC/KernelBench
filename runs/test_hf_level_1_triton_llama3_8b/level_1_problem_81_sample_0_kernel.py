import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Pointer to input
    weight_ptr,  # Pointer to weight
    bias_ptr,  # Pointer to bias
    out_ptr,  # Pointer to output
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < batch_size
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels * kernel_size * kernel_size, mask=mask, other=0.0)
    # Reshape the input to (batch_size, in_channels, kernel_size, kernel_size)
    x = tl.reshape(x, (batch_size, in_channels, kernel_size, kernel_size))
    # Load weight values
    weight = tl.load(weight_ptr + offsets * out_channels * in_channels * kernel_size * kernel_size, mask=mask, other=0.0)
    # Reshape the weight to (out_channels, in_channels, kernel_size, kernel_size)
    weight = tl.reshape(weight, (out_channels, in_channels, kernel_size, kernel_size))
    # Perform the convolution
    out = tl.conv_transpose2d(x, weight, stride=stride, padding=padding, dilation=dilation)
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets * out_channels, mask=mask, other=0.0)
        out = out + bias
    # Store the result
    tl.store(out_ptr + offsets * out_channels * kernel_size * kernel_size, out, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    stride = 5
    padding = 1
    dilation = 2
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid](x, weight, bias, out, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(x, self.weight, self.bias)