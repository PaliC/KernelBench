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
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the convolution kernel (height, width)
    stride,  # Stride of the convolution (height, width)
    padding,  # Padding applied to the input (height, width)
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    THREAD_ID: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = BLOCK_ID * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < x.shape[0]
    # Load input values
    x = tl.load(x_ptr + offsets * x.shape[1] * x.shape[2] * x.shape[3], mask=mask, other=0.0)
    # Load weight values
    weight = tl.load(weight_ptr + offsets * weight.shape[1] * weight.shape[2] * weight.shape[3], mask=mask, other=0.0)
    # Load bias values
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    # Perform the convolution
    out = F.conv_transpose2d(x, weight, bias, stride=stride, padding=padding)
    # Store the result
    tl.store(out_ptr + offsets * out.shape[1] * out.shape[2] * out.shape[3], out, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid, BLOCK_SIZE, BLOCK_SIZE](x, weight, bias, out, x.shape[1], x.shape[0], weight.shape[2:], weight.shape[3:], (weight.shape[2] - 1, weight.shape[3] - 1), BLOCK_SIZE, tl.program_id(0), tl.thread_id(0))
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(x, self.weight, self.bias)