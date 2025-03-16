import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_depthwise_kernel(
    x_ptr,  # Pointer to input
    w_ptr,  # Pointer to depthwise kernel
    out_ptr,  # Pointer to output
    height,  # Height of the input
    width,  # Width of the input
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the kernel
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
    mask = offsets < width
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels, mask=mask, other=0.0)
    # Perform the depthwise convolution
    w = tl.load(w_ptr + offsets * in_channels, mask=mask, other=0.0)
    out = tl.dot(x, w)
    # Store the result
    tl.store(out_ptr + offsets * out_channels, out, mask=mask)


@triton.jit
def conv2d_pointwise_kernel(
    x_ptr,  # Pointer to input
    w_ptr,  # Pointer to pointwise kernel
    out_ptr,  # Pointer to output
    height,  # Height of the input
    width,  # Width of the input
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the kernel
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
    mask = offsets < width
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels, mask=mask, other=0.0)
    # Perform the pointwise convolution
    w = tl.load(w_ptr + offsets * out_channels, mask=mask, other=0.0)
    out = x * w
    # Store the result
    tl.store(out_ptr + offsets * out_channels, out, mask=mask)


def triton_conv2d_depthwise(x: torch.Tensor, w: torch.Tensor):
    """
    This function wraps the Triton kernel call for depthwise convolution. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    height, width, in_channels = x.shape[2:]
    out_channels = in_channels
    kernel_size = w.shape[-2]
    stride = 1
    padding = 0
    dilation = 1

    # Determine the number of blocks needed
    grid = lambda meta: ((width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_depthwise_kernel[grid](x, w, out, height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=128)
    return out


def triton_conv2d_pointwise(x: torch.Tensor, w: torch.Tensor):
    """
    This function wraps the Triton kernel call for pointwise convolution. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    height, width, in_channels = x.shape[2:]
    out_channels = w.shape[0]
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1

    # Determine the number of blocks needed
    grid = lambda meta: ((width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_pointwise_kernel[grid](x, w, out, height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        x = triton_conv2d_depthwise(x, self.depthwise.weight)
        x = triton_conv2d_pointwise(x, self.pointwise.weight)
        return x