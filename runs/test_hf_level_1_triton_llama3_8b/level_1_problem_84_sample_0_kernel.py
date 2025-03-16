import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,  # Pointer to input
    w_ptr,  # Pointer to kernel
    out_ptr,  # Pointer to output
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the square convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < BLOCK_SIZE
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels * kernel_size * kernel_size, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets * out_channels * in_channels * kernel_size * kernel_size, mask=mask, other=0.0)
    # Perform the elementwise multiplication
    out = tl.dot(x, w)
    # Store the result
    tl.store(out_ptr + offsets * out_channels, out, mask=mask)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
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
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    out_channels = w.shape[0]
    kernel_size = w.shape[2]
    stride = 1
    padding = 0
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((batch_size * in_channels * kernel_size * kernel_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, out, batch_size, in_channels, out_channels, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Instead of "return self.conv2d(x)", call our Triton-based convolution
        return triton_conv2d(x, self.conv2d.weight)