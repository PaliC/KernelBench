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
    kernel_size_h,  # Height of the kernel
    kernel_size_w,  # Width of the kernel
    stride_h,  # Stride in height dimension
    stride_w,  # Stride in width dimension
    padding_h,  # Padding in height dimension
    padding_w,  # Padding in width dimension
    dilation_h,  # Spacing between kernel elements in height dimension
    dilation_w,  # Spacing between kernel elements in width dimension
    groups,  # Number of blocked connections from input channels to output channels
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < batch_size
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels * kernel_size_h * kernel_size_w, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets * out_channels * in_channels * kernel_size_h * kernel_size_w, mask=mask, other=0.0)
    # Perform the convolution
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
    kernel_size_h = w.shape[2]
    kernel_size_w = w.shape[3]
    stride_h = 1
    stride_w = 1
    padding_h = 0
    padding_w = 0
    dilation_h = 1
    dilation_w = 1
    groups = in_channels

    # Determine the number of blocks needed
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, out, batch_size, in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.weight) + self.bias.view(1, -1, 1, 1) if self.bias else triton_conv2d(x, self.weight)