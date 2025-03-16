import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to kernel weights
    out_ptr,  # Pointer to output tensor
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the square convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    groups,  # Number of blocked connections from input channels to output channels
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ID: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = BLOCK_ID * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < x.shape[0]
    # Load input values
    x = tl.load(x_ptr + offsets * x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4], mask=mask, other=0.0)
    # Load kernel weights
    w = tl.load(w_ptr + offsets * w.shape[1] * w.shape[2] * w.shape[3], mask=mask, other=0.0)
    # Perform the convolution
    out = tl.dot(x, w)
    # Store the result
    tl.store(out_ptr + offsets * out_channels * out.shape[2] * out.shape[3] * out.shape[4], out, mask=mask)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor):
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
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    BLOCK_ID = 0  # Block ID

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv3d_kernel[grid](x, w, out, x.shape[1], x.shape[1], 1, 1, 0, 1, 1, BLOCK_SIZE, BLOCK_ID)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        # Instead of "return self.conv3d(x)", call our Triton-based convolution
        return triton_conv3d(x, self.conv3d.weight)