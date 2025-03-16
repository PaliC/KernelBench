import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to kernel tensor
    out_ptr,  # Pointer to output tensor
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    groups,  # Number of blocked connections
    BLOCK_SIZE: tl.constexpr,
    meta,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < meta["n_elements"]
    # Load input values
    x = tl.load(x_ptr + offsets * meta["in_channels"], mask=mask, other=0.0)
    # Load kernel values
    w = tl.load(w_ptr + offsets * meta["kernel_size"] * meta["in_channels"], mask=mask, other=0.0)
    # Perform the convolution
    out = tl.dot(x, w, groups=groups)
    # Apply stride and padding
    out = tl.pad(out, padding, mode="constant", value=0)
    out = tl.stride(out, stride)
    # Store the result
    tl.store(out_ptr + offsets * meta["out_channels"], out, mask=mask)


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, dilation: tuple, groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda and out.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    out = out.contiguous()

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Prepare meta data
    meta = {
        "n_elements": n_elements,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size[0] * kernel_size[1] * kernel_size[2],
        "stride": stride[0] * stride[1] * stride[2],
        "padding": padding[0] * padding[1] * padding[2],
        "dilation": dilation[0] * dilation[1] * dilation[2],
        "groups": groups,
        "BLOCK_SIZE": BLOCK_SIZE,
    }

    # Launch the Triton kernel
    conv3d_kernel[grid][meta](x, w, out, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Instead of "return self.conv3d(x)", call our Triton-based convolution
        return triton_conv3d(x, self.conv3d.weight, torch.zeros_like(self.conv3d.weight), self.conv3d.in_channels, self.conv3d.out_channels, self.conv3d.kernel_size, self.conv3d.stride, self.conv3d.padding, self.conv3d.dilation, self.conv3d.groups)