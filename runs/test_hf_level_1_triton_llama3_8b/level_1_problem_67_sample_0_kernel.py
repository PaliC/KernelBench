import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr,  # Pointer to input
    w_ptr,  # Pointer to weights
    out_ptr,  # Pointer to output
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the convolution kernel
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    groups,  # Number of blocked connections from input channels to output channels
    BLOCK_SIZE: tl.constexpr,
    meta,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < meta["length"]
    # Load input values
    x = tl.load(x_ptr + offsets * meta["length"], mask=mask, other=0.0)
    # Load weights values
    w = tl.load(w_ptr + offsets * kernel_size, mask=mask, other=0.0)
    # Perform the convolution
    out = tl.dot(x, w)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int, groups: int):
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
    length = x.shape[-1]

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1d_kernel[grid](x, w, out, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, BLOCK_SIZE=BLOCK_SIZE, meta={"length": length})
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Calculate the output shape
        length_out = (x.shape[-1] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out = torch.empty((x.shape[0], self.out_channels, length_out), device=x.device)

        # Create the weights tensor
        w = torch.randn(self.in_channels, self.out_channels, self.kernel_size, device=x.device)

        # Launch the Triton kernel
        triton_conv1d(x, w, out, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        return out