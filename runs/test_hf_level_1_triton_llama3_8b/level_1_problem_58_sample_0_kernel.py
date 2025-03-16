import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Pointer to input
    weight_ptr,  # Pointer to weight
    bias_ptr,  # Pointer to bias
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    groups: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets * 5, mask=mask, other=0.0)
    # Load weight values
    weight = tl.load(weight_ptr + offsets * 5, mask=mask, other=0.0)
    # Perform the convolution transpose operation
    out = F.conv_transpose3d(x, weight, bias=bias_ptr, stride=stride, padding=padding, groups=groups)
    # Store the result
    tl.store(out_ptr + offsets * 5, out, mask=mask)


def triton_conv_transpose3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
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
    kernel_size = (3, 5, 7)  # Asymmetric kernel size
    stride = (1, 1, 1)  # Stride
    padding = (0, 0, 0)  # Padding
    groups = 1  # Number of groups

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose3d_kernel[grid](x, weight, bias, out, n_elements, BLOCK_SIZE, kernel_size, stride, padding, groups)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(*kernel_size, in_channels, out_channels))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv_transpose3d(x, self.weight, self.bias)