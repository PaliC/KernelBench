import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # Pointer to input
    w_ptr,  # Pointer to kernel
    out_ptr,  # Pointer to output
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Kernel size (height, width)
    stride,  # Stride of the convolution
    padding,  # Padding applied to the input
    dilation,  # Spacing between kernel elements
    groups,  # Number of blocked connections
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < batch_size
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels * kernel_size[0] * kernel_size[1], mask=mask, other=0.0)
    # Load kernel values
    w = tl.load(w_ptr + offsets * out_channels * kernel_size[0] * kernel_size[1], mask=mask, other=0.0)
    # Perform the convolution
    out = tl.dot(x, w)
    # Store the result
    tl.store(out_ptr + offsets * out_channels * kernel_size[0] * kernel_size[1], out, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor, w: torch.Tensor):
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
    kernel_size = w.shape[2:]
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1

    # Determine the number of blocks needed
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Launch the Triton kernel
    conv_transpose2d_kernel[grid(batch_size, BLOCK_SIZE)](
        x, w, out, batch_size, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv_transpose2d(x, self.conv_transpose2d.weight)

# Define the grid function
def grid(batch_size, BLOCK_SIZE):
    return lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)