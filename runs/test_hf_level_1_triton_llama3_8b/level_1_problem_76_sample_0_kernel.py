import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to kernel weights
    out_ptr,  # Pointer to output tensor
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_size,  # Size of the square convolution kernel
    stride,  # Stride of the convolution
    dilation,  # Spacing between kernel elements
    bias_ptr,  # Pointer to bias (if applicable)
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets * in_channels, mask=mask, other=0.0)
    # Load kernel weights
    w = tl.load(w_ptr + offsets * kernel_size, mask=mask, other=0.0)
    # Perform the convolution
    out = tl.dot(x, w)
    # Apply dilation
    out = tl.where(tl.arange(0, BLOCK_SIZE) < kernel_size, out, 0.0)
    # Apply stride
    out = tl.where(tl.arange(0, BLOCK_SIZE) % stride == 0, out, 0.0)
    # Store the result
    tl.store(out_ptr + offsets * out_channels, out, mask=mask)


def triton_conv1d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor = None):
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

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv1d_kernel[grid](x, w, out, x.shape[1], out.shape[1], w.shape[2], 1, 1, bias, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Instead of "return self.conv1d(x)", call our Triton-based convolution
        return triton_conv1d(x, self.conv1d.weight, self.conv1d.bias)