import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,  # Pointer to input tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    scale_ptr,  # Pointer to scale tensor
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
    num_features: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets * num_features, mask=mask, other=0.0)
    # Load mean and variance values
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    var = tl.load(var_ptr + offsets, mask=mask, other=1.0)
    # Calculate the normalized value
    normalized = (x - mean) / tl.sqrt(var + 1e-5)
    # Load scale and bias values
    scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    # Apply scale and bias
    out = scale * normalized + bias
    # Store the result
    tl.store(out_ptr + offsets * num_features, out, mask=mask)


def triton_instance_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    scale = scale.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    num_features = x.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    instance_norm_kernel[grid](x, mean, var, scale, bias, out, n_elements, BLOCK_SIZE, num_features)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        # Calculate mean and variance
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.var(x, dim=[2, 3], keepdim=True)
        scale = torch.ones_like(mean)
        bias = torch.zeros_like(mean)
        # Instead of "return self.inorm(x)", call our Triton-based instance normalization
        return triton_instance_norm(x, mean, var, scale, bias)