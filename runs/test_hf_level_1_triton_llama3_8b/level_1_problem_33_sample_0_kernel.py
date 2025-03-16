import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batch_norm_kernel(
    x_ptr,  # Pointer to input
    mean_ptr,  # Pointer to mean
    var_ptr,  # Pointer to variance
    gamma_ptr,  # Pointer to gamma
    beta_ptr,  # Pointer to beta
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    num_features,  # Number of features
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets * num_features, mask=mask, other=0.0)
    # Load mean and variance
    mean = tl.load(mean_ptr + offsets * num_features, mask=mask, other=0.0)
    var = tl.load(var_ptr + offsets * num_features, mask=mask, other=1.0)
    # Load gamma and beta
    gamma = tl.load(gamma_ptr + offsets * num_features, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offsets * num_features, mask=mask, other=0.0)
    # Perform the batch normalization
    out = (x - mean) / (tl.sqrt(var) + 1e-5)
    out = gamma * out + beta
    # Store the result
    tl.store(out_ptr + offsets * num_features, out, mask=mask)


def triton_batch_norm(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
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
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    num_features = x.shape[1]
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    batch_norm_kernel[grid](x, mean, var, gamma, beta, out, n_elements, num_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # Instead of "return self.bn(x)", call our Triton-based batch normalization
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.var(x, dim=[2, 3], keepdim=True)
        gamma = self.bn.weight.view(1, -1, 1, 1)
        beta = self.bn.bias.view(1, -1, 1, 1)
        return triton_batch_norm(x, mean, var, gamma, beta)