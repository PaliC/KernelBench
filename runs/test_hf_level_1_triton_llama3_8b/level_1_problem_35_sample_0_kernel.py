import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr,  # Pointer to input tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    out_ptr,  # Pointer to output tensor
    num_groups_ptr,  # Pointer to number of groups
    num_features_ptr,  # Pointer to number of features
    batch_size_ptr,  # Pointer to batch size
    dim1_ptr,  # Pointer to dim1
    dim2_ptr,  # Pointer to dim2
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < batch_size_ptr
    # Load input values
    x = tl.load(x_ptr + offsets * num_features_ptr * dim1_ptr * dim2_ptr, mask=mask, other=0.0)
    # Calculate mean and variance
    mean = tl.mean(x, axis=0, mask=mask)
    var = tl.variance(x, axis=0, mask=mask)
    # Calculate epsilon
    epsilon = 1e-5
    # Calculate inv_var
    inv_var = 1 / (var + epsilon)
    # Calculate gamma
    gamma = 1.0
    # Calculate beta
    beta = 0.0
    # Calculate output
    out = gamma * (x - mean) * inv_var + beta
    # Store the result
    tl.store(out_ptr + offsets * num_features_ptr * dim1_ptr * dim2_ptr, out, mask=mask)


def triton_group_norm(x: torch.Tensor, num_groups: int, num_features: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    batch_size = x.shape[0]
    dim1 = x.shape[2]
    dim2 = x.shape[3]
    n_elements = batch_size * num_features * dim1 * dim2
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    group_norm_kernel[grid](x, x, x, out, num_groups, num_features, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = nn.ModuleList([triton_group_norm for _ in range(num_groups)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return torch.cat([gn(x) for gn in self.gn], dim=1)