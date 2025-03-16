import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cosine_similarity_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,  # Pointer to targets
    cosine_sim_ptr,  # Pointer to cosine similarity
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
    predictions = tl.load(predictions_ptr + offsets * dim, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets * dim, mask=mask, other=0.0)
    # Compute dot product
    dot_product = tl.dot(predictions, targets, mask=mask)
    # Compute magnitudes
    predictions_magnitude = tl.sqrt(tl.sum(predictions ** 2, mask=mask))
    targets_magnitude = tl.sqrt(tl.sum(targets ** 2, mask=mask))
    # Compute cosine similarity
    cosine_sim = dot_product / (predictions_magnitude * targets_magnitude)
    # Store the result
    tl.store(cosine_sim_ptr + offsets, cosine_sim, mask=mask)


@triton.jit
def mean_kernel(
    cosine_sim_ptr,  # Pointer to cosine similarity
    out_ptr,  # Pointer to output
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
    cosine_sim = tl.load(cosine_sim_ptr + offsets, mask=mask, other=0.0)
    # Compute mean
    out = tl.mean(cosine_sim, mask=mask)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_cosine_similarity(predictions: torch.Tensor, targets: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Prepare output tensor
    cosine_sim = torch.empty_like(predictions)

    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    cosine_similarity_kernel[grid](predictions, targets, cosine_sim, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return cosine_sim


def triton_mean(cosine_sim: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert cosine_sim.is_cuda, "Tensor must be on CUDA."
    cosine_sim = cosine_sim.contiguous()

    # Prepare output tensor
    out = torch.empty_like(cosine_sim)

    # Number of elements in the tensor
    n_elements = cosine_sim.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    mean_kernel[grid](cosine_sim, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Instead of "return torch.mean(1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=1))",
        # call our Triton-based cosine similarity and mean
        cosine_sim = triton_cosine_similarity(predictions, targets)
        return triton_mean(1 - cosine_sim)