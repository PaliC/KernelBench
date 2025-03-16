import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    pred_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)
    
    diff = pred - target
    squared = diff * diff
    
    # Reduction within block
    block_sum = tl.sum(squared, axis=0)
    tl.atomic_add(output_ptr, block_sum)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    n_elements = predictions.numel()
    
    total_sum = torch.zeros(1, device=predictions.device)
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mse_kernel[grid](predictions, targets, total_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return total_sum / n_elements

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)