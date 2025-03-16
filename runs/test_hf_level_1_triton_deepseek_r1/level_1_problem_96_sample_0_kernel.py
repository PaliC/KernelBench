import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_kernel(
    pred_ptr,
    target_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    diff = pred - target
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    
    sum_loss = tl.sum(loss, axis=0)
    tl.store(partial_sums_ptr + pid, sum_loss)


def triton_smooth_l1_loss(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA"
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tune based on hardware

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )
    num_blocks = grid({"BLOCK_SIZE": BLOCK_SIZE})[0]
    partial_sums = torch.zeros(num_blocks, device=predictions.device, dtype=predictions.dtype)

    smooth_l1_kernel[grid](predictions, targets, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return partial_sums.sum() / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)