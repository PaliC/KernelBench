import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    n_rows,
    n_cols,
    logit_row_stride,
    logit_col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    # Compute row offset
    row_offset = row_idx * logit_row_stride
    
    # Load targets (one per row)
    target_idx = tl.load(targets_ptr + row_idx)
    
    # Compute max(logits) for numerical stability
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    logits = tl.load(logits_ptr + row_offset + col_offsets * logit_col_stride, mask=mask, other=-float('inf'))
    row_max = tl.max(logits, axis=0)
    
    # Compute log(sum(exp(logits - max)))
    shifted = tl.where(mask, tl.exp(logits - row_max), 0.0)
    row_sum = tl.sum(shifted, axis=0)
    row_log_sum = tl.log(row_sum)
    
    # Get target logit and compute loss component
    target_offset = row_offset + target_idx * logit_col_stride
    target_logit = tl.load(logits_ptr + target_offset)
    loss_component = (row_log_sum - target_logit + row_max)
    
    # Atomic add to loss
    tl.atomic_add(loss_ptr, loss_component)

def triton_cross_entropy(logits, targets):
    assert logits.is_cuda and targets.is_cuda
    logits = logits.contiguous()
    targets = targets.contiguous().long()

    n_rows, n_cols = logits.shape
    loss = torch.zeros(1, device=logits.device)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    cross_entropy_kernel[grid](
        logits, targets, loss,
        n_rows, n_cols,
        logits.stride(0),
        logits.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return loss / n_rows

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)