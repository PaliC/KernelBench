import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    predictions = tl.load(predictions_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + offsets, mask=mask)

    log_p = tl.log(predictions)
    log_t = tl.log(targets)
    term = targets * (log_t - log_p)
    block_sum = tl.sum(term, axis=0)

    tl.atomic_add(output_ptr, block_sum)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    n_elements = predictions.numel()
    batch_size = predictions.size(0)
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    kl_div_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output / batch_size


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)