import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    input_ptr,
    output_ptr,
    B,
    D,
    S,
    D_other,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // D_other
    other_dim = pid % D_other

    input_base = input_ptr + (batch * D * D_other) + (other_dim * S)

    max_value = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    max_index = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for i in range(0, D, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        ptrs = input_base + offsets * S
        values = tl.load(ptrs, mask=mask, other=-float('inf'))

        current_values = tl.where(mask, values, -float('inf'))
        greater = current_values > max_value

        max_value = tl.where(greater, current_values, max_value)
        max_index = tl.where(greater, offsets, max_index)

    argmax_idx = tl.argmax(max_value, axis=0)
    global_max_index = max_index[argmax_idx]
    tl.store(output_ptr + pid, global_max_index)


def triton_argmax(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()

    B, D1, D2 = x.shape
    if dim == 1:
        D = D1
        S = D2
        D_other = D2
    elif dim == 2:
        D = D2
        S = 1
        D_other = D1
    else:
        raise ValueError("Only dim 1 or 2 supported")

    output = torch.empty((B, D_other), dtype=torch.int64, device=x.device)
    grid = (B * D_other,)
    BLOCK_SIZE = 128

    argmax_kernel[grid](x, output, B, D, S, D_other, B * D_other, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)