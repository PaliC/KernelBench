import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_forward_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    if b >= B:
        return
    
    start_idx = block_idx * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    mean = tl.load(mean_ptr + b)
    var_val = tl.load(var_ptr + b)
    std = tl.sqrt(var_val + eps)

    x = tl.load(x_ptr + b * N + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    normalized = (x - mean) / std
    out = normalized * weight + bias
    tl.store(output_ptr + b * N + offsets, out, mask=mask)


@triton.jit
def layer_norm_sum_kernel(
    x_ptr,
    sum_ptr,
    sum_sq_ptr,
    B,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    if b >= B:
        return
    
    start_idx = block_idx * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + b * N + offsets, mask=mask, other=0.0)
    sum_part = tl.sum(x, axis=0)
    sum_sq_part = tl.sum(x * x, axis=0)

    tl.atomic_add(sum_ptr + b, sum_part)
    tl.atomic_add(sum_sq_ptr + b, sum_sq_part)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, D1, D2 = x.shape
        N = F * D1 * D2
        x_flat = x.contiguous().view(B, -1)
        
        sum_ = torch.zeros(B, device=x.device, dtype=x.dtype)
        sum_sq = torch.zeros(B, device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE = 1024
        grid_sum = (B, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
        layer_norm_sum_kernel[grid_sum](
            x_flat, sum_, sum_sq,
            B, N,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        mean = sum_ / N
        var = (sum_sq / N) - (mean * mean)
        
        weight_flat = self.weight.contiguous().view(-1)
        bias_flat = self.bias.contiguous().view(-1)
        output = torch.empty_like(x_flat)
        
        grid_norm = (B, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
        layer_norm_forward_kernel[grid_norm](
            x_flat, mean, var,
            weight_flat, bias_flat, output,
            B, N, self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output.view(B, F, D1, D2)