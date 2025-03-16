import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    num_features,
    eps,
    stride_b,
    stride_f,
    stride_d1,
    stride_d2,
    B, D1, D2,
    BLOCK_SIZE_F: tl.constexpr,
):
    b = tl.program_id(0)
    d1 = tl.program_id(1)
    d2 = tl.program_id(2)
    
    if b >= B or d1 >= D1 or d2 >= D2:
        return

    base_offset = b * stride_b + d1 * stride_d1 + d2 * stride_d2
    f_offsets = base_offset + tl.arange(0, BLOCK_SIZE_F) * stride_f
    mask = tl.arange(0, BLOCK_SIZE_F) < num_features

    x = tl.load(x_ptr + f_offsets, mask=mask, other=0.0)
    x_squared = x * x
    sum_sq = tl.sum(x_squared, axis=0)
    mean_sq = sum_sq / num_features
    rms = tl.sqrt(mean_sq + eps)
    normalized = x / rms
    tl.store(out_ptr + f_offsets, normalized, mask=mask)

def triton_rms_norm(x: torch.Tensor, num_features: int, eps: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    B, F, D1, D2 = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    
    stride_b = x.stride(0)
    stride_f = x.stride(1)
    stride_d1 = x.stride(2)
    stride_d2 = x.stride(3)

    BLOCK_SIZE_F = triton.next_power_of_2(num_features)
    grid = (B, D1, D2)
    
    rms_norm_kernel[grid](
        x, out, num_features, eps,
        stride_b, stride_f, stride_d1, stride_d2,
        B, D1, D2,
        BLOCK_SIZE_F=BLOCK_SIZE_F,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.num_features, self.eps)