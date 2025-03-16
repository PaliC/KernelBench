import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr,
    out_ptr,
    numel,
    eps,
    num_features,
    stride_batch,
    stride_feature,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_features
    feature_idx = pid % num_features

    x_channel_ptr = x_ptr + batch_idx * stride_batch + feature_idx * stride_feature
    out_channel_ptr = out_ptr + batch_idx * stride_batch + feature_idx * stride_feature

    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    thread_idx = tl.arange(0, BLOCK_SIZE)

    for i in range(0, numel, BLOCK_SIZE):
        elem_idx = i + thread_idx
        mask = elem_idx < numel
        x = tl.load(x_channel_ptr + elem_idx, mask=mask, other=0.0)
        sum += x
        sum_sq += x * x

    total_sum = tl.sum(sum, axis=0)
    total_sum_sq = tl.sum(sum_sq, axis=0)
    mean = total_sum / numel
    variance = (total_sum_sq / numel) - (mean * mean)
    variance += eps
    std_inv = 1.0 / tl.sqrt(variance)

    for i in range(0, numel, BLOCK_SIZE):
        elem_idx = i + thread_idx
        mask = elem_idx < numel
        x = tl.load(x_channel_ptr + elem_idx, mask=mask, other=0.0)
        normalized = (x - mean) * std_inv
        tl.store(out_channel_ptr + elem_idx, normalized, mask=mask)


def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda and x.is_contiguous(), "Input must be contiguous and on CUDA."
    x = x.contiguous()
    batch_size, num_features, H, W = x.shape
    numel = H * W
    out = torch.empty_like(x)
    stride_batch = x.stride(0)
    stride_feature = x.stride(1)
    grid = (batch_size * num_features,)
    BLOCK_SIZE = 1024

    instance_norm_kernel[grid](
        x, out, numel, eps, num_features,
        stride_batch, stride_feature,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x, eps=1e-5)