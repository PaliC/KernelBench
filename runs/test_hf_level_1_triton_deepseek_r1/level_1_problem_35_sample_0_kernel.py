import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    num_channels,
    num_groups,
    h,
    w,
    eps,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    c_start = g * channels_per_group
    M = channels_per_group * H * W

    # First pass: compute sum and sum of squares
    sum_partial = 0.0
    sum_sq_partial = 0.0
    for idx in range(0, M, BLOCK_SIZE):
        thread_idx = idx + tl.arange(0, BLOCK_SIZE)
        mask = thread_idx < M
        
        c_rel = thread_idx // (H * W)
        hw_idx = thread_idx % (H * W)
        hi = hw_idx // W
        wi = hw_idx % W
        c = c_start + c_rel
        
        offset = n * C * H * W + c * H * W + hi * W + wi
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        sum_partial += tl.sum(x, mask=mask)
        sum_sq_partial += tl.sum(x * x, mask=mask)

    # Block-wide reduction
    shared_sum = tl.static_shared_dtype((BLOCK_SIZE,), tl.float32)
    shared_sum_sq = tl.static_shared_dtype((BLOCK_SIZE,), tl.float32)
    tid = tl.arange(0, BLOCK_SIZE)
    shared_sum[tid] = sum_partial
    shared_sum_sq[tid] = sum_sq_partial
    tl.barrier()
    total_sum = tl.sum(shared_sum, axis=0)
    total_sum_sq = tl.sum(shared_sum_sq, axis=0)

    # Compute statistics
    mean = total_sum / M
    var = total_sum_sq / M - mean * mean
    std = tl.sqrt(var + eps)

    # Second pass: normalize and apply affine transform
    for idx in range(0, M, BLOCK_SIZE):
        thread_idx = idx + tl.arange(0, BLOCK_SIZE)
        mask = thread_idx < M
        
        c_rel = thread_idx // (H * W)
        hw_idx = thread_idx % (H * W)
        hi = hw_idx // W
        wi = hw_idx % W
        c = c_start + c_rel
        
        offset = n * C * H * W + c * H * W + hi * W + wi
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        x_normalized = (x - mean) / std
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        y = x_normalized * gamma + beta
        
        tl.store(y_ptr + offset, y, mask=mask)


def triton_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, eps: float = 1e-5):
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = 128

    grid = (N, num_groups)
    group_norm_kernel[grid](
        x, gamma, beta, y,
        num_channels=C,
        num_groups=num_groups,
        h=H, w=W,
        eps=eps,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_group_norm(x, self.gamma, self.beta, self.num_groups, self.eps)