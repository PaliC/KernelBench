import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, stride_ab,
    stride_bk, stride_bn, stride_bb,
    stride_cm, stride_cn, stride_cb,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    batch_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_m // GROUP_M
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    offs_m = first_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + batch_idx * stride_ab + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + batch_idx * stride_bb + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_idx in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (offs_k[None, :] < k - k_idx * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k - k_idx * BLOCK_K) & (offs_n[None, :] < n), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + batch_idx * stride_cb + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    
    c_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def triton_bmm(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.shape[0] == b.shape[0], "Batch dimensions must match"
    
    batch_size, m, k = a.shape
    _, k_, n = b.shape
    assert k == k_, "Inner dimensions must match"
    
    c = torch.empty((batch_size, m, n), device=a.device, dtype=a.dtype)
    
    grid = lambda meta: (
        batch_size,
        triton.cdiv(m, meta['BLOCK_M']),
        triton.cdiv(n, meta['BLOCK_N']),
    )
    
    bmm_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(1), a.stride(2), a.stride(0),
        b.stride(1), b.stride(2), b.stride(0),
        c.stride(1), c.stride(2), c.stride(0),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        GROUP_M=8,
        num_warps=4,
        num_stages=3
    )
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)