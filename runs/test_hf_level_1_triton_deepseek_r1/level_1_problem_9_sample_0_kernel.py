import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bm + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_bk
    
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < K, other=0.0)
    b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < K, other=0.0)

    c = tl.dot(a, b, trans_b=True)
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"Incompatible dimensions: {K} vs {K_b}"
    b_t = b.transpose(0, 1).contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 16
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        a, b_t, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_t.stride(0), b_t.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)