import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, K)[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + tl.arange(0, K)[:, None] * stride_bk
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    
    offs_cm = offs_m[:, None]
    offs_cn = offs_n[None, :]
    c_ptrs = c_ptr + offs_cm * stride_cm + offs_cn * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_cm < M) & (offs_cn < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda opt: (triton.cdiv(M, opt['BLOCK_M']), triton.cdiv(N, opt['BLOCK_N']))
    
    matmul_kernel[grid](
        a, b.t().contiguous(), c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4,
        num_stages=3,
    )
    return c

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)