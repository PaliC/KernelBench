import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    
    mask_m = (rm < M)[:, 0]
    mask_n = (rn < N)[0, :]
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + rm * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + rn * stride_bn
        
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offs[:, None] < K) & mask_n[None, :], other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=True)
    
    c_ptrs = c_ptr + rm * stride_cm + rn * stride_cn
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])

def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()
    
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, f"Dimension mismatch: {K} vs {K_check}"
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
    )
    return c

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)