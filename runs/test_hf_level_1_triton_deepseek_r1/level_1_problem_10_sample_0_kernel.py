import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    N, M, K, L,
    stride_an, stride_am, stride_ak,
    stride_bk, stride_bl,
    stride_cn, stride_cm, stride_cl,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + pid_n * stride_an + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_l[None, :] * stride_bl
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_l[None, :] < L), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + pid_n * stride_cn + offs_m[:, None] * stride_cm + offs_l[None, :] * stride_cl
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_l[None, :] < L))


def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    L = B.shape[1]
    C = torch.empty((N, M, L), device=A.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_L = 64
    BLOCK_SIZE_K = 32
    
    grid = (
        N,
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(L, BLOCK_SIZE_L),
    )
    
    batched_matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_SIZE_M, BLOCK_SIZE_L, BLOCK_SIZE_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)