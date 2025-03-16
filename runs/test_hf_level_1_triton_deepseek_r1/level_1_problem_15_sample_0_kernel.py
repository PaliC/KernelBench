import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lower_tri_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m < pid_n:
        return
    
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block in range(pid_n, pid_m + 1):
        rk = k_block * BLOCK_SIZE_K
        a = tl.load(a_ptr + (rm + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am + (rk + tl.arange(0, BLOCK_SIZE_K)[None, :]),
                    mask=(rm + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (rk + tl.arange(0, BLOCK_SIZE_K)[None, :] < M),
                    other=0.0)
        b = tl.load(b_ptr + (rk + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_bk + (rn + tl.arange(0, BLOCK_SIZE_N)[None, :]),
                    mask=(rk + tl.arange(0, BLOCK_SIZE_K)[:, None] < M) & (rn + tl.arange(0, BLOCK_SIZE_N)[None, :] < M),
                    other=0.0)
        acc += tl.dot(a, b)
    
    rows = rm + tl.arange(0, BLOCK_SIZE_M)[:, None]
    cols = rn + tl.arange(0, BLOCK_SIZE_N)[None, :]
    mask = (rows >= cols) & (rows < M) & (cols < M)
    c_ptrs = c_ptr + rows * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=mask)


def triton_lower_tri_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA."
    assert a.shape == b.shape and a.shape[0] == a.shape[1], "Inputs must be square matrices."
    M = a.shape[0]
    a = a.contiguous()
    b = b.contiguous()
    c = torch.zeros_like(a)
    
    BLOCK_SIZE = 64
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(M, BLOCK_SIZE))
    
    lower_tri_matmul_kernel[grid](
        a, b, c,
        M,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE,
        BLOCK_SIZE_N=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_lower_tri_matmul(A, B)