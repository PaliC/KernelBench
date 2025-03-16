import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    elements_per_program = (N * M + num_pid - 1) // num_pid
    
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < N * M
    
    a_idx = (offset + tl.arange(0, BLOCK_SIZE)) // M
    b_idx = offset + tl.arange(0, BLOCK_SIZE)
    
    a = tl.load(A_ptr + a_idx, mask=mask, other=0)
    b = tl.load(B_ptr + b_idx, mask=mask, other=0)
    c = a * b
    
    tl.store(C_ptr + b_idx, c, mask=mask)


def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    C = torch.empty_like(B)
    
    grid = lambda meta: (triton.cdiv(N * M, meta['BLOCK_SIZE']),)
    diag_matmul_kernel[grid](A, B, C, N, M, BLOCK_SIZE=1024)
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_diag_matmul(A, B)