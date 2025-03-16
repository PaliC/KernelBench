import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, I, J, L, K,
    stride_ab, stride_ai, stride_aj, stride_al,
    stride_bl, stride_bk,
    stride_cb, stride_ci, stride_cj, stride_ck,
    BLOCK_SIZE_L: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    
    pid_k = tl.program_id(3)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    offs_b = pid_b
    offs_i = pid_i
    offs_j = pid_j
    
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(B, I, J, L),
        strides=(stride_ab, stride_ai, stride_aj, stride_al),
        offsets=(pid_b, pid_i, pid_j, 0),
        block_shape=(1, 1, 1, BLOCK_SIZE_L),
        order=(3, 2, 1, 0)
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(L, K),
        strides=(stride_bl, stride_bk),
        offsets=(0, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_L, BLOCK_SIZE_K),
        order=(1, 0)
    )
    
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(B, I, J, K),
        strides=(stride_cb, stride_ci, stride_cj, stride_ck),
        offsets=(pid_b, pid_i, pid_j, pid_k * BLOCK_SIZE_K),
        block_shape=(1, 1, 1, BLOCK_SIZE_K),
        order=(3, 2, 1, 0)
    )
    
    acc = tl.zeros((1, BLOCK_SIZE_K), dtype=tl.float32)
    
    for l in range(0, tl.cdiv(L, BLOCK_SIZE_L)):
        a = tl.load(a_block_ptr, boundary_check=(3,))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_block_ptr = tl.advance(a_block_ptr, (0, 0, 0, BLOCK_SIZE_L))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_L, 0))
    
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(3,))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        B_dim, I_dim, J_dim, L_dim = A.shape
        L_check, K_dim = B.shape
        assert L_dim == L_check
        
        C = torch.empty((B_dim, I_dim, J_dim, K_dim), device=A.device, dtype=A.dtype)
        
        grid = (B_dim, I_dim, J_dim, triton.cdiv(K_dim, 64))
        
        batched_matmul_kernel[grid](
            A, B, C,
            B_dim, I_dim, J_dim, L_dim, K_dim,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            BLOCK_SIZE_L=64, BLOCK_SIZE_K=64
        )
        
        return C