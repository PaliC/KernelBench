import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    sum_total = 0
    i = 0
    while i < N:
        current_sum = N - i
        if sum_total + current_sum > pid:
            break
        sum_total += current_sum
        i += 1
    j = pid - sum_total + i
    if j >= N:
        return
    
    acc = 0.0
    start_k = i
    end_k = j + 1
    num_vectors = (end_k - start_k + VECTOR_SIZE - 1) // VECTOR_SIZE
    for vec in range(num_vectors):
        k_start = start_k + vec * VECTOR_SIZE
        k_end = min(k_start + VECTOR_SIZE, end_k)
        k_indices = k_start + tl.arange(0, VECTOR_SIZE)
        mask = k_indices < k_end
        
        a_offsets = i * stride_am + k_indices * stride_ak
        a = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
        
        b_offsets = k_indices * stride_bk + j * stride_bn
        b = tl.load(B_ptr + b_offsets, mask=mask, other=0.0)
        
        acc += tl.sum(a * b)
    
    tl.store(C_ptr + i * stride_cm + j * stride_cn, acc)


def triu_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape and A.shape[0] == A.shape[1], "Matrices must be square."
    N = A.shape[0]
    C = torch.zeros_like(A)
    
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    total_elements = (N * (N + 1)) // 2
    BLOCK_SIZE = 128
    VECTOR_SIZE = 16
    
    grid = lambda meta: ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    triu_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        VECTOR_SIZE=VECTOR_SIZE
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triu_matmul(A, B)