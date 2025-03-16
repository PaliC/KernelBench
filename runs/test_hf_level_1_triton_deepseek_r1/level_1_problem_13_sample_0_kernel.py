import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def symmetric_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    N,  # Matrix size (N x N)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # Calculate block range for this program
    blocks_per_row = tl.cdiv(N, BLOCK_SIZE)
    row = pid // blocks_per_row
    col = pid % blocks_per_row

    # Offsets for the block
    row_start = row * BLOCK_SIZE
    col_start = col * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks to handle boundary conditions
    row_mask = row_offsets < N
    col_mask = col_offsets < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < N
        
        # Load blocks from symmetric matrices (optimized memory access)
        a = tl.load(a_ptr + row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Since B is symmetric, we can load columns as rows
        b = tl.load(b_ptr + k_offsets[:, None] * stride_bk + col_offsets[None, :] * stride_bn,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=True)

    # Store result
    tl.store(c_ptr + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn,
             acc, mask=row_mask[:, None] & col_mask[None, :])


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    a = a.contiguous()
    b = b.contiguous()
    
    N = a.size(0)
    c = torch.empty_like(a)
    
    # Tuned for A100 with BLOCK_SIZE=64
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE) * triton.cdiv(N, BLOCK_SIZE),)
    
    symmetric_matmul_kernel[grid](
        a, b, c,
        N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)