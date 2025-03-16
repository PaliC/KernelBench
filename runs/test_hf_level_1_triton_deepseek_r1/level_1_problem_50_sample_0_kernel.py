import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def prod_reduce_kernel(
    x_ptr,
    out_ptr,
    B,
    D1,
    D2,
    stride_b,
    stride_k,
    stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)
    x_base = x_ptr + b * stride_b + d * stride_d
    
    tid = tl.arange(0, BLOCK_SIZE)
    total_product = 1.0
    shared = tl.static_shared_memory((BLOCK_SIZE,), tl.float32)
    
    for k in range(0, D1, BLOCK_SIZE):
        current_k = k + tid
        mask = current_k < D1
        x_val = tl.load(x_base + current_k * stride_k, mask=mask, other=1.0)
        shared[tid] = x_val
        tl.barrier()
        
        active_threads = BLOCK_SIZE
        while active_threads > 1:
            half = active_threads // 2
            if tid < half:
                shared[tid] *= shared[tid + half]
            active_threads = half
            tl.barrier()
        
        if tid == 0:
            total_product *= shared[0]
        tl.barrier()
    
    if tid == 0:
        out_offset = b * D2 + d
        tl.store(out_ptr + out_offset, total_product)


def triton_prod(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim != 1:
        return torch.prod(x, dim=dim)
    
    assert x.is_cuda and x.is_contiguous()
    B, D1, D2 = x.shape
    out = torch.empty((B, D2), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = min(triton.next_power_of_2(D1), 1024)
    grid = (B, D2)
    
    prod_reduce_kernel[grid](
        x, out, B, D1, D2,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_prod(x, self.dim)