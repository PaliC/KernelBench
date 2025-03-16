import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    input_ptr,
    output_ptr,
    n_reduce,
    reduce_dim,
    batch_size,
    dim1,
    dim2,
    stride_batch,
    stride_dim1,
    stride_dim2,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if reduce_dim == 0:
        dim1_idx = pid // dim2
        dim2_idx = pid % dim2
        base = dim1_idx * stride_dim1 + dim2_idx * stride_dim2
    elif reduce_dim == 1:
        batch_idx = pid // dim2
        dim2_idx = pid % dim2
        base = batch_idx * stride_batch + dim2_idx * stride_dim2
    else:
        batch_idx = pid // dim1
        dim1_idx = pid % dim1
        base = batch_idx * stride_batch + dim1_idx * stride_dim1

    max_val = tl.zeros((1,), tl.float32) - float('inf')
    
    for i in range(0, n_reduce, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_reduce
        
        if reduce_dim == 0:
            ptrs = input_ptr + base + offsets * stride_batch
        elif reduce_dim == 1:
            ptrs = input_ptr + base + offsets * stride_dim1
        else:
            ptrs = input_ptr + base + offsets * stride_dim2
            
        vals = tl.load(ptrs, mask=mask, other=-float('inf'))
        curr_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    
    tl.store(output_ptr + pid, max_val)


def triton_max_reduce(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    
    shape = list(x.shape)
    n_reduce = shape[dim]
    shape.pop(dim)
    output = torch.empty(shape, device=x.device)
    output_flat = output.view(-1)
    
    batch, dim1, dim2 = x.shape
    grid = (output_flat.numel(),)
    
    max_reduce_kernel[grid](
        x,
        output_flat,
        n_reduce,
        dim,
        batch,
        dim1,
        dim2,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        BLOCK_SIZE=128
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduce(x, self.dim)