import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batchnorm_forward_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    eps,
    num_elements_per_channel,
    H,
    W,
    N,
    C,
    input_batch_stride,
    input_channel_stride,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    
    pid = tl.program_id(1)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements_per_channel
    
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    
    sum = 0.0
    sum_sq = 0.0
    
    for i in range(0, tl.cdiv(num_elements_per_channel, BLOCK_SIZE)):
        curr_offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        curr_mask = curr_offsets < num_elements_per_channel
        
        n = curr_offsets // (H * W)
        remaining = curr_offsets % (H * W)
        h = remaining // W
        w = remaining % W
        
        input_offsets = n * input_batch_stride + c * input_channel_stride + h * W + w
        x = tl.load(input_ptr + input_offsets, mask=curr_mask, other=0.0)
        
        sum += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    
    mean = sum / num_elements_per_channel
    var = (sum_sq / num_elements_per_channel) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for i in range(0, tl.cdiv(num_elements_per_channel, BLOCK_SIZE)):
        curr_offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        curr_mask = curr_offsets < num_elements_per_channel
        
        n = curr_offsets // (H * W)
        remaining = curr_offsets % (H * W)
        h = remaining // W
        w = remaining % W
        
        input_offsets = n * input_batch_stride + c * input_channel_stride + h * W + w
        x = tl.load(input_ptr + input_offsets, mask=curr_mask, other=0.0)
        
        normalized = (x - mean) * inv_std * gamma + beta
        tl.store(output_ptr + input_offsets, normalized, mask=curr_mask)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA"
        x = x.contiguous()
        N, C, H, W = x.shape
        output = torch.empty_like(x)
        num_elements_per_channel = N * H * W
        
        BLOCK_SIZE = 128
        grid = (C, triton.cdiv(num_elements_per_channel, BLOCK_SIZE))
        
        batchnorm_forward_kernel[grid](
            x,
            self.gamma,
            self.beta,
            output,
            self.eps,
            num_elements_per_channel,
            H,
            W,
            N,
            C,
            x.stride(0),
            x.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output