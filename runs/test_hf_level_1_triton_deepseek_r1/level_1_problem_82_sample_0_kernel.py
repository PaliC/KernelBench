import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional
import math

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    B, C, H, W,
    K, S, P,
    out_H, out_W,
    x_batch_stride, x_channel_stride, x_height_stride, x_width_stride,
    weight_channel_stride, weight_kh_stride, weight_kw_stride,
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = B * C * out_H * out_W
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    bc = offsets // (out_H * out_W)
    b = bc // C
    c = bc % C
    oh_ow = offsets % (out_H * out_W)
    oh = oh_ow // out_W
    ow = oh_ow % out_W

    sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kh in range(K):
        for kw in range(K):
            h_in = oh * S + kh - P
            w_in = ow * S + kw - P
            
            h_valid = (h_in >= 0) & (h_in < H)
            w_valid = (w_in >= 0) & (w_in < W)
            valid = h_valid & w_valid

            x_offset = (b * x_batch_stride + 
                       c * x_channel_stride + 
                       h_in * x_height_stride + 
                       w_in * x_width_stride)
            x_val = tl.load(x_ptr + x_offset, mask=valid & mask, other=0.0)
            
            weight_offset = c * weight_channel_stride + kh * weight_kh_stride + kw * weight_kw_stride
            weight_val = tl.load(weight_ptr + weight_offset)
            
            sum += x_val * weight_val

    out_offset = (b * out_batch_stride + 
                 c * out_channel_stride + 
                 oh * out_height_stride + 
                 ow * out_width_stride)
    tl.store(out_ptr + out_offset, sum, mask=mask)

def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, 
                           bias: Optional[torch.Tensor], stride: int, padding: int):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    B, C, H, W = x.shape
    K = weight.shape[2]
    
    out_H = (H + 2*padding - K) // stride + 1
    out_W = (W + 2*padding - K) // stride + 1
    out = torch.empty(B, C, out_H, out_W, device=x.device, dtype=x.dtype)
    
    x = x.contiguous()
    weight = weight.contiguous()
    
    total_elements = out.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    depthwise_conv2d_kernel[grid](
        x, weight, out,
        B, C, H, W,
        K, stride, padding,
        out_H, out_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, 
                stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)