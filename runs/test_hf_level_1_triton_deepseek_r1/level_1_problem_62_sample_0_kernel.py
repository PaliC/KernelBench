import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional, Tuple

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    N, C_in, H, W,
    C_out, C_in_g, K_H, K_W,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    input_n_stride, input_c_stride, input_h_stride, input_w_stride,
    weight_c_out_stride, weight_c_in_stride, weight_kh_stride, weight_kw_stride,
    output_n_stride, output_c_stride, output_h_stride, output_w_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = N * C_out * H_out * W_out
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    n = offsets // (C_out * H_out * W_out)
    remainder = offsets % (C_out * H_out * W_out)
    c_out = remainder // (H_out * W_out)
    remainder = remainder % (H_out * W_out)
    h_out = remainder // W_out
    w_out = remainder % W_out
    
    valid = mask & (n < N) & (c_out < C_out) & (h_out < H_out) & (w_out < W_out)
    
    group_idx = c_out // (C_out // groups)
    c_in_start = group_idx * (C_in // groups)
    
    acc = tl.zeros(tl.float32, BLOCK_SIZE)
    
    for c_in in range(C_in // groups):
        c_in_total = c_in_start + c_in
        for kh in range(K_H):
            for kw in range(K_W):
                h_in = h_out * stride_h - pad_h + kh * dilation_h
                w_in = w_out * stride_w - pad_w + kw * dilation_w
                
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    input_off = n * input_n_stride + c_in_total * input_c_stride + h_in * input_h_stride + w_in * input_w_stride
                    input_val = tl.load(input_ptr + input_off, mask=valid, other=0.0)
                else:
                    input_val = 0.0
                
                weight_off = c_out * weight_c_out_stride + c_in * weight_c_in_stride + kh * weight_kh_stride + kw * weight_kw_stride
                weight_val = tl.load(weight_ptr + weight_off, mask=valid, other=0.0)
                acc += input_val * weight_val
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c_out, mask=valid, other=0.0)
        acc += bias_val
    
    output_off = n * output_n_stride + c_out * output_c_stride + h_out * output_h_stride + w_out * output_w_stride
    tl.store(output_ptr + output_off, acc, mask=valid)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.shape
        K_H, K_W = self.kernel_size
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (K_H - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (K_W - 1) - 1) // self.stride[1] + 1
        
        output = torch.empty((N, self.out_channels, H_out, W_out), device=x.device, dtype=x.dtype)
        
        def grid(meta): return ((N * self.out_channels * H_out * W_out + meta['BLOCK_SIZE'] -1) // meta['BLOCK_SIZE'], )
        
        conv2d_kernel[grid](
            x, self.weight, output, self.bias,
            N, self.in_channels, H, W,
            self.out_channels, self.in_channels // self.groups, K_H, K_W,
            H_out, W_out,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), self.weight.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_SIZE=1024
        )
        
        return output