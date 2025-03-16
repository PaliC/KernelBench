import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    groups,
    in_channels,
    out_channels_g,
    H_in, W_in,
    H_out, W_out,
    kernel_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    if pid_b >= tl.num_programs(0) or pid_oc >= tl.num_programs(1) or pid_h >= H_out or pid_w >= W_out:
        return
    
    group = pid_oc // out_channels_g
    oc_local = pid_oc % out_channels_g
    
    acc = 0.0
    for kh in tl.static_range(kernel_size):
        for kw in tl.static_range(kernel_size):
            h_in = (pid_h + padding_h - kh) // stride_h
            w_in = (pid_w + padding_w - kw) // stride_w
            
            if (pid_h + padding_h - kh) % stride_h != 0:
                continue
            if (pid_w + padding_w - kw) % stride_w != 0:
                continue
            if h_in < 0 or h_in >= H_in or w_in < 0 or w_in >= W_in:
                continue
            
            for in_ch in tl.static_range(in_channels):
                input_idx = pid_b * in_channels * H_in * W_in + in_ch * H_in * W_in + h_in * W_in + w_in
                weight_idx = in_ch * out_channels_g * kernel_size * kernel_size + oc_local * kernel_size * kernel_size + kh * kernel_size + kw
                acc += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)
    
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + pid_oc)
    
    output_idx = pid_b * (out_channels_g * groups) * H_out * W_out + pid_oc * H_out * W_out + pid_h * W_out + pid_w
    tl.store(output_ptr + output_idx, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.output_padding = (output_padding, output_padding)
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x):
        B, C, H_in, W_in = x.shape
        K = self.kernel_size
        H_out = (H_in - 1) * self.stride[0] - 2 * self.padding[0] + K + self.output_padding[0]
        W_out = (W_in - 1) * self.stride[1] - 2 * self.padding[1] + K + self.output_padding[1]
        y = torch.empty((B, self.out_channels, H_out, W_out), device=x.device, dtype=x.dtype)
        
        grid = (B, self.out_channels, H_out, W_out)
        conv_transpose2d_kernel[grid](
            x, self.weight, y, self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1],
            self.groups,
            self.in_channels,
            self.out_channels // self.groups,
            H_in, W_in,
            H_out, W_out,
            self.kernel_size,
            BLOCK_SIZE_H=16, BLOCK_SIZE_W=16
        )
        return y