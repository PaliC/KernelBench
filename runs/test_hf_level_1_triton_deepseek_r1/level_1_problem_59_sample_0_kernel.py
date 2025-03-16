import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    B,
    C_in,
    C_out,
    H,
    W,
    D,
    K,
    S,
    P,
    H_out,
    W_out,
    x_stride_b,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    x_stride_d,
    w_stride_oc,
    w_stride_ic,
    w_stride_kh,
    w_stride_kw,
    out_stride_b,
    out_stride_oc,
    out_stride_h,
    out_stride_w,
    out_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    total_operations = B * C_out * H_out * W_out * D
    if pid >= total_operations:
        return

    d = pid % D
    pid = pid // D
    w_out = pid % W_out
    pid = pid // W_out
    h_out = pid % H_out
    pid = pid // H_out
    oc = pid % C_out
    b = pid // C_out

    acc = 0.0
    for ic in range(C_in):
        for kh in range(K):
            h_in = h_out * S - P + kh
            if h_in < 0 or h_in >= H:
                continue
            for kw in range(K):
                w_in = w_out * S - P + kw
                if w_in < 0 or w_in >= W:
                    continue
                
                x_offset = b * x_stride_b + ic * x_stride_c + h_in * x_stride_h + w_in * x_stride_w + d * x_stride_d
                w_offset = oc * w_stride_oc + ic * w_stride_ic + kh * w_stride_kh + kw * w_stride_kw
                
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc)
        acc += bias_val

    out_offset = b * out_stride_b + oc * out_stride_oc + h_out * out_stride_h + w_out * out_stride_w + d * out_stride_d
    tl.store(out_ptr + out_offset, acc)

def triton_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int):
    B, C_in, H, W, D = x.shape
    C_out, _, K, _, _ = weight.shape
    
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    output = torch.empty((B, C_out, H_out, W_out, D), device=x.device, dtype=x.dtype)
    
    def grid(meta):
        return (triton.cdiv(B * C_out * H_out * W_out * D, meta['BLOCK_SIZE']),)
    
    conv3d_kernel[grid](
        x, weight, output, bias,
        B, C_in, C_out, H, W, D, K, stride, padding, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_SIZE=128
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        assert groups == 1 and dilation == 1, "Only groups=1 and dilation=1 supported"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_conv3d(x, self.weight, self.bias, self.stride, self.padding)