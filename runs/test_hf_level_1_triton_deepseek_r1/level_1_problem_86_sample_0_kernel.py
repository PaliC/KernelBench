import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_depthwise_pointwise_kernel(
    input_ptr,
    depthwise_weight_ptr,
    pointwise_weight_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_h_stride,
    input_w_stride,
    depthwise_channel_stride,
    depthwise_kh_stride,
    depthwise_kw_stride,
    pointwise_out_stride,
    pointwise_in_stride,
    output_batch_stride,
    output_channel_stride,
    output_h_stride,
    output_w_stride,
    B, Cin, H, W,
    Cout, K,
    stride, padding, dilation,
    H_out, W_out,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_h = tl.cdiv(H_out, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(W_out, BLOCK_SIZE_W)
    pid_batch = pid // (num_pid_h * num_pid_w)
    pid = pid % (num_pid_h * num_pid_w)
    pid_h = (pid // num_pid_w) * BLOCK_SIZE_H
    pid_w = (pid % num_pid_w) * BLOCK_SIZE_W
    
    batch = pid_batch
    co = tl.arange(0, Cout)
    
    for c in range(0, Cin, BLOCK_SIZE_C):
        ci = c + tl.arange(0, BLOCK_SIZE_C)
        for kh in range(K):
            for kw in range(K):
                h_in = pid_h * stride - padding + kh * dilation
                w_in = pid_w * stride - padding + kw * dilation
                if h_in >= 0 and h_in < H and w_in >=0 and w_in < W:
                    input_offset = batch * input_batch_stride + ci * input_channel_stride + h_in * input_h_stride + w_in * input_w_stride
                    dw_offset = ci * depthwise_channel_stride + kh * depthwise_kh_stride + kw * depthwise_kw_stride
                    input_val = tl.load(input_ptr + input_offset, mask=ci < Cin, other=0.0)
                    dw_val = tl.load(depthwise_weight_ptr + dw_offset, mask=ci < Cin, other=0.0)
                    depthwise = input_val * dw_val
                    
                    pw_offset = co[:, None] * pointwise_out_stride + ci[None, :] * pointwise_in_stride
                    pw_val = tl.load(pointwise_weight_ptr + pw_offset, mask=(co[:, None] < Cout) & (ci[None, :] < Cin), other=0.0)
                    partial = tl.sum(depthwise[None, :] * pw_val, axis=1)
                    
                    for oh in range(BLOCK_SIZE_H):
                        for ow in range(BLOCK_SIZE_W):
                            h_idx = pid_h + oh
                            w_idx = pid_w + ow
                            if h_idx < H_out and w_idx < W_out:
                                output_offset = batch * output_batch_stride + co * output_channel_stride + h_idx * output_h_stride + w_idx * output_w_stride
                                tl.atomic_add(output_ptr + output_offset, partial)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.depthwise_weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        if bias:
            nn.init.uniform_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, H, W = x.shape
        K = self.kernel_size
        H_out = (H + 2*self.padding - self.dilation*(K-1) -1) // self.stride +1
        W_out = (W + 2*self.padding - self.dilation*(K-1) -1) // self.stride +1
        output = torch.zeros((B, self.out_channels, H_out, W_out), device=x.device, dtype=x.dtype)
        
        def grid(meta):
            return (triton.cdiv(B * H_out * W_out, meta['BLOCK_SIZE_H'] * meta['BLOCK_SIZE_W']),)
        
        fused_depthwise_pointwise_kernel[grid](
            x, self.depthwise_weight, self.pointwise_weight, output,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.depthwise_weight.stride(0), self.depthwise_weight.stride(2), self.depthwise_weight.stride(3),
            self.pointwise_weight.stride(0), self.pointwise_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            B, Cin, H, W,
            self.out_channels, K,
            self.stride, self.padding, self.dilation,
            H_out, W_out,
            BLOCK_SIZE_C=32,
            BLOCK_SIZE_H=16,
            BLOCK_SIZE_W=16,
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output