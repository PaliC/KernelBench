import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    B, C, H, W,
    K,
    stride_h,
    padding_h,
    dilation_h,
    H_out,
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    weight_stride_c,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_w = tl.program_id(2)

    if pid_b >= B or pid_c >= C:
        return

    w_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    w_mask = w_offsets < W

    weight_offset = pid_c * weight_stride_c
    weights = tl.load(weight_ptr + weight_offset + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)

    input_bc_offset = pid_b * input_stride_b + pid_c * input_stride_c
    output_bc_offset = pid_b * output_stride_b + pid_c * output_stride_c

    for h_out in range(H_out):
        acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)
        for k in range(K):
            h_in = h_out * stride_h - padding_h + k * dilation_h
            h_valid = (h_in >= 0) & (h_in < H)
            input_offset = input_bc_offset + h_in * input_stride_h + w_offsets * input_stride_w
            input_val = tl.load(input_ptr + input_offset, mask=h_valid & w_mask, other=0.0)
            acc += input_val * weights[k]
        output_offset = output_bc_offset + h_out * output_stride_h + w_offsets * output_stride_w
        tl.store(output_ptr + output_offset, acc, mask=w_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
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
        output = triton_depthwise_conv2d(x, self.weight, self.stride, 
                                       self.padding, self.dilation)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output


def triton_depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, 
                          stride: int, padding: int, dilation: int):
    x = x.contiguous()
    weight = weight.contiguous()
    B, C, H, W = x.shape
    K = weight.shape[2]
    
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    output = torch.empty((B, C, H_out, W), device=x.device, dtype=x.dtype)
    
    input_strides = x.stride()
    output_strides = output.stride()
    
    grid = (B, C, triton.cdiv(W, 128))
    
    depthwise_conv2d_kernel[grid](
        x, weight, output,
        B, C, H, W,
        K,
        stride,
        padding,
        dilation,
        H_out,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        weight.stride(0),
        BLOCK_SIZE_W=128,
    )
    return output