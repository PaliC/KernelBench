import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    in_channels,
    out_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h,
    kernel_w,
    input_bs,
    input_cs,
    input_hs,
    input_ws,
    weight_ics,
    weight_ocs,
    weight_hs,
    weight_ws,
    output_bs,
    output_cs,
    output_hs,
    output_ws,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    group_id = pid_c // (out_channels // groups)
    group_channels = out_channels // groups
    
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    c_mask = c_offsets < out_channels
    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w
    
    acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    cin_start = group_id * (in_channels // groups)
    cin_end = (group_id + 1) * (in_channels // groups)
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for cin in range(cin_start, cin_end):
                for b in range(input_bs):
                    input_h = (h_offsets[:, None, None] + padding_h - kh * dilation_h) // stride_h
                    input_w = (w_offsets[None, :, None] + padding_w - kw * dilation_w) // stride_w
                    
                    valid_h = (input_h >= 0) & (input_h < in_h)
                    valid_w = (input_w >= 0) & (input_w < in_w)
                    valid = valid_h & valid_w
                    
                    input_val = tl.load(
                        input_ptr + b * input_bs + cin * input_cs + 
                        input_h * input_hs + input_w * input_ws,
                        mask=valid,
                        other=0.0
                    )
                    
                    weight_val = tl.load(
                        weight_ptr + cin * weight_ics + 
                        c_offsets[:, None, None] * weight_ocs +
                        kh * weight_hs + kw * weight_ws,
                        mask=c_mask[:, None, None],
                        other=0.0
                    )
                    
                    acc += tl.where(valid, input_val * weight_val, 0.0)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c_offsets[:, None, None], mask=c_mask[:, None, None], other=0.0)
        acc += bias
    
    output_offset = (
        pid_b * output_bs + 
        c_offsets[:, None, None] * output_cs +
        h_offsets[:, None, None] * output_hs +
        w_offsets[None, :, None] * output_ws
    )
    
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=(c_mask[:, None, None] & h_mask[:, None, None] & w_mask[None, :, None])
    )


def conv_transpose2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1
):
    batch, in_c, in_h, in_w = x.shape
    out_c = weight.shape[1] * groups
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + 1
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + 1
    
    output = torch.empty((batch, out_c, out_h, out_w), device=x.device, dtype=x.dtype)
    
    BLOCK_C = 16
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid = (
        batch,
        triton.cdiv(out_c, BLOCK_C),
        triton.cdiv(out_h, BLOCK_H),
        triton.cdiv(out_w, BLOCK_W),
    )
    
    conv_transpose2d_kernel[grid](
        x, weight, output, bias,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        in_c, out_c,
        in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_triton(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )