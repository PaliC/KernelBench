import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    in_depth,
    in_height,
    in_width,
    out_channels,
    out_depth,
    out_height,
    out_width,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    groups,
    input_b_stride,
    input_c_stride,
    input_d_stride,
    input_h_stride,
    input_w_stride,
    weight_oc_stride,
    weight_ic_stride,
    weight_d_stride,
    weight_h_stride,
    weight_w_stride,
    output_b_stride,
    output_c_stride,
    output_d_stride,
    output_h_stride,
    output_w_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = batch_size * out_channels * out_depth * out_height * out_width
    if pid * BLOCK_SIZE >= num_elements:
        return
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    b = offsets // (out_channels * out_depth * out_height * out_width)
    remainder = offsets % (out_channels * out_depth * out_height * out_width)
    oc = remainder // (out_depth * out_height * out_width)
    remainder = remainder % (out_depth * out_height * out_width)
    d_out = remainder // (out_height * out_width)
    remainder = remainder % (out_height * out_width)
    h_out = remainder // out_width
    w_out = remainder % out_width
    
    group_idx = oc // (out_channels // groups)
    in_per_group = in_channels // groups
    ic_start = group_idx * in_per_group
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for ic in range(in_per_group):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    d_in = (d_out - kd + pad_d) // stride_d
                    h_in = (h_out - kh + pad_h) // stride_h
                    w_in = (w_out - kw + pad_w) // stride_w
                    
                    valid = (d_out - kd + pad_d) % stride_d == 0
                    valid &= (h_out - kh + pad_h) % stride_h == 0
                    valid &= (w_out - kw + pad_w) % stride_w == 0
                    valid &= (d_in >= 0) & (d_in < in_depth)
                    valid &= (h_in >= 0) & (h_in < in_height)
                    valid &= (w_in >= 0) & (w_in < in_width)
                    
                    input_off = b * input_b_stride + (ic_start + ic) * input_c_stride + d_in * input_d_stride + h_in * input_h_stride + w_in * input_w_stride
                    weight_off = oc * weight_oc_stride + ic * weight_ic_stride + kd * weight_d_stride + kh * weight_h_stride + kw * weight_w_stride
                    
                    input_val = tl.load(input_ptr + input_off, mask=valid & mask, other=0.0)
                    weight_val = tl.load(weight_ptr + weight_off, mask=valid & mask, other=0.0)
                    acc += input_val * weight_val
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc, mask=mask, other=0.0)
        acc += bias
    
    output_off = b * output_b_stride + oc * output_c_stride + d_out * output_d_stride + h_out * output_h_stride + w_out * output_w_stride
    tl.store(output_ptr + output_off, acc, mask=mask)

def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int
):
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda)
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, in_depth, in_height, in_width = x.shape
    _, out_channels_div_groups, kernel_d, kernel_h, kernel_w = weight.shape
    out_channels = out_channels_div_groups * groups
    
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    
    out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d
    out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
    out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w
    
    output = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), device=x.device, dtype=x.dtype)
    
    input_b_stride = x.stride(0)
    input_c_stride = x.stride(1)
    input_d_stride = x.stride(2)
    input_h_stride = x.stride(3)
    input_w_stride = x.stride(4)
    
    weight_oc_stride = weight.stride(0)
    weight_ic_stride = weight.stride(1)
    weight_d_stride = weight.stride(2)
    weight_h_stride = weight.stride(3)
    weight_w_stride = weight.stride(4)
    
    output_b_stride = output.stride(0)
    output_c_stride = output.stride(1)
    output_d_stride = output.stride(2)
    output_h_stride = output.stride(3)
    output_w_stride = output.stride(4)
    
    num_elements = output.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    conv_transpose3d_kernel[grid](
        x, weight, output, bias,
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups,
        input_b_stride, input_c_stride, input_d_stride, input_h_stride, input_w_stride,
        weight_oc_stride, weight_ic_stride, weight_d_stride, weight_h_stride, weight_w_stride,
        output_b_stride, output_c_stride, output_d_stride, output_h_stride, output_w_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.groups
        )