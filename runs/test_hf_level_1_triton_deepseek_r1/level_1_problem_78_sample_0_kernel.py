import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    H_in,
    W_in,
    H_out,
    W_out,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_in_channel_stride,
    weight_out_channel_stride,
    weight_kernel_h_stride,
    weight_kernel_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_output_elements = batch_size * out_channels * H_out * W_out
    if pid >= num_output_elements:
        return

    b = pid // (out_channels * H_out * W_out)
    remainder = pid % (out_channels * H_out * W_out)
    oc = remainder // (H_out * W_out)
    remainder = remainder % (H_out * W_out)
    oh = remainder // W_out
    ow = remainder % W_out

    accumulator = tl.zeros((1,), dtype=tl.float32)

    for ic in range(in_channels):
        for kh in tl.static_range(KERNEL_H):
            for kw in tl.static_range(KERNEL_W):
                ih = (oh + padding_h - kh) // stride_h
                iw = (ow + padding_w - kw) // stride_w

                if (oh + padding_h - kh) % stride_h != 0:
                    continue
                if (ow + padding_w - kw) % stride_w != 0:
                    continue

                if ih >= 0 and ih < H_in and iw >= 0 and iw < W_in:
                    input_offset = (b * input_batch_stride +
                                    ic * input_channel_stride +
                                    ih * input_height_stride +
                                    iw * input_width_stride)
                    weight_offset = (ic * weight_in_channel_stride +
                                     oc * weight_out_channel_stride +
                                     kh * weight_kernel_h_stride +
                                     kw * weight_kernel_w_stride)

                    input_val = tl.load(input_ptr + input_offset)
                    weight_val = tl.load(weight_ptr + weight_offset)
                    accumulator += input_val * weight_val

    output_offset = (b * output_batch_stride +
                     oc * output_channel_stride +
                     oh * output_height_stride +
                     ow * output_width_stride)
    tl.store(output_ptr + output_offset, accumulator)


def triton_conv_transpose2d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], stride: tuple, padding: tuple) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    batch_size, in_channels, H_in, W_in = x.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_channels = weight.shape[1]
    stride_h, stride_w = stride
    padding_h, padding_w = padding

    H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h
    W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_w

    output = torch.empty((batch_size, out_channels, H_out, W_out), device=x.device, dtype=x.dtype)

    input_batch_stride = x.stride(0)
    input_channel_stride = x.stride(1)
    input_height_stride = x.stride(2)
    input_width_stride = x.stride(3)

    weight_in_channel_stride = weight.stride(0)
    weight_out_channel_stride = weight.stride(1)
    weight_kernel_h_stride = weight.stride(2)
    weight_kernel_w_stride = weight.stride(3)

    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_height_stride = output.stride(2)
    output_width_stride = output.stride(3)

    num_output_elements = batch_size * out_channels * H_out * W_out
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(num_output_elements, meta['BLOCK_SIZE']),)

    conv_transpose2d_kernel[grid](
        x, weight, output,
        batch_size, in_channels, out_channels,
        H_in, W_in, H_out, W_out,
        stride_h, stride_w,
        padding_h, padding_w,
        input_batch_stride,
        input_channel_stride,
        input_height_stride,
        input_width_stride,
        weight_in_channel_stride,
        weight_out_channel_stride,
        weight_kernel_h_stride,
        weight_kernel_w_stride,
        output_batch_stride,
        output_channel_stride,
        output_height_stride,
        output_width_stride,
        KERNEL_H=kernel_h,
        KERNEL_W=kernel_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
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
        return triton_conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding)