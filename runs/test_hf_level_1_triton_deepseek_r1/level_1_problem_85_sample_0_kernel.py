import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    batch_size,
    in_channels,
    input_h,
    input_w,
    output_h,
    output_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    block_idx = tl.program_id(2)

    output_elements = output_h * output_w
    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_elements

    h_out = offsets // output_w
    w_out = offsets % output_w

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            h_in = h_out * stride_h - padding_h + kh * dilation_h
            w_in = w_out * stride_w - padding_w + kw * dilation_w

            h_in_valid = (h_in >= 0) & (h_in < input_h)
            w_in_valid = (w_in >= 0) & (w_in < input_w)
            valid = h_in_valid & w_in_valid & mask

            input_offset = (
                batch_idx * in_channels * input_h * input_w +
                channel_idx * input_h * input_w +
                h_in * input_w +
                w_in
            )
            input_val = tl.load(input_ptr + input_offset, mask=valid, other=0.0)

            kernel_offset = channel_idx * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
            kernel_val = tl.load(kernel_ptr + kernel_offset)

            accumulator += input_val * kernel_val

    output_offset = (
        batch_idx * in_channels * output_h * output_w +
        channel_idx * output_h * output_w +
        h_out * output_w +
        w_out
    )
    tl.store(output_ptr + output_offset, accumulator, mask=mask)

def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, input_h, input_w = x.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]

    output_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    output_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    output = torch.empty((batch_size, in_channels, output_h, output_w), device=x.device, dtype=x.dtype)

    output_elements = output_h * output_w
    BLOCK_SIZE = 128
    num_blocks = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, in_channels, num_blocks)

    depthwise_conv2d_kernel[grid](
        x, weight, output,
        batch_size,
        in_channels,
        input_h,
        input_w,
        output_h,
        output_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        KERNEL_H=kernel_h,
        KERNEL_W=kernel_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert groups == in_channels and in_channels == out_channels

        self.in_channels = in_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x,
            self.weight,
            self.bias,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
        )