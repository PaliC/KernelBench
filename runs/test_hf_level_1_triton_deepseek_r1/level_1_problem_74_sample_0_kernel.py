import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional

@triton.jit
def conv_transpose1d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation,
    in_channels,
    out_channels,
    input_batch_stride,
    input_channel_stride,
    input_length_stride,
    weight_in_stride,
    weight_out_stride,
    weight_kernel_stride,
    output_batch_stride,
    output_channel_stride,
    output_length_stride,
    BLOCK_LENGTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_output_block = tl.program_id(1)
    pid_group = tl.program_id(2)

    output_start = pid_output_block * BLOCK_LENGTH
    group_start = pid_group * GROUP_SIZE

    output_offsets = output_start + tl.arange(0, BLOCK_LENGTH)
    channel_offsets = group_start + tl.arange(0, GROUP_SIZE)

    output_mask = output_offsets < output_length
    channel_mask = channel_offsets < out_channels

    acc = tl.zeros((BLOCK_LENGTH, GROUP_SIZE), dtype=tl.float32)

    for k in range(kernel_size):
        input_pos = (output_offsets // stride) - padding + k * dilation
        input_pos = tl.maximum(tl.minimum(input_pos, input_length - 1), 0)
        input_pos_valid = (input_pos >= 0) & (input_pos < input_length)

        for in_c in range(in_channels):
            input_idx = pid_batch * input_batch_stride + in_c * input_channel_stride + input_pos * input_length_stride
            x = tl.load(input_ptr + input_idx, mask=input_pos_valid & (in_c < in_channels), other=0.0)

            weight_idx = in_c * weight_in_stride + channel_offsets * weight_out_stride + k * weight_kernel_stride
            w = tl.load(weight_ptr + weight_idx, mask=channel_mask, other=0.0)

            acc += x[:, None] * w[None, :]

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0)
        acc += bias[None, :]

    output_indices = (
        pid_batch * output_batch_stride +
        channel_offsets[None, :] * output_channel_stride +
        output_offsets[:, None] * output_length_stride
    )
    tl.store(output_ptr + output_indices, acc, mask=output_mask[:, None] & channel_mask[None, :])

def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: int,
    dilation: int
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda)
    
    batch_size, in_channels, input_length = x.shape
    out_channels = weight.shape[1]
    kernel_size = weight.shape[2]

    output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    x = x.contiguous()
    weight = weight.contiguous()
    output = torch.empty((batch_size, out_channels, output_length), device=x.device, dtype=x.dtype)

    BLOCK_LENGTH = 128
    GROUP_SIZE = 32
    grid = (
        batch_size,
        (output_length + BLOCK_LENGTH - 1) // BLOCK_LENGTH,
        (out_channels + GROUP_SIZE - 1) // GROUP_SIZE,
    )

    conv_transpose1d_kernel[grid](
        x, weight, bias, output,
        input_length, output_length,
        kernel_size, stride, padding, dilation,
        in_channels, out_channels,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_LENGTH=BLOCK_LENGTH, GROUP_SIZE=GROUP_SIZE
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
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
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )