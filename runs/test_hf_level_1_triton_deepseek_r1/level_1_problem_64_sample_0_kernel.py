import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    length_in,
    length_out,
    batch_size,
    x_batch_stride,
    x_in_chan_stride,
    x_len_stride,
    w_in_chan_stride,
    w_out_chan_stride,
    w_kernel_stride,
    out_batch_stride,
    out_out_chan_stride,
    out_len_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_input_elements = batch_size * in_channels * length_in
    if pid >= num_input_elements:
        return

    batch_idx = pid // (in_channels * length_in)
    remainder = pid % (in_channels * length_in)
    in_chan_idx = remainder // length_in
    input_pos = remainder % length_in

    group_size_in = in_channels // groups
    group_idx = in_chan_idx // group_size_in
    group_size_out = out_channels // groups
    out_chan_start = group_idx * group_size_out

    x_offset = (batch_idx * x_batch_stride + 
               in_chan_idx * x_in_chan_stride + 
               input_pos * x_len_stride)
    x_val = tl.load(x_ptr + x_offset)

    for k in range(kernel_size):
        output_pos = input_pos * stride + k - padding
        if output_pos < 0 or output_pos >= length_out:
            continue

        for out_chan_idx in range(group_size_out):
            out_chan = out_chan_start + out_chan_idx
            w_offset = (in_chan_idx * w_in_chan_stride + 
                       out_chan_idx * w_out_chan_stride + 
                       k * w_kernel_stride)
            w_val = tl.load(w_ptr + w_offset)
            contribution = x_val * w_val

            out_offset = (batch_idx * out_batch_stride +
                         out_chan * out_out_chan_stride +
                         output_pos * out_len_stride)
            tl.atomic_add(out_ptr + out_offset, contribution)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, kernel_size))
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
        length_in = x.size(2)
        length_out = ((length_in - 1) * self.stride - 2 * self.padding 
                      + self.kernel_size + self.output_padding)

        x = x.contiguous()
        weight = self.weight.contiguous()
        batch_size = x.size(0)

        out = torch.zeros(batch_size, self.out_channels, length_out,
                         device=x.device, dtype=x.dtype)

        x_b_stride, x_c_stride, x_l_stride = x.stride()
        w_c_stride, w_oc_stride, w_k_stride = weight.stride()
        out_b_stride, out_c_stride, out_l_stride = out.stride()

        num_input_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_input_elements, meta['BLOCK_SIZE']),)

        conv_transpose1d_kernel[grid](
            x, weight, out,
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.groups,
            length_in, length_out, batch_size,
            x_b_stride, x_c_stride, x_l_stride,
            w_c_stride, w_oc_stride, w_k_stride,
            out_b_stride, out_c_stride, out_l_stride,
            BLOCK_SIZE=1024,
        )

        if self.bias is not None:
            out += self.bias.view(1, -1, 1)

        return out