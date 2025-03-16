import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B,
    C_in,
    H,
    W,
    C_out,
    oH,
    oW,
    kH,
    kW,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    C_in_per_group,
    C_out_per_group,
    x_b_stride,
    x_c_stride,
    x_h_stride,
    x_w_stride,
    weight_c_in_stride,
    weight_c_out_stride,
    weight_h_stride,
    weight_w_stride,
    out_b_stride,
    out_c_stride,
    out_h_stride,
    out_w_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    total_elements = B * C_out * oH * oW
    elements_per_program = (total_elements + num_pid - 1) // num_pid
    start = pid * elements_per_program
    end = tl.minimum(start + elements_per_program, total_elements)

    for idx in range(start, end):
        b = idx // (C_out * oH * oW)
        remainder = idx % (C_out * oH * oW)
        oc = remainder // (oH * oW)
        remainder = remainder % (oH * oW)
        oh = remainder // oW
        ow = remainder % oW

        group_idx = oc // C_out_per_group
        c_in_start = group_idx * C_in_per_group
        c_in_end = c_in_start + C_in_per_group

        acc = 0.0
        for kh in range(kH):
            dilated_kh = kh * dilation_h
            for kw in range(kW):
                dilated_kw = kw * dilation_w
                ih = oh + padding_h - dilated_kh
                iw = ow + padding_w - dilated_kw

                if ih < 0 or ih % stride_h != 0:
                    continue
                if iw < 0 or iw % stride_w != 0:
                    continue

                ih_adj = ih // stride_h
                iw_adj = iw // stride_w

                if ih_adj >= H or iw_adj >= W:
                    continue

                for c_in in range(c_in_start, c_in_end):
                    x_offset = (
                        b * x_b_stride +
                        c_in * x_c_stride +
                        ih_adj * x_h_stride +
                        iw_adj * x_w_stride
                    )
                    x_val = tl.load(x_ptr + x_offset)

                    weight_c_out = oc % C_out_per_group
                    weight_offset = (
                        c_in * weight_c_in_stride +
                        weight_c_out * weight_c_out_stride +
                        kh * weight_h_stride +
                        kw * weight_w_stride
                    )
                    w_val = tl.load(weight_ptr + weight_offset)

                    acc += x_val * w_val

        if bias_ptr is not None:
            bias_offset = oc
            bias_val = tl.load(bias_ptr + bias_offset)
            acc += bias_val

        out_offset = (
            b * out_b_stride +
            oc * out_c_stride +
            oh * out_h_stride +
            ow * out_w_stride
        )
        tl.store(out_ptr + out_offset, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert in_channels % groups == 0 and out_channels % groups == 0
        self.C_in_per_group = in_channels // groups
        self.C_out_per_group = out_channels // groups

        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        B, C_in, H, W = x.shape
        kH, kW = self.kernel_size
        C_out = self.out_channels

        oH = (H - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (kH - 1) + 1
        oW = (W - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (kW - 1) + 1
        oH, oW = int(oH), int(oW)

        out = torch.empty((B, C_out, oH, oW), device=x.device, dtype=x.dtype)
        if x.is_cuda:
            grid = lambda meta: (triton.cdiv(out.numel(), meta['BLOCK_SIZE']),)
            conv_transpose2d_kernel[grid](
                x, self.weight, self.bias, out,
                B, C_in, H, W,
                C_out, oH, oW,
                kH, kW,
                self.stride[0], self.stride[1],
                self.padding[0], self.padding[1],
                self.dilation[0], self.dilation[1],
                self.groups,
                self.C_in_per_group,
                self.C_out_per_group,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), self.weight.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                BLOCK_SIZE=128,
            )
            return out
        else:
            return nn.functional.conv_transpose2d(
                x, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )