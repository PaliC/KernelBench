import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    elif len(value) == 3:
        return value
    raise ValueError("Value must be an int or 3-element tuple")

@triton.jit
def conv_transpose3d_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    B, IC, OC, K,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dilation_d, dilation_h, dilation_w,
    x_bs, x_ic, x_d, x_h, x_w,
    w_ic, w_oc, w_d, w_h, w_w,
    out_bs, out_oc, out_d, out_h, out_w,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    d_mask = d_offs < D_out
    h_mask = h_offs < H_out
    w_mask = w_offs < W_out
    mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    output = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    if bias_ptr != 0:
        bias = tl.load(bias_ptr + pid_oc)
        output += bias

    for ic in range(IC):
        for kd in range(K):
            for kh in range(K):
                for kw in range(K):
                    d_in = (d_offs + pad_d - kd * dilation_d) // stride_d
                    h_in = (h_offs + pad_h - kh * dilation_h) // stride_h
                    w_in = (w_offs + pad_w - kw * dilation_w) // stride_w
                    
                    valid_d = (d_in >= 0) & (d_in < D_in) & ((d_offs + pad_d - kd * dilation_d) % stride_d == 0)
                    valid_h = (h_in >= 0) & (h_in < H_in) & ((h_offs + pad_h - kh * dilation_h) % stride_h == 0)
                    valid_w = (w_in >= 0) & (w_in < W_in) & ((w_offs + pad_w - kw * dilation_w) % stride_w == 0)
                    valid = valid_d[:, None, None] & valid_h[None, :, None] & valid_w[None, None, :]

                    x_offs = pid_b * x_bs + ic * x_ic + d_in * x_d + h_in * x_h + w_in * x_w
                    x_val = tl.load(x_ptr + x_offs, mask=valid & mask, other=0.0)

                    w_off = ic * w_ic + pid_oc * w_oc + kd * w_d + kh * w_h + kw * w_w
                    w_val = tl.load(weight_ptr + w_off)

                    output += x_val * w_val

    out_offs = (pid_b * out_bs + pid_oc * out_oc 
                + d_offs[:, None, None] * out_d 
                + h_offs[None, :, None] * out_h 
                + w_offs[None, None, :] * out_w)
    tl.store(output_ptr + out_offs, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, dilation: int = 1, groups: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.dilation = _triple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, 
                                      self.output_padding, self.dilation, self.groups)

def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    assert groups == 1, "Only groups=1 supported"
    x = x.contiguous()
    weight = weight.contiguous()

    B, IC, D_in, H_in, W_in = x.shape
    OC = weight.shape[1]
    Kd, Kh, Kw = weight.shape[2:]
    Sd, Sh, Sw = stride
    Pd, Ph, Pw = padding
    Od, Oh, Ow = output_padding
    Dd, Dh, Dw = dilation

    D_out = (D_in - 1) * Sd - 2 * Pd + Dd * (Kd - 1) + Od + 1
    H_out = (H_in - 1) * Sh - 2 * Ph + Dh * (Kh - 1) + Oh + 1
    W_out = (W_in - 1) * Sw - 2 * Pw + Dw * (Kw - 1) + Ow + 1

    output = torch.empty((B, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_D, BLOCK_H, BLOCK_W = 4, 4, 4
    grid = (
        B, 
        OC,
        triton.cdiv(D_out, BLOCK_D),
        triton.cdiv(H_out, BLOCK_H),
        triton.cdiv(W_out, BLOCK_W),
    )

    conv_transpose3d_kernel[grid](
        x, weight, bias if bias is not None else 0, output,
        B, IC, OC, Kd,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Sd, Sh, Sw,
        Pd, Ph, Pw,
        Dd, Dh, Dw,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )
    return output