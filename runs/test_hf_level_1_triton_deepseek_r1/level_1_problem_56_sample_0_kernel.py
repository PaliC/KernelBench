import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_H, K_W,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
    w_outc_stride, w_inc_stride, w_h_stride, w_w_stride,
    y_batch_stride, y_channel_stride, y_h_stride, y_w_stride,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow_base = tl.program_id(3) * BLOCK_SIZE_W

    group_size = C_in // groups
    g = pid_c // (C_out // groups)
    c_in_start = g * group_size
    c_in_end = (g + 1) * group_size

    y_offsets = pid_b * y_batch_stride + pid_c * y_channel_stride + pid_oh * y_h_stride + pid_ow_base * y_w_stride
    y_ptrs = y_offsets + tl.arange(0, BLOCK_SIZE_W) * y_w_stride

    oh = pid_oh
    ow = pid_ow_base + tl.arange(0, BLOCK_SIZE_W)
    ih = oh * stride_h - pad_h
    iw = ow * stride_w - pad_w

    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)

    for kh in range(K_H):
        for kw in range(K_W):
            h_in = ih + kh * dilation_h
            w_in = iw + kw * dilation_w
            mask_h = (h_in >= 0) & (h_in < H_in)
            mask_w = (w_in >= 0) & (w_in < W_in)
            valid = mask_h & mask_w

            for c in tl.range(c_in_start, c_in_end, num_stages=3):
                w_offsets = pid_c * w_outc_stride + c * w_inc_stride + kh * w_h_stride + kw * w_w_stride
                w = tl.load(w_ptr + w_offsets)

                x_offsets = pid_b * x_batch_stride + c * x_channel_stride + h_in * x_h_stride + w_in * x_w_stride
                x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0)
                acc += x * w

    if b_ptr is not None:
        b = tl.load(b_ptr + pid_c)
        acc += b

    tl.store(y_ptrs, acc.to(tl.float16), mask=ow < W_out)

def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    B, C_in, H_in, W_in = x.shape
    C_out, _, K_H, K_W = weight.shape

    H_out = (H_in + 2 * padding[0] - dilation[0] * (K_H - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (K_W - 1) - 1) // stride[1] + 1

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_H = 1
    BLOCK_SIZE_W = 64
    grid = (B, C_out, H_out, triton.cdiv(W_out, BLOCK_SIZE_W))
    
    conv2d_kernel[grid](
        x, weight, bias, y,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_H, K_W,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return y

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), 
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )