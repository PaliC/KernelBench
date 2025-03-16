import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional, Tuple

@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N, C, H, W,
    K, KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    OH, OW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    n = offsets // (K * OH * OW)
    remainder = offsets % (K * OH * OW)
    k = remainder // (OH * OW)
    remainder = remainder % (OH * OW)
    oh = remainder // OW
    ow = remainder % OW

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for c in range(C):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * stride_h - pad_h + kh * dilation_h
                iw = ow * stride_w - pad_w + kw * dilation_w
                
                within_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_off = n * C * H * W + c * H * W + ih * W + iw
                x_val = tl.load(x_ptr + x_off, mask=within_bounds & mask, other=0.0)
                
                w_off = k * C * KH * KW + c * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask, other=0.0)
                
                acc += x_val * w_val

    if b_ptr is not None:
        b_off = k
        b_val = tl.load(b_ptr + b_off, mask=mask, other=0.0)
        acc += b_val

    out_off = n * K * OH * OW + k * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=mask)

def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C, H, W = x.shape
    K, _, KH, KW = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1

    out = torch.empty((N, K, OH, OW), device=x.device, dtype=x.dtype)
    n_elements = N * K * OH * OW
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    conv2d_kernel[grid](
        x, weight, bias, out,
        N, C, H, W,
        K, KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        OH, OW,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
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
        return triton_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )