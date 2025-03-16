import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, H, W,
    O, K, OH, OW,
    stride, padding, dilation,
    groups, use_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * O * OH * OW

    # Decompose index
    n = tl.math.divide(offsets, O * OH * OW, mask=mask)
    remainder = offsets - n * O * OH * OW
    o = tl.math.divide(remainder, OH * OW, mask=mask)
    remainder = remainder - o * OH * OW
    oh = tl.math.divide(remainder, OW, mask=mask)
    ow = remainder - oh * OW

    # Group calculations
    C_per_group = C // groups
    O_per_group = O // groups
    g = o // O_per_group
    c_start = g * C_per_group
    c_end = (g + 1) * C_per_group

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop through kernel and channels
    for c in range(c_start, c_end):
        for kh in range(K):
            for kw in range(K):
                h_in = oh * stride + kh * dilation - padding
                w_in = ow * stride + kw * dilation - padding
                if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                    input_off = n * C * H * W + c * H * W + h_in * W + w_in
                    weight_off = o * C_per_group * K * K + (c - c_start) * K * K + kh * K + kw
                    input_val = tl.load(input_ptr + input_off, mask=mask, other=0.0)
                    weight_val = tl.load(weight_ptr + weight_off, mask=mask, other=0.0)
                    acc += input_val * weight_val

    if use_bias:
        bias_val = tl.load(bias_ptr + o, mask=mask, other=0.0)
        acc += bias_val

    output_off = n * O * OH * OW + o * OH * OW + oh * OW + ow
    tl.store(output_ptr + output_off, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            bound = 1 / math.sqrt(in_channels // groups * kernel_size ** 2)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        N, C, H, W = x.shape
        K = self.kernel_size
        OH = (H + 2*self.padding - self.dilation*(K-1) - 1) // self.stride + 1
        OW = (W + 2*self.padding - self.dilation*(K-1) - 1) // self.stride + 1
        output = torch.empty(N, self.out_channels, OH, OW, device=x.device, dtype=x.dtype)

        BLOCK_SIZE = 128
        grid = lambda meta: ((N * self.out_channels * OH * OW + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
        
        conv2d_kernel[grid](
            x, self.weight, self.bias if self.use_bias else None, output,
            N, C, H, W,
            self.out_channels, self.kernel_size, OH, OW,
            self.stride, self.padding, self.dilation,
            self.groups, self.use_bias,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output.reshape(N, self.out_channels, OH, OW)