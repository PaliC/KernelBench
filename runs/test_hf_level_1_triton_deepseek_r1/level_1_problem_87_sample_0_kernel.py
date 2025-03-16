import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    B, C_in, C_out, H, W,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    elements_per_program = tl.cdiv(B * C_out * H * W, num_pid)
    start = pid * elements_per_program
    offsets = start + tl.arange(0, elements_per_program)
    mask = offsets < B * C_out * H * W

    # Convert linear index to 4D coordinates
    w = offsets % W
    offsets = tl.div(offsets, W)
    h = offsets % H
    offsets = tl.div(offsets, H)
    c_out = offsets % C_out
    b = tl.div(offsets, C_out)

    acc = tl.zeros((1,), dtype=tl.float32)
    
    for c_in in range(0, C_in, BLOCK_SIZE_C):
        c_offs = c_in + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_offs < C_in
        
        # Load input [B, C_in, H, W]
        x_offs = b * stride_xb + c_offs * stride_xc + h * stride_xh + w
        x_val = tl.load(x_ptr + x_offs, mask=c_mask & mask, other=0.0)
        
        # Load weights [C_out, C_in]
        w_offs = c_out * stride_woc + c_offs * stride_wic
        w_val = tl.load(w_ptr + w_offs, mask=c_mask, other=0.0)
        
        acc += tl.sum(x_val * w_val)

    # Store output [B, C_out, H, W]
    y_offs = b * stride_yb + c_out * stride_yc + h * stride_yh + w
    tl.store(y_ptr + y_offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, C_in, H, W = x.shape
        y = torch.empty((B, self.out_channels, H, W), device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (triton.cdiv(B * self.out_channels * H * W, meta['BLOCK_SIZE_C']),)
        
        pointwise_conv_kernel[grid](
            x, self.weight, y,
            B, C_in, self.out_channels, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            BLOCK_SIZE_C=128
        )
        
        if self.bias is not None:
            y += self.bias.view(1, -1, 1, 1)
            
        return y