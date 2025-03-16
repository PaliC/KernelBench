import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    in_channels,
    D_in,
    H_in,
    W_in,
    out_channels,
    D_out,
    H_out,
    W_out,
    K,
    stride,
    padding,
    output_padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = N * out_channels * D_out * H_out * W_out
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_idx < num_elements
    element_idx = tl.where(mask, element_idx, 0)

    # Decompose element index into output tensor coordinates
    oc_per_group = out_channels // groups
    elements_per_channel = D_out * H_out * W_out
    elements_per_batch = out_channels * elements_per_channel

    n = element_idx // elements_per_batch
    remainder = element_idx % elements_per_batch
    oc = remainder // elements_per_channel
    remainder = remainder % elements_per_channel
    d = remainder // (H_out * W_out)
    remainder = remainder % (H_out * W_out)
    h = remainder // W_out
    w = remainder % W_out

    group = oc // oc_per_group
    oc_in_group = oc % oc_per_group
    in_per_group = in_channels // groups
    ic_start = group * in_per_group

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kd in range(K):
        for kh in range(K):
            for kw in range(K):
                d_in = (d + padding - kd) // stride
                h_in = (h + padding - kh) // stride
                w_in = (w + padding - kw) // stride

                valid_d = (d + padding - kd) % stride == 0
                valid_h = (h + padding - kh) % stride == 0
                valid_w = (w + padding - kw) % stride == 0
                in_bounds = (d_in >= 0) & (d_in < D_in) & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                valid = valid_d & valid_h & valid_w & in_bounds

                for ic in range(in_per_group):
                    x_off = n * in_channels * D_in * H_in * W_in + (ic_start + ic) * D_in * H_in * W_in + d_in * H_in * W_in + h_in * W_in + w_in
                    w_off = (ic_start + ic) * (oc_per_group * K * K * K) + oc_in_group * K * K * K + kd * K * K + kh * K + kw
                    
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    w_val = tl.load(weight_ptr + w_off)
                    acc += tl.where(valid, x_val * w_val, 0.0)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc)
        acc += bias

    out_off = n * out_channels * D_out * H_out * W_out + oc * D_out * H_out * W_out + d * H_out * W_out + h * W_out + w
    tl.store(out_ptr + out_off, acc, mask=mask)

def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: int,
    output_padding: int,
    groups: int
) -> torch.Tensor:
    N, in_channels, D_in, H_in, W_in = x.shape
    K = weight.shape[2]
    out_channels = weight.shape[1] * groups

    D_out = (D_in - 1) * stride - 2 * padding + K + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + K + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K + output_padding

    out = torch.empty((N, out_channels, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    num_elements = out.numel()

    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, in_channels, D_in, H_in, W_in,
        out_channels, D_out, H_out, W_out,
        K, stride, padding, output_padding, groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.output_padding, self.groups
        )