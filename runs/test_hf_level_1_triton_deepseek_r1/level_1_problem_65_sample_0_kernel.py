import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    B,
    IC,
    IH,
    IW,
    OC,
    OH,
    OW,
    KH,
    KW,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    w_input_channel_stride,
    w_output_channel_stride,
    w_kernel_h_stride,
    w_kernel_w_stride,
    out_batch_stride,
    out_channel_stride,
    out_height_stride,
    out_width_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = B * OC * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Decompose offsets into 4D indices
    b = offsets // (OC * OH * OW)
    rem = offsets % (OC * OH * OW)
    oc = rem // (OH * OW)
    rem = rem % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                ih = (oh + padding_h - kh) // stride_h
                iw = (ow + padding_w - kw) // stride_w
                
                valid_h = (oh + padding_h - kh) % stride_h == 0
                valid_w = (ow + padding_w - kw) % stride_w == 0
                valid = valid_h & valid_w & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)

                x_off = b * x_batch_stride + ic * x_channel_stride + ih * x_height_stride + iw * x_width_stride
                w_off = ic * w_input_channel_stride + oc * w_output_channel_stride + kh * w_kernel_h_stride + kw * w_kernel_w_stride
                
                x_val = tl.load(x_ptr + x_off, mask=valid & mask, other=0.0)
                w_val = tl.load(w_ptr + w_off, mask=valid & mask, other=0.0)
                acc += x_val * w_val

    out_off = b * out_batch_stride + oc * out_channel_stride + oh * out_height_stride + ow * out_width_stride
    tl.store(out_ptr + out_off, acc, mask=mask)


def triton_conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups):
    assert groups == 1, "Only groups=1 supported in Triton kernel"
    B, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OC *= groups
    
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    padding_h, padding_w = (padding, padding) if isinstance(padding, int) else padding
    output_padding_h, output_padding_w = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding

    OH = (IH - 1) * stride_h - 2 * padding_h + KH + output_padding_h
    OW = (IW - 1) * stride_w - 2 * padding_w + KW + output_padding_w
    
    out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(out.numel(), meta['BLOCK_SIZE']),)
    conv_transpose2d_kernel[grid](
        x, weight, out,
        B, IC, IH, IW, OC, OH, OW, KH, KW,
        stride_h, stride_w,
        padding_h, padding_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=128
    )
    
    return out + bias.view(1, -1, 1, 1) if bias is not None else out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        self.weight = nn.Parameter(torch.randn(
            in_channels,
            out_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.output_padding, self.groups
        )