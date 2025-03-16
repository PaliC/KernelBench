import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    B, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = B * OC * OD * OH * OW
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Convert linear index to B, OC, OD, OH, OW
    b = offsets // (OC * OD * OH * OW)
    remaining = offsets % (OC * OD * OH * OW)
    oc = remaining // (OD * OH * OW)
    remaining = remaining % (OD * OH * OW)
    od = remaining // (OH * OW)
    remaining = remaining % (OH * OW)
    oh = remaining // OW
    ow = remaining % OW

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for ic in range(IC):
        for kd in range(KD):
            for kh in range(KH):
                for kw in range(KW):
                    d_in = od * stride_d - padding_d + kd * dilation_d
                    h_in = oh * stride_h - padding_h + kh * dilation_h
                    w_in = ow * stride_w - padding_w + kw * dilation_w

                    valid = (d_in >= 0) & (d_in < D) & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                    
                    input_idx = b * IC * D * H * W + ic * D * H * W + d_in * H * W + h_in * W + w_in
                    weight_idx = oc * IC * KD * KH * KW + ic * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    
                    input_val = tl.load(input_ptr + input_idx, mask=valid & mask, other=0.0)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
                    
                    accumulator += input_val * weight_val

    tl.store(output_ptr + offsets, accumulator, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        assert groups == 1, "Groups not supported in Triton kernel"
        assert not bias, "Bias not supported in Triton kernel"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        
        OD = (D + 2*self.padding[0] - self.dilation[0]*(KD-1) - 1) // self.stride[0] + 1
        OH = (H + 2*self.padding[1] - self.dilation[1]*(KH-1) - 1) // self.stride[1] + 1
        OW = (W + 2*self.padding[2] - self.dilation[2]*(KW-1) - 1) // self.stride[2] + 1
        
        x = x.contiguous()
        output = torch.empty((B, self.out_channels, OD, OH, OW), device=x.device)
        n_elements = output.numel()
        
        grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        conv3d_kernel[grid](
            x, self.weight, output,
            B, self.in_channels, D, H, W,
            self.out_channels, KD, KH, KW,
            OD, OH, OW,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2],
            BLOCK_SIZE=128
        )
        return output