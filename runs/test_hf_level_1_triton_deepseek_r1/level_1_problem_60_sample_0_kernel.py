import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dilation_d, dilation_h, dilation_w,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_od = tl.program_id(2)
    pid_oh = pid_n // (N * OD)
    pid_ow = pid_n % (N * OD) // OD
    
    if pid_oc >= OC or pid_od >= OD or pid_oh >= OH or pid_ow >= OW:
        return
    
    input_d = pid_od * stride_d - pad_d
    input_h = pid_oh * stride_h - pad_h
    input_w = pid_ow * stride_w - pad_w
    
    acc = 0.0
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                id_d = input_d + kd * dilation_d
                id_h = input_h + kh * dilation_h
                id_w = input_w + kw * dilation_w
                
                if id_d >=0 and id_d < ID and id_h >=0 and id_h < IH and id_w >=0 and id_w < IW:
                    for ic in range(0, IC, BLOCK_SIZE_IC):
                        mask_ic = ic + tl.arange(0, BLOCK_SIZE_IC) < IC
                        x_off = (pid_n // N) * IC * ID * IH * IW + (ic + tl.arange(0, BLOCK_SIZE_IC)) * ID * IH * IW + id_d * IH * IW + id_h * IW + id_w
                        w_off = pid_oc * IC * KD * KH * KW + (ic + tl.arange(0, BLOCK_SIZE_IC)) * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        x = tl.load(x_ptr + x_off, mask=mask_ic, other=0.0)
                        w = tl.load(w_ptr + w_off, mask=mask_ic, other=0.0)
                        acc += tl.sum(x * w)
    
    if b_ptr is not None:
        acc += tl.load(b_ptr + pid_oc)
    
    y_off = (pid_n // N) * OC * OD * OH * OW + pid_oc * OD * OH * OW + pid_od * OH * OW + pid_oh * OW + pid_ow
    tl.store(y_ptr + y_off, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation, groups):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = (ID + 2*padding[0] - dilation[0]*(KD-1) -1) // stride[0] + 1
    OH = (IH + 2*padding[1] - dilation[1]*(KH-1) -1) // stride[1] + 1
    OW = (IW + 2*padding[2] - dilation[2]*(KW-1) -1) // stride[2] + 1
    
    y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
    grid = (N*OH*OW, OC, OD)
    
    BLOCK_SIZE_IC = 16
    conv3d_kernel[grid](
        x, weight, bias, y,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        BLOCK_SIZE_IC=BLOCK_SIZE_IC,
        BLOCK_SIZE_KD=1, BLOCK_SIZE_KH=1, BLOCK_SIZE_KW=1
    )
    return y

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride,)*3 if isinstance(stride, int) else stride
        self.padding = (padding,)*3 if isinstance(padding, int) else padding
        self.dilation = (dilation,)*3 if isinstance(dilation, int) else dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels//groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
            
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(x, self.weight, self.bias, 
                            self.stride, self.padding, 
                            self.dilation, self.groups)