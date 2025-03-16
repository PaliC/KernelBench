import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    B, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KS, S, P, D,
    BLOCK_SIZE: tl.constexpr,
    IC_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    numel = B * OC * OD * OH * OW
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Reconstruct 5D indices
    b = offsets // (OC * OD * OH * OW)
    rem = offsets % (OC * OD * OH * OW)
    oc = rem // (OD * OH * OW)
    rem = rem % (OD * OH * OW)
    oz = rem // (OH * OW)
    rem = rem % (OH * OW)
    oy = rem // OW
    ox = rem % OW

    # Initialize output
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop through kernel
    for kz in range(KS):
        for ky in range(KS):
            for kx in range(KS):
                iz = (oz + P - kz * D) // S
                iy = (oy + P - ky * D) // S
                ix = (ox + P - kx * D) // S
                
                valid_z = (iz >= 0) & (iz < ID) & ((oz + P - kz * D) % S == 0)
                valid_y = (iy >= 0) & (iy < IH) & ((oy + P - ky * D) % S == 0)
                valid_x = (ix >= 0) & (ix < IW) & ((ox + P - kx * D) % S == 0)
                valid = valid_z & valid_y & valid_x

                # Vectorized channel processing
                for ic_block in range(0, IC, IC_BLOCK):
                    ic = ic_block + tl.arange(0, IC_BLOCK)
                    ic_mask = ic < IC

                    # Calculate input pointers
                    x_idx = b * IC * ID * IH * IW + \
                            ic[:, None] * ID * IH * IW + \
                            iz * IH * IW + iy * IW + ix
                    x_val = tl.load(x_ptr + x_idx, mask=ic_mask[:, None] & valid, other=0.0)

                    # Calculate weight pointers
                    w_idx = ic[:, None] * OC * KS**3 + \
                            oc * KS**3 + \
                            kz * KS**2 + ky * KS + kx
                    w_val = tl.load(w_ptr + w_idx, mask=ic_mask[:, None], other=0.0)

                    acc += tl.sum(x_val * w_val, axis=0)

    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + oc, mask=mask, other=0.0)
        acc += bias

    # Store result
    y_idx = b * OC * OD * OH * OW + \
            oc * OD * OH * OW + \
            oz * OH * OW + oy * OW + ox
    tl.store(y_ptr + y_idx, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels,
            kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_channels * kernel_size**3)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KS = self.kernel_size
        S, P, D = self.stride, self.padding, self.dilation

        OD = (ID - 1) * S - 2 * P + D * (KS - 1) + 1
        OH = (IH - 1) * S - 2 * P + D * (KS - 1) + 1
        OW = (IW - 1) * S - 2 * P + D * (KS - 1) + 1

        y = torch.empty((B, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (triton.cdiv(y.numel(), meta['BLOCK_SIZE']),)
        conv_transpose3d_kernel[grid](
            x, self.weight, self.bias, y,
            B, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KS, S, P, D,
            BLOCK_SIZE=128,
            IC_BLOCK=16
        )
        return y