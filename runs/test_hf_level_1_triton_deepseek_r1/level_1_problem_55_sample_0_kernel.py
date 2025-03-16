import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    b_ptr,
    B,
    C_in,
    H,
    W,
    C_out,
    K,
    S,
    P,
    D,
    G,
    H_out,
    W_out,
    has_bias,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Calculate 4D indices
    offset = offsets
    b_idx = offset // (C_out * H_out * W_out)
    rem = offset % (C_out * H_out * W_out)
    c_out = rem // (H_out * W_out)
    rem = rem % (H_out * W_out)
    h_out = rem // W_out
    w_out = rem % W_out

    # Group calculations
    C_out_g = C_out // G
    g = c_out // C_out_g
    C_in_g = C_in // G
    start_c_in = g * C_in_g

    # Convolution loop
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kh in range(K):
        for kw in range(K):
            h_in = h_out * S - P + kh * D
            w_in = w_out * S - P + kw * D
            
            h_valid = (h_in >= 0) & (h_in < H)
            w_valid = (w_in >= 0) & (w_in < W)
            valid_mask = h_valid & w_valid
            
            for c_in_offset in range(C_in_g):
                x_idx = (b_idx * C_in + start_c_in + c_in_offset) * H * W + h_in * W + w_in
                w_idx = c_out * (C_in_g * K * K) + c_in_offset * (K * K) + kh * K + kw
                
                x_val = tl.load(x_ptr + x_idx, mask=valid_mask & mask, other=0.0)
                w_val = tl.load(w_ptr + w_idx, mask=mask, other=0.0)
                acc += x_val * w_val

    # Add bias
    if has_bias:
        bias = tl.load(b_ptr + c_out, mask=mask, other=0.0)
        acc += bias

    # Store output
    out_idx = offsets
    tl.store(out_ptr + out_idx, acc, mask=mask)

def triton_conv2d(x, weight, bias, stride, padding, dilation, groups):
    B, C_in, H, W = x.shape
    C_out, C_in_g, K, _ = weight.shape
    G = groups
    H_out = (H + 2*padding - dilation*(K-1) - 1) // stride + 1
    W_out = (W + 2*padding - dilation*(K-1) - 1) // stride + 1
    total_elements = B * C_out * H_out * W_out

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    conv2d_kernel[grid](
        x, weight, out, bias,
        B, C_in, H, W, C_out, K,
        stride, padding, dilation, G,
        H_out, W_out, bias is not None,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_conv2d(
            x, self.weight, self.bias, self.stride, 
            self.padding, self.dilation, self.groups
        )