import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    B,
    C_in,
    C_out,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    output_pad_d,
    output_pad_h,
    output_pad_w,
    input_bs, input_cs, input_ds, input_hs, input_ws,
    weight_cs, weight_ocs, weight_ds, weight_hs, weight_ws,
    output_bs, output_cs, output_ds, output_hs, output_ws,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(B * C_out * D_out * H_out * W_out, BLOCK_SIZE)
    if pid >= num_pid:
        return

    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < B * C_out * D_out * H_out * W_out

    # Reconstruct 5D indices
    w_out = off % W_out
    off = off // W_out
    h_out = off % H_out
    off = off // H_out
    d_out = off % D_out
    off = off // D_out
    c_out = off % C_out
    b = off // C_out

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop through input channels and kernel dimensions
    for c_in in range(C_in):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate input positions
                    d_in = (d_out + pad_d - kd) // stride_d
                    w_in = (w_out + pad_w - kw) // stride_w
                    h_in = (h_out + pad_h - kh) // stride_h

                    # Check if in bounds and valid
                    d_valid = (d_out + pad_d - kd) % stride_d == 0
                    w_valid = (w_out + pad_w - kw) % stride_w == 0
                    h_valid = (h_out + pad_h - kh) % stride_h == 0
                    in_bounds = (d_in >= 0) & (d_in < D_in) & (w_in >= 0) & (w_in < W_in) & (h_in >= 0) & (h_in < H_in)
                    
                    if d_valid & w_valid & h_valid & in_bounds:
                        # Load input
                        input_idx = b * input_bs + c_in * input_cs + d_in * input_ds + h_in * input_hs + w_in * input_ws
                        i_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
                        
                        # Load weight
                        weight_idx = c_in * weight_cs + c_out * weight_ocs + kd * weight_ds + kh * weight_hs + kw * weight_ws
                        w_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
                        
                        acc += i_val * w_val

    # Write output
    output_idx = b * output_bs + c_out * output_cs + d_out * output_ds + h_out * output_hs + w_out * output_ws
    tl.store(output_ptr + output_idx, acc, mask=mask)

def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    B, C_in, D_in, H_in, W_in = x.shape
    C_out = weight.shape[1] * groups
    kernel_d, kernel_h, kernel_w = weight.shape[2:]
    
    # Calculate output shape
    D_out = (D_in - 1) * stride[0] - 2 * padding[0] + kernel_d + output_padding[0]
    H_out = (H_in - 1) * stride[1] - 2 * padding[1] + kernel_h + output_padding[1]
    W_out = (W_in - 1) * stride[2] - 2 * padding[2] + kernel_w + output_padding[2]
    
    x = x.contiguous()
    weight = weight.contiguous()
    output = torch.empty((B, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(B * C_out * D_out * H_out * W_out, meta['BLOCK_SIZE']),)
    
    conv_transpose3d_kernel[grid](
        x, weight, output,
        B, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_SIZE=128
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
        
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
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
            *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_conv_transpose3d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            self.groups
        )