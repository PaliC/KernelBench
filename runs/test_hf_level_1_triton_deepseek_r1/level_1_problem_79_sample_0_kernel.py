import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    B,
    InC,
    OutC,
    L_in,
    K,
    stride,
    padding,
    dilation,
    L_out,
    input_batch_stride,
    input_in_c_stride,
    input_l_stride,
    weight_out_c_stride,
    weight_in_c_stride,
    weight_k_stride,
    output_batch_stride,
    output_out_c_stride,
    output_l_stride,
    BLOCK_SIZE_L: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    l_idx = tl.program_id(2) * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    mask = (l_idx < L_out) & (batch_idx < B) & (out_c_idx < OutC)
    
    acc = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)
    
    for k in range(K):
        input_l = (l_idx + padding - k * dilation) // stride
        valid = (input_l >= 0) & (input_l < L_in) & ((l_idx + padding - k * dilation) % stride == 0)
        
        for in_c in range(InC):
            weight_offset = out_c_idx * weight_out_c_stride + in_c * weight_in_c_stride + k * weight_k_stride
            w = tl.load(weight_ptr + weight_offset)
            
            input_offset = batch_idx * input_batch_stride + in_c * input_in_c_stride + input_l * input_l_stride
            x = tl.load(input_ptr + input_offset, mask=valid, other=0.0)
            
            acc += x * w
    
    output_offset = batch_idx * output_batch_stride + out_c_idx * output_out_c_stride + l_idx * output_l_stride
    tl.store(output_ptr + output_offset, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, InC, L_in = x.shape
        L_out = (L_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        
        x = x.contiguous()
        output = torch.empty((B, self.out_channels, L_out), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_L = 128
        grid = (
            B,
            self.out_channels,
            (L_out + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L,
        )
        
        conv_transpose1d_kernel[grid](
            x, self.weight, output,
            B, self.in_channels, self.out_channels, L_in, self.kernel_size,
            self.stride, self.padding, self.dilation, L_out,
            x.stride(0), x.stride(1), x.stride(2),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE_L=BLOCK_SIZE_L,
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)
            
        return output