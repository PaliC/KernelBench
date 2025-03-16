import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B, C, K, L, S,
    stride,
    padding,
    dilation,
    L_out,
    x_batch_stride, x_channel_stride, x_length_stride,
    w_out_stride, w_in_stride, w_kernel_stride,
    out_batch_stride, out_channel_stride, out_length_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    
    for idx in range(pid, B * K * L_out, n_programs):
        b = idx // (K * L_out)
        remainder = idx % (K * L_out)
        k = remainder // L_out
        l_out = remainder % L_out
        
        l_in_start = l_out * stride - padding
        acc = 0.0
        
        for c_block in range(0, C, BLOCK_C):
            c = c_block + tl.arange(0, BLOCK_C)
            c_mask = c < C
            
            for s_block in range(0, S, BLOCK_S):
                s = s_block + tl.arange(0, BLOCK_S)
                s_mask = s < S
                
                l_in = l_in_start + s * dilation
                valid = (l_in >= 0) & (l_in < L)
                
                x_offsets = (b * x_batch_stride + c[:, None] * x_channel_stride + 
                            l_in[None, :] * x_length_stride)
                w_offsets = (k * w_out_stride + c[:, None] * w_in_stride + 
                            s[None, :] * w_kernel_stride)
                
                mask = c_mask[:, None] & s_mask[None, :] & valid[None, :]
                x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
                w = tl.load(w_ptr + w_offsets, mask=mask, other=0.0)
                
                acc += tl.sum(x * w)
        
        if HAS_BIAS:
            acc += tl.load(b_ptr + k)
        
        out_offset = (b * out_batch_stride + 
                     k * out_channel_stride + 
                     l_out * out_length_stride)
        tl.store(out_ptr + out_offset, acc)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        assert groups == 1, "Groups >1 not supported in Triton kernel"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        B, C, L = x.shape
        K, S = self.out_channels, self.kernel_size
        
        L_out = (L + 2*self.padding - self.dilation*(S-1) - 1) // self.stride + 1
        x = x.contiguous()
        out = torch.empty((B, K, L_out), device=x.device, dtype=x.dtype)
        
        has_bias = self.bias is not None
        bias_ptr = self.bias.data_ptr() if has_bias else 0
        
        def grid(meta): return (triton.cdiv(B * K * L_out, 256) * 256,)
        
        conv1d_kernel[grid](
            x, self.weight, self.bias, out,
            B, C, K, L, S,
            self.stride, self.padding, self.dilation, L_out,
            x.stride(0), x.stride(1), x.stride(2),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            HAS_BIAS=has_bias,
            BLOCK_C=16,
            BLOCK_S=4
        )
        
        return out