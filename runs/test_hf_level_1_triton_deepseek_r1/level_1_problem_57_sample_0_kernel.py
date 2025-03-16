import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride,
    padding,
    output_padding,
    groups,
    in_channels,
    out_channels,
    kernel_size,
    input_h,
    input_w,
    output_h,
    output_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Calculate output position indices
    batch = pid // (out_channels * output_h * output_w)
    ch_out = (pid // (output_h * output_w)) % out_channels
    h_out = (pid // output_w) % output_h
    w_out = pid % output_w
    
    group_size = out_channels // groups
    group_id = ch_out // group_size
    in_ch_per_group = in_channels // groups
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel positions
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input position considering stride and padding
            h_in = (h_out + padding - kh) // stride
            w_in = (w_out + padding - kw) // stride
            
            if (h_out + padding - kh) % stride == 0 and (w_out + padding - kw) % stride == 0:
                h_in = (h_out + padding - kh) // stride
                w_in = (w_out + padding - kw) // stride
                
                if h_in >= 0 and h_in < input_h and w_in >= 0 and w_in < input_w:
                    # Loop over input channels in group
                    for ch_in in range(in_ch_per_group):
                        x_idx = batch * in_channels * input_h * input_w + \
                                (group_id * in_ch_per_group + ch_in) * input_h * input_w + \
                                h_in * input_w + w_in
                                
                        w_idx = ch_out * in_channels // groups * kernel_size * kernel_size + \
                                ch_in * kernel_size * kernel_size + \
                                kh * kernel_size + kw
                                
                        x_val = tl.load(x_ptr + x_idx, mask=None, eviction_policy="evict_last")
                        w_val = tl.load(w_ptr + w_idx, mask=None, eviction_policy="evict_last")
                        acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        b_val = tl.load(b_ptr + ch_out)
        acc += b_val
        
    # Apply output padding
    if h_out < output_h - output_padding and w_out < output_w - output_padding:
        out_idx = batch * out_channels * output_h * output_w + \
                  ch_out * output_h * output_w + \
                  h_out * output_w + w_out
        tl.store(out_ptr + out_idx, acc, mask=None)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.weight = nn.Parameter(torch.randn(
            out_channels, 
            in_channels // groups, 
            kernel_size, 
            kernel_size
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_h, input_w = x.shape
        
        # Calculate output dimensions
        output_h = (input_h - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_w = (input_w - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        
        out = torch.empty(
            (batch_size, self.out_channels, output_h, output_w),
            device=x.device, 
            dtype=x.dtype
        )
        
        n_elements = batch_size * self.out_channels * output_h * output_w
        grid = lambda meta: (n_elements,)
        
        conv_transpose2d_kernel[grid](
            x, 
            self.weight,
            self.bias if self.bias is not None else None,
            out,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            input_h,
            input_w,
            output_h,
            output_w,
            BLOCK_SIZE=128,
        )
        
        return out