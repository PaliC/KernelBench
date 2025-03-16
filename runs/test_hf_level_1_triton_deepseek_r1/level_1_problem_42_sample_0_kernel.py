import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    input_ptr,
    output_ptr,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    input_h,
    input_w,
    output_h,
    output_w,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Create block of output indices
    block = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + block
    
    # Calculate 4D coordinates (b, c, oh, ow)
    b = idx // (channels * output_h * output_w)
    remainder = idx % (channels * output_h * output_w)
    c = remainder // (output_h * output_w)
    remainder = remainder % (output_h * output_w)
    oh = remainder // output_w
    ow = remainder % output_w

    valid = idx < channels * output_h * output_w * b.shape[0]
    
    # Calculate input window positions with dilation and padding
    ih_start = oh * stride_h - pad_h
    iw_start = ow * stride_w - pad_w
    
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), tl.float32)
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = ih_start + kh * dilation_h
            iw = iw_start + kw * dilation_w
            
            # Check if input indices are within bounds
            in_bounds = (ih >= 0) & (ih < input_h) & (iw >= 0) & (iw < input_w)
            cond = valid & in_bounds
            
            # Calculate input pointer offsets
            input_offset = (b * channels * input_h * input_w + 
                            c * input_h * input_w + 
                            ih * input_w + 
                            iw)
            
            val = tl.load(input_ptr + input_offset, mask=cond, other=-float('inf'))
            max_val = tl.maximum(max_val, val)
    
    # Store results
    output_offset = (b * channels * output_h * output_w + 
                     c * output_h * output_w + 
                     oh * output_w + 
                     ow)
    tl.store(output_ptr + output_offset, max_val, mask=valid)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size, channels, in_h, in_w = x.shape
        
        # Calculate output dimensions
        out_h = (in_h + 2*self.padding - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
        out_w = (in_w + 2*self.padding - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
        
        output = torch.empty((batch_size, channels, out_h, out_w), device=x.device)
        
        # Flatten spatial dimensions for kernel
        n_elements = batch_size * channels * out_h * out_w
        BLOCK_SIZE = 128
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        maxpool2d_kernel[grid](
            x, output,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.padding, self.padding,
            self.dilation, self.dilation,
            in_h, in_w,
            out_h, out_w,
            channels,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output