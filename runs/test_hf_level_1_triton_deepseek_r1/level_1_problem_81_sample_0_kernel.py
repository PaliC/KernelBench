import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Tensor dimensions
    batch_size, in_channels, out_channels,
    h_in, w_in,
    kernel_size,
    stride,
    padding,
    dilation,
    h_out, w_out,
    # Memory strides
    input_bs_stride, input_ic_stride, input_h_stride, input_w_stride,
    weight_ic_stride, weight_oc_stride, weight_kh_stride, weight_kw_stride,
    output_bs_stride, output_oc_stride, output_h_stride, output_w_stride,
    # Kernel config
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    elements_per_program = tl.cdiv(batch_size * out_channels * h_out * w_out, num_pids)
    start = pid * elements_per_program
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * out_channels * h_out * w_out
    
    # Convert flat offset to 4D index
    b = offsets // (out_channels * h_out * w_out)
    oc = (offsets % (out_channels * h_out * w_out)) // (h_out * w_out)
    ohw = offsets % (h_out * w_out)
    oh = ohw // w_out
    ow = ohw % w_out
    
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop through input channels and kernel
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input positions
                ih = (oh - kh*dilation + padding) // stride
                iw = (ow - kw*dilation + padding) // stride
                
                # Check boundaries and stride divisibility
                valid = ((oh - kh*dilation + padding) % stride == 0) & \
                        (ih >= 0) & (ih < h_in) & \
                        (iw >= 0) & (iw < w_in)
                
                # Calculate offsets
                input_off = b*input_bs_stride + ic*input_ic_stride + ih*input_h_stride + iw*input_w_stride
                weight_off = ic*weight_ic_stride + oc*weight_oc_stride + kh*weight_kh_stride + kw*weight_kw_stride
                
                # Masked loads
                input_val = tl.load(input_ptr + input_off, mask=valid & mask, other=0.0)
                weight_val = tl.load(weight_ptr + weight_off, mask=mask, other=0.0)
                accumulator += input_val * weight_val
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc, mask=mask, other=0.0)
        accumulator += bias
    
    # Store result
    output_off = b*output_bs_stride + oc*output_oc_stride + oh*output_h_stride + ow*output_w_stride
    tl.store(output_ptr + output_off, accumulator, mask=mask)

def triton_conv_transpose2d(x, weight, bias, stride, padding, dilation):
    batch_size, in_channels, h_in, w_in = x.shape
    out_channels = weight.shape[1]
    kernel_size = weight.shape[2]
    
    # Calculate output shape
    h_out = (h_in - 1)*stride - 2*padding + dilation*(kernel_size-1) + 1
    w_out = (w_in - 1)*stride - 2*padding + dilation*(kernel_size-1) + 1
    
    output = torch.empty((batch_size, out_channels, h_out, w_out), 
              device=x.device, dtype=x.dtype)
    
    # Configure kernel
    total_elements = batch_size * out_channels * h_out * w_out
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels,
        h_in, w_in, kernel_size, stride, padding, dilation,
        h_out, w_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

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
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, 
                                            kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(x, self.weight, self.bias, 
                                     self.stride, self.padding, self.dilation)