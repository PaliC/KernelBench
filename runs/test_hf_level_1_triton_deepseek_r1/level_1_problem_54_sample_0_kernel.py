import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride, padding, dilation,
    kernel_size,
    x_batch_stride, x_channel_stride, x_d_stride, x_h_stride, x_w_stride,
    weight_oc_stride, weight_ic_stride, weight_d_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_oc_stride, output_d_stride, output_h_stride, output_w_stride,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # Calculate output coordinates
    batch_id = pid // (out_channels * D_out * H_out * W_out)
    remaining = pid % (out_channels * D_out * H_out * W_out)
    oc = remaining // (D_out * H_out * W_out)
    remaining = remaining % (D_out * H_out * W_out)
    d_out = remaining // (H_out * W_out)
    remaining = remaining % (H_out * W_out)
    h_out = remaining // W_out
    w_out = remaining % W_out

    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop through kernel dimensions and input channels
    for kd in range(K):
        d_in = d_out * stride - padding + kd * dilation
        if d_in < 0 or d_in >= D_in:
            continue
            
        for kh in range(K):
            h_in = h_out * stride - padding + kh * dilation
            if h_in < 0 or h_in >= H_in:
                continue
                
            for kw in range(K):
                w_in = w_out * stride - padding + kw * dilation
                if w_in < 0 or w_in >= W_in:
                    continue
                
                for ic in range(0, in_channels, BLOCK_SIZE):
                    off_ic = ic + tl.arange(0, BLOCK_SIZE)
                    mask_ic = off_ic < in_channels
                    
                    # Calculate input and weight offsets
                    x_offset = (batch_id * x_batch_stride + 
                               off_ic * x_channel_stride + 
                               d_in * x_d_stride + 
                               h_in * x_h_stride + 
                               w_in * x_w_stride)
                    
                    weight_offset = (oc * weight_oc_stride + 
                                    off_ic * weight_ic_stride + 
                                    kd * weight_d_stride + 
                                    kh * weight_h_stride + 
                                    kw * weight_w_stride)
                    
                    x = tl.load(x_ptr + x_offset, mask=mask_ic, other=0)
                    w = tl.load(weight_ptr + weight_offset, mask=mask_ic, other=0)
                    acc += tl.sum(x * w)

    if has_bias:
        bias = tl.load(bias_ptr + oc)
        acc += bias
        
    # Calculate output offset and store
    output_offset = (batch_id * output_batch_stride +
                    oc * output_oc_stride +
                    d_out * output_d_stride +
                    h_out * output_h_stride +
                    w_out * output_w_stride)
    tl.store(output_ptr + output_offset, acc)

def triton_conv3d(x, weight, bias, stride, padding, dilation, kernel_size):
    batch, in_channels, D_in, H_in, W_in = x.shape
    out_channels = weight.shape[0]
    K = kernel_size
    
    D_out = (D_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    H_out = (H_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    output = torch.empty((batch, out_channels, D_out, H_out, W_out), 
                        device=x.device, dtype=x.dtype)
    
    # Flatten spatial dimensions for kernel
    total_output_elements = batch * out_channels * D_out * H_out * W_out
    BLOCK_SIZE = 128
    
    grid = lambda meta: (total_output_elements,)
    
    conv3d_kernel[grid](
        x, weight, bias, output,
        in_channels, out_channels,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride, padding, dilation, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        K=K,
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        assert groups == 1, "Grouped conv not implemented"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, 
            kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        return triton_conv3d(
            x, self.weight, self.bias, 
            self.stride, self.padding, self.dilation, self.kernel_size
        )