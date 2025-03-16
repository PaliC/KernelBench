import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    # Tensor dimensions
    n, c, h_in, w_in,
    h_out, w_out,
    # Convolution parameters
    stride,
    padding,
    kernel_size,
    # Memory strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_c, w_stride_h, w_stride_w,
    # Meta parameters
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n * c * h_out * w_out, BLOCK_SIZE)
    
    # Split program ID into batch, channel, and spatial components
    pid_batch = pid // (c * h_out * w_out)
    pid_c = (pid // (h_out * w_out)) % c
    pid_h = (pid // w_out) % h_out
    pid_w = pid % w_out

    # Calculate output spatial position
    h_start = pid_h * stride - padding
    w_start = pid_w * stride - padding
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over kernel elements
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h = h_start + kh
            w = w_start + kw
            
            # Check if input is out of bounds (considering padding)
            if h >= 0 and h < h_in and w >= 0 and w < w_in:
                x_offset = (pid_batch * x_stride_n + 
                           pid_c * x_stride_c + 
                           h * x_stride_h + 
                           w * x_stride_w)
                w_offset = (pid_c * w_stride_c + 
                           kh * w_stride_h + 
                           kw * w_stride_w)
                
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c)
        acc += bias

    # Calculate output offset
    out_offset = (pid_batch * c * h_out * w_out +
                  pid_c * h_out * w_out +
                  pid_h * w_out +
                  pid_w)
    
    tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
):
    # Ensure tensors are on CUDA and contiguous
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get tensor dimensions
    n, c, h_in, w_in = x.shape
    kernel_size = weight.shape[2]
    
    # Calculate output dimensions
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((n, c, h_out, w_out), device=x.device)
    
    # Calculate number of elements
    total_elements = n * c * h_out * w_out
    
    # Configure kernel grid
    grid = lambda meta: (total_elements,)
    
    # Get memory strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_c, w_stride_h, w_stride_w = weight.stride()
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        n, c, h_in, w_in, h_out, w_out,
        stride, padding, kernel_size,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        w_stride_c, w_stride_h, w_stride_w,
        BLOCK_SIZE=128
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        assert in_channels == out_channels, "Depthwise requires in_channels == out_channels"
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, 1, kernel_size, kernel_size)
        )
        
        # Initialize bias if required
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding
        )