import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_size,
    stride,
    padding,
    input_stride_b,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_output = batch_size * channels * out_h * out_w
    mask = pid < total_output
    
    b = pid // (channels * out_h * out_w)
    remainder = pid % (channels * out_h * out_w)
    c = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    oh = remainder // out_w
    ow = remainder % out_w

    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h = oh * stride + kh - padding
            w = ow * stride + kw - padding
            
            in_bounds = (h >= 0) & (h < in_h) & (w >= 0) & (w < in_w)
            h = tl.maximum(tl.minimum(h, in_h-1), 0)
            w = tl.maximum(tl.minimum(w, in_w-1), 0)
            
            input_offset = b * input_stride_b + c * input_stride_c + h * input_stride_h + w * input_stride_w
            val = tl.load(input_ptr + input_offset, mask=in_bounds & mask, other=0.0)
            sum_val += val
    
    avg = sum_val / (kernel_size * kernel_size)
    output_offset = b * output_stride_b + c * output_stride_c + oh * output_stride_h + ow * output_stride_w
    tl.store(output_ptr + output_offset, avg, mask=mask)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    B, C, H, W = x.shape
    out_h = (H + 2*padding - kernel_size) // stride + 1
    out_w = (W + 2*padding - kernel_size) // stride + 1
    
    output = torch.empty((B, C, out_h, out_w), device=x.device, dtype=x.dtype)
    total_elements = B * C * out_h * out_w
    
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    avg_pool2d_kernel[grid](
        x.data_ptr(), output.data_ptr(),
        B, C, H, W, out_h, out_w,
        kernel_size, stride, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=128
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)