import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    kernel_size,
    stride,
    padding,
    input_length,
    output_length,
    batch_size,
    in_channels,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    channels_out = in_channels * output_length
    batch = offsets // channels_out
    remaining = offsets % channels_out
    channel = remaining // output_length
    out_idx = remaining % output_length

    start_in = out_idx * stride - padding
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(kernel_size):
        pos = start_in + i
        input_offset = batch * in_channels * input_length + channel * input_length + pos
        val = tl.load(
            input_ptr + input_offset,
            mask=(pos >= 0) & (pos < input_length) & mask,
            other=0.0
        )
        sum_val += val

    avg = sum_val / kernel_size
    output_offset = batch * in_channels * output_length + channel * output_length + out_idx
    tl.store(output_ptr + output_offset, avg, mask=mask)


class TritonAvgPool1DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding):
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        x = x.contiguous()
        batch_size, in_channels, input_length = x.shape
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1
        output_shape = (batch_size, in_channels, output_length)
        out = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        
        n_elements = batch_size * in_channels * output_length
        BLOCK_SIZE = 128
        grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        
        avg_pool1d_kernel[grid](
            x, out,
            kernel_size, stride, padding,
            input_length, output_length,
            batch_size, in_channels,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = torch.nn.functional.avg_pool1d_backward(
            grad_output, x,
            ctx.kernel_size, ctx.stride, ctx.padding,
            count_include_pad=True
        )
        return grad_input, None, None, None


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonAvgPool1DFunction.apply(x, self.kernel_size, self.stride, self.padding)