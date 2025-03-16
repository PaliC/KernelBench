import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,
    out_ptr,
    input_length,
    output_length,
    batch_size,
    channels,
    kernel_size,
    stride,
    padding,
    dilation,
    total_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_output_elements

    for idx in tl.range(BLOCK_SIZE):
        if mask[idx]:
            global_idx = offsets[idx]
            
            b = global_idx // (channels * output_length)
            remaining = global_idx % (channels * output_length)
            c = remaining // output_length
            out_l = remaining % output_length

            input_start = out_l * stride - padding
            max_val = -float('inf')

            for i in range(kernel_size):
                input_pos = input_start + i * dilation
                if input_pos >= 0 and input_pos < input_length:
                    input_offset = b * channels * input_length + c * input_length + input_pos
                    val = tl.load(x_ptr + input_offset)
                else:
                    val = -float('inf')
                max_val = tl.maximum(max_val, val)

            tl.store(out_ptr + global_idx, max_val)


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    batch_size, channels, input_length = x.shape

    output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((batch_size, channels, output_length), device=x.device, dtype=x.dtype)
    total_output_elements = out.numel()

    BLOCK_SIZE = 128
    grid = lambda meta: ((total_output_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    maxpool1d_kernel[grid](
        x, out,
        input_length,
        output_length,
        batch_size,
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        total_output_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)