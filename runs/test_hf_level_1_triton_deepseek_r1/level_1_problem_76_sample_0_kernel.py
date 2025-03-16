import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    in_channels,
    kernel_size,
    stride,
    dilation,
    input_len,
    output_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_pid = pid // output_len
    out_pos = pid % output_len

    w_offsets = tl.arange(0, BLOCK_SIZE)
    x_offsets = out_pos * stride + w_offsets * dilation
    valid_mask = x_offsets < input_len

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c in range(0, in_channels):
        w = tl.load(w_ptr + c * kernel_size + w_offsets, mask=valid_mask, other=0.0)
        x = tl.load(x_ptr + batch_pid * in_channels * input_len + c * input_len + x_offsets, mask=valid_mask, other=0.0)
        acc += w * x

    if b_ptr is not None:
        bias = tl.load(b_ptr)
        acc += bias

    tl.store(y_ptr + batch_pid * output_len + out_pos, tl.sum(acc, axis=0))


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        if bias:
            bound = 1 / (in_channels * kernel_size) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, _, input_len = x.shape
        padding = 0
        output_len = (input_len + 2*padding - self.dilation*(self.kernel_size-1) -1) // self.stride +1
        
        y = torch.empty((batch_size, self.out_channels, output_len), device=x.device)
        
        grid = (batch_size * output_len,)
        conv1d_kernel[grid](
            x, self.weight, self.bias, y,
            self.in_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
            input_len,
            output_len,
            BLOCK_SIZE=self.kernel_size,
        )
        return y