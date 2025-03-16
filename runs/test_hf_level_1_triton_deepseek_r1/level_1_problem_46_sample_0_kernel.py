import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    input_ptr,
    output_ptr,
    B,
    C,
    D,
    H,
    W,
    D_out,
    H_out,
    W_out,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < B * C * D_out * H_out * W_out
    
    b = (idx // (C * D_out * H_out * W_out)) % B
    c = (idx // (D_out * H_out * W_out)) % C
    d_out = (idx // (H_out * W_out)) % D_out
    h_out = (idx // W_out) % H_out
    w_out = idx % W_out

    d_start = d_out * stride_d - padding_d
    h_start = h_out * stride_h - padding_h
    w_start = w_out * stride_w - padding_w

    sum_val = 0.0
    count = 0

    for kd in range(kernel_d):
        d_in = d_start + kd
        if d_in < 0 or d_in >= D:
            continue
        for kh in range(kernel_h):
            h_in = h_start + kh
            if h_in < 0 or h_in >= H:
                continue
            for kw in range(kernel_w):
                w_in = w_start + kw
                if w_in < 0 or w_in >= W:
                    continue
                input_idx = (b * C * D * H * W + 
                            c * D * H * W + 
                            d_in * H * W + 
                            h_in * W + 
                            w_in)
                sum_val += tl.load(input_ptr + input_idx)
                count += 1

    avg = sum_val / tl.maximum(count, 1)
    tl.store(output_ptr + idx, avg, mask=mask)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    B, C, D, H, W = x.shape
    
    D_out = (D + 2*padding - kernel_size) // stride + 1
    H_out = (H + 2*padding - kernel_size) // stride + 1
    W_out = (W + 2*padding - kernel_size) // stride + 1
    
    output = torch.empty((B, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    num_elements = output.numel()
    BLOCK_SIZE = 128
    
    grid = lambda meta: ((num_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    avg_pool3d_kernel[grid](
        x, output,
        B, C, D, H, W,
        D_out, H_out, W_out,
        stride_d=stride,
        stride_h=stride,
        stride_w=stride,
        padding_d=padding,
        padding_h=padding,
        padding_w=padding,
        kernel_d=kernel_size,
        kernel_h=kernel_size,
        kernel_w=kernel_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)