import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    input_batch,
    input_channels,
    input_d,
    input_h,
    input_w,
    output_d,
    output_h,
    output_w,
    stride,
    padding,
    dilation,
    KERNEL_SIZE: tl.constexpr,
    CEIL_MODE: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_output_elements = input_batch * input_channels * output_d * output_h * output_w
    if pid >= total_output_elements:
        return

    channels_outd_outh_outw = input_channels * output_d * output_h * output_w
    batch = pid // channels_outd_outh_outw
    remainder = pid % channels_outd_outh_outw

    outd_outh_outw = output_d * output_h * output_w
    channel = remainder // outd_outh_outw
    remainder = remainder % outd_outh_outw

    outh_outw = output_h * output_w
    d = remainder // outh_outw
    remainder = remainder % outh_outw

    h = remainder // output_w
    w = remainder % output_w

    start_d = d * stride - padding
    start_h = h * stride - padding
    start_w = w * stride - padding

    max_val = tl.full((1,), -float('inf'), tl.float32)
    max_d = tl.full((1,), -1, tl.int32)
    max_h = tl.full((1,), -1, tl.int32)
    max_w = tl.full((1,), -1, tl.int32)

    for kd in tl.static_range(KERNEL_SIZE):
        current_d = start_d + kd * dilation
        if current_d < 0 or current_d >= input_d:
            continue
        for kh in tl.static_range(KERNEL_SIZE):
            current_h_pos = start_h + kh * dilation
            if current_h_pos < 0 or current_h_pos >= input_h:
                continue
            for kw in tl.static_range(KERNEL_SIZE):
                current_w_pos = start_w + kw * dilation
                if current_w_pos < 0 or current_w_pos >= input_w:
                    continue
                input_idx = (
                    batch * input_channels * input_d * input_h * input_w +
                    channel * input_d * input_h * input_w +
                    current_d * input_h * input_w +
                    current_h_pos * input_w +
                    current_w_pos
                )
                val = tl.load(input_ptr + input_idx)
                is_max = val > max_val
                max_val = tl.where(is_max, val, max_val)
                max_d = tl.where(is_max, current_d, max_d)
                max_h = tl.where(is_max, current_h_pos, max_h)
                max_w = tl.where(is_max, current_w_pos, max_w)

    output_idx = pid
    tl.store(output_ptr + output_idx, max_val)
    if RETURN_INDICES:
        index_value = max_d * input_h * input_w + max_h * input_w + max_w
        tl.store(indices_ptr + output_idx, index_value)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()

        def compute_output_size(input_size):
            numerator = input_size + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
            if numerator < 0:
                numerator = 0
            if self.ceil_mode:
                return (numerator + self.stride - 1) // self.stride + 1
            else:
                return numerator // self.stride + 1

        input_d, input_h, input_w = x.shape[2], x.shape[3], x.shape[4]
        output_d = compute_output_size(input_d)
        output_h = compute_output_size(input_h)
        output_w = compute_output_size(input_w)

        output_shape = (x.shape[0], x.shape[1], output_d, output_h, output_w)
        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        indices = torch.empty(output_shape, device=x.device, dtype=torch.long) if self.return_indices else torch.empty(0, device=x.device, dtype=torch.long)

        total_output_elements = output.numel()
        BLOCK_SIZE = 128

        grid = lambda meta: (triton.cdiv(total_output_elements, meta['BLOCK_SIZE']), )

        maxpool3d_kernel[grid](
            x, output, indices,
            x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4],
            output_d, output_h, output_w,
            self.stride, self.padding, self.dilation,
            KERNEL_SIZE=self.kernel_size,
            CEIL_MODE=self.ceil_mode,
            RETURN_INDICES=self.return_indices,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if self.return_indices:
            return output, indices
        return output