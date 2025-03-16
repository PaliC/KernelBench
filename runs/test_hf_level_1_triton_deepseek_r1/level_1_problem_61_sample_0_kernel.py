import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_d,
    input_h,
    input_w,
    output_d,
    output_h,
    output_w,
    stride,
    padding,
    output_padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    GROUPS: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < batch_size * out_channels * output_d * output_h * output_w

    # Calculate output indices
    batch = pid // (out_channels * output_d * output_h * output_w)
    pid_remainder = pid % (out_channels * output_d * output_h * output_w)
    co = pid_remainder // (output_d * output_h * output_w)
    pid_remainder = pid_remainder % (output_d * output_h * output_w)
    d = pid_remainder // (output_h * output_w)
    pid_remainder = pid_remainder % (output_h * output_w)
    h = pid_remainder // output_w
    w = pid_remainder % output_w

    # Group parameters
    out_channels_per_group = out_channels // GROUPS
    g = co // out_channels_per_group
    co_in_group = co % out_channels_per_group
    in_channels_per_group = in_channels // GROUPS
    ci_start = g * in_channels_per_group

    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)

    # Iterate over kernel
    for kd in range(KERNEL_SIZE):
        for kh in range(KERNEL_SIZE):
            for kw in range(KERNEL_SIZE):
                d_in = (d - kd + PADDING) // STRIDE
                h_in = (h - kh + PADDING) // STRIDE
                w_in = (w - kw + PADDING) // STRIDE

                # Check valid input indices and divisibility
                valid = True
                valid &= (d - kd + PADDING) % STRIDE == 0
                valid &= (h - kh + PADDING) % STRIDE == 0
                valid &= (w - kw + PADDING) % STRIDE == 0
                valid &= d_in >= 0 and d_in < input_d
                valid &= h_in >= 0 and h_in < input_h
                valid &= w_in >= 0 and w_in < input_w

                if valid:
                    for ci_offset in range(in_channels_per_group):
                        ci = ci_start + ci_offset
                        # Calculate input index
                        input_idx = (
                            batch * in_channels * input_d * input_h * input_w +
                            ci * input_d * input_h * input_w +
                            d_in * input_h * input_w +
                            h_in * input_w +
                            w_in
                        )
                        input_val = tl.load(input_ptr + input_idx, mask=None)

                        # Calculate weight index
                        weight_idx = (
                            ci * (out_channels_per_group * KERNEL_SIZE**3) +
                            co_in_group * KERNEL_SIZE**3 +
                            kd * KERNEL_SIZE**2 +
                            kh * KERNEL_SIZE +
                            kw
                        )
                        weight_val = tl.load(weight_ptr + weight_idx, mask=None)
                        acc += input_val * weight_val

    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + co, mask=mask)
        acc += bias_val

    tl.store(output_ptr + pid, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        batch_size, _, input_d, input_h, input_w = x.shape

        # Calculate output dimensions
        output_d = (input_d - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        output_h = (input_h - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        output_w = (input_w - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding

        output = torch.empty((batch_size, self.out_channels, output_d, output_h, output_w),
                            device=x.device, dtype=x.dtype)

        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        conv_transpose3d_kernel[grid](
            x, self.weight, self.bias, output,
            batch_size, self.in_channels, self.out_channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            self.stride, self.padding, self.output_padding, self.groups,
            BLOCK_SIZE=128,
            KERNEL_SIZE=self.kernel_size,
            STRIDE=self.stride,
            PADDING=self.padding,
            GROUPS=self.groups,
        )

        return output