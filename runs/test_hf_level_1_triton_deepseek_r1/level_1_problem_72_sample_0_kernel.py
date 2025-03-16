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
    # Input dimensions
    B, Cin, Din, Hin, Win,
    # Output dimensions
    Cout, Dout, Hout, Wout,
    # Kernel parameters
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    groups,
    # Tensor strides
    input_stride_b, input_stride_cin, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_cout, weight_stride_cin, weight_stride_d, weight_stride_h, weight_stride_w,
    output_stride_b, output_stride_cout, output_stride_d, output_stride_h, output_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Calculate output element indices
    num_output_elements = B * Cout * Dout * Hout * Wout
    elements_per_program = (num_output_elements + num_pid - 1) // num_pid
    start_idx = pid * elements_per_program
    end_idx = tl.minimum((pid + 1) * elements_per_program, num_output_elements)
    
    group_size = Cout // groups
    g = tl.arange(0, end_idx - start_idx)  # Current group of elements
    
    for idx in range(start_idx + tl.multiple_of(g, BLOCK_SIZE), end_idx, BLOCK_SIZE):
        # Convert linear index to 5D output tensor indices
        b = idx // (Cout * Dout * Hout * Wout)
        remainder = idx % (Cout * Dout * Hout * Wout)
        cout = remainder // (Dout * Hout * Wout)
        remainder = remainder % (Dout * Hout * Wout)
        dout = remainder // (Hout * Wout)
        remainder = remainder % (Hout * Wout)
        hout = remainder // Wout
        wout = remainder % Wout

        # Calculate input indices based on output indices and stride
        din = (dout + padding_d) // stride_d
        hin = (hout + padding_h) // stride_h
        win = (wout + padding_w) // stride_w
        
        # Calculate kernel offsets
        kd_offset = (dout + padding_d) % stride_d
        kh_offset = (hout + padding_h) % stride_h
        kw_offset = (wout + padding_w) % stride_w

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Group processing
        g = cout // group_size
        cin_start = g * (Cin // groups)
        cin_end = (g + 1) * (Cin // groups)
        
        # Loop through kernel dimensions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Calculate actual input position
                    din_actual = din * stride_d - padding_d + kd + kd_offset
                    hin_actual = hin * stride_h - padding_h + kh + kh_offset
                    win_actual = win * stride_w - padding_w + kw + kw_offset
                    
                    # Check input boundaries
                    if (din_actual >= 0 and din_actual < Din and
                        hin_actual >= 0 and hin_actual < Hin and
                        win_actual >= 0 and win_actual < Win):
                        
                        # Load input value
                        input_offset = (b * input_stride_b + 
                                        (cin_start // (Cin // groups)) * input_stride_cin + 
                                        din_actual * input_stride_d + 
                                        hin_actual * input_stride_h + 
                                        win_actual * input_stride_w)
                        input_val = tl.load(input_ptr + input_offset + g, mask=g < cin_end, other=0.0)
                        
                        # Load weight value
                        weight_offset = (cout * weight_stride_cout + 
                                         (cin_start // (Cin // groups)) * weight_stride_cin + 
                                         kd * weight_stride_d + 
                                         kh * weight_stride_h + 
                                         kw * weight_stride_w)
                        weight_val = tl.load(weight_ptr + weight_offset + g, mask=g < cin_end, other=0.0)
                        
                        acc += input_val * weight_val

        # Add bias if present
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + cout)
            acc += bias_val

        # Store result
        output_offset = (b * output_stride_b + 
                        cout * output_stride_cout + 
                        dout * output_stride_d + 
                        hout * output_stride_h + 
                        wout * output_stride_w)
        tl.store(output_ptr + output_offset, acc)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Cin, Din, Hin, Win = x.shape
        Kd, Kh, Kw = self.kernel_size
        
        # Calculate output shape
        Dout = (Din - 1) * self.stride[0] - 2 * self.padding[0] + Kd + self.output_padding[0]
        Hout = (Hin - 1) * self.stride[1] - 2 * self.padding[1] + Kh + self.output_padding[1]
        Wout = (Win - 1) * self.stride[2] - 2 * self.padding[2] + Kw + self.output_padding[2]
        
        output = torch.empty((B, self.out_channels, Dout, Hout, Wout), 
                            device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']),)
        
        conv_transpose3d_kernel[grid](
            x, self.weight, self.bias, output,
            B, Cin, Din, Hin, Win,
            self.out_channels, Dout, Hout, Wout,
            Kd, Kh, Kw,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.groups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), 
            self.weight.stride(3), self.weight.stride(4),
            output.stride(0), output.stride(1), output.stride(2), 
            output.stride(3), output.stride(4),
            BLOCK_SIZE=128
        )
        
        return output