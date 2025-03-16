import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 262144 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 5, 5), (480, 75, 25, 5, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 64, 64, 64), (8388608, 262144, 
        4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 66, 60, 60), (12976128, 20160, 
            304, 60, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(2079360)](buf1, primals_2, 
            2079360, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
    return reinterpret_tensor(buf1, (16, 64, 66, 60, 60), (12976128, 1, 
        20160, 360, 6), 0
        ), primals_1, primals_3


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_2 = self.conv_transpose3d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]