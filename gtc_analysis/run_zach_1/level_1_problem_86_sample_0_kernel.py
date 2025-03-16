import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 16384 % 3
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (3, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_2, (3,), (1,))
    assert_size_stride(primals_3, (16, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(primals_4, (64, 3, 1, 1), (3, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf0, (16, 3, 254, 254), (196452, 65528, 254, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(3145728)](buf1, primals_2, 
            3145728, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (16, 64, 254, 254), (508032, 65528, 254, 1))
    return buf2, primals_1, primals_3, primals_4, buf1


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.depthwise.weight
        primals_2 = self.pointwise.weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]