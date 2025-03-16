import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 128 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32 * x2 + 4096 * y1), tmp0, xmask & ymask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_2, (4, 32, 128, 128), (524288, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 32, 128, 128), (524288, 1, 4096, 32),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(512, 128)](primals_2, buf0, 512,
            128, XBLOCK=128, YBLOCK=2, num_warps=4, num_stages=1)
        del primals_2
        buf1 = extern_kernels.convolution(buf0, primals_1, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 64, 128, 128), (1048576, 1, 8192, 64))
        del buf0
    return buf1, primals_1


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose2d.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]