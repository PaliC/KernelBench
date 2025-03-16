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
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 36 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (16, 32, 3, 5, 7), (6720, 210, 70, 14, 2))
    assert_size_stride(primals_2, (4, 32, 4, 4, 4), (2048, 64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 6, 8, 8), (3072, 192, 32, 4, 1))
        buf1 = empty_strided_cuda((4, 16, 6, 8, 8), (3072, 192, 32, 4, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(128)](buf0, buf1, 128, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf0
    return reinterpret_tensor(buf1, (4, 6, 16, 8, 8), (3072, 512, 32, 4, 1), 0
        ), primals_1, primals_2


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of 3 integers representing the kernel size in the form (depth, height, width).
        stride (tuple, optional): Tuple of 3 integers representing the stride in the form (depth, height, width). Defaults to (1, 1, 1).
        padding (tuple, optional): Tuple of 3 integers representing the padding in the form (depth, height, width). Defaults to (0, 0, 0).
        output_padding (tuple, optional): Tuple of 3 integers representing the output padding in the form (depth, height, width). Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, input_0):
        primals_1 = self.conv_transpose3d.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]