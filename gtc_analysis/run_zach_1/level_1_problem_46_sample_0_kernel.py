import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool3d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1 + 4096 * (x1 % 4 // 4) + 16384 * (
        (4 * (x1 // 4 % 4) + x1 % 4) // 16)), xmask)
    tmp1 = tl.load(in_ptr0 + (32 + x0 + 64 * x1 + 4096 * (x1 % 4 // 4) + 
        16384 * ((4 * (x1 // 4 % 4) + x1 % 4) // 16)), xmask)
    tmp3 = tl.load(in_ptr0 + (8192 + x0 + 64 * x1 + 4096 * (x1 % 4 // 4) + 
        16384 * ((4 * (x1 // 4 % 4) + x1 % 4) // 16)), xmask)
    tmp5 = tl.load(in_ptr0 + (8224 + x0 + 64 * x1 + 4096 * (x1 % 4 // 4) + 
        16384 * ((4 * (x1 // 4 % 4) + x1 % 4) // 16)), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 32, 64, 64, 64), (8388608, 262144, 4096,
        64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 32, 32, 32, 32), (2097152, 65536, 
            2048, 64, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool3d_0[grid(512)](arg0_1, buf0, 512, XBLOCK=
            256, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]