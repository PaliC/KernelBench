import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 115200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 60 % 60
    x0 = xindex % 60
    x3 = xindex // 60
    x4 = xindex
    tmp0 = -1 + 2 * x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = -1 + 2 * x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (-129 + 2 * x0 + 256 * x3), tmp10 & xmask,
        eviction_policy='evict_last', other=float('-inf'))
    tmp12 = 2 * x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (-128 + 2 * x0 + 256 * x3), tmp16 & xmask,
        eviction_policy='evict_last', other=float('-inf'))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 2 * x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp22 & tmp9
    tmp24 = tl.load(in_ptr0 + (-1 + 2 * x0 + 256 * x3), tmp23 & xmask,
        eviction_policy='evict_last', other=float('-inf'))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = tmp22 & tmp15
    tmp27 = tl.load(in_ptr0 + (2 * x0 + 256 * x3), tmp26 & xmask,
        eviction_policy='evict_last', other=float('-inf'))
    tmp28 = triton_helpers.maximum(tmp27, tmp25)
    tl.store(out_ptr0 + x4, tmp28, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 128, 128), (524288, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 32, 60, 60), (115200, 3600, 60, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(115200)](arg0_1,
            buf0, 115200, XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]