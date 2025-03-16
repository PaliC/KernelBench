import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_div_linalg_vector_norm_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 16384
    x2 = xindex // 32768
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + (x0 + 32768 * x2), None, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (16384 + x0 + 32768 * x2), None,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (32768 + x0 + 32768 * x2), None,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (40960 + x0 + 32768 * x2), None,
        eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tmp0 / tmp12
    tl.store(out_ptr0 + x3, tmp13, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 16384), (16384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 16384), (16384, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_div_linalg_vector_norm_0[grid(262144)](arg0_1, buf0,
            262144, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        super(ModelNew, self).__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]