import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), None, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x3, tmp9, None)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), None, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x3, tmp8, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 16384), (16384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 16384), (16384, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(262144)](arg0_1, buf0, 262144,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((16, 16384), (16384, 1), torch.float32)
        triton_poi_fused__softmax_1[grid(262144)](buf0, buf1, 262144,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]