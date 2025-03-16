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
def triton_poi_fused__log_softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 16
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (16 + x0 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (32 + x0 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (48 + x0 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_per_fused__log_softmax_div_mul_neg_sum_1(in_out_ptr0, in_ptr0,
    in_ptr1, xnumel, rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r3 = rindex
    r0 = rindex % 16
    r2 = rindex // 64
    tmp0 = tl.load(in_ptr0 + r3, None)
    tmp1 = tl.load(in_ptr0 + (r0 + 64 * r2), None, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (16 + r0 + 64 * r2), None, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (32 + r0 + 64 * r2), None, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (48 + r0 + 64 * r2), None, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr1 + r3, None)
    tmp2 = tl_math.exp(tmp1)
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp2 + tmp4
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp5 + tmp7
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp0 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = -tmp18
    tmp20 = 0.015625
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([1], 0, tl.int32), tmp21, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__log_softmax_0[grid(256)](arg1_1, buf0, 256,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1
        del buf1
        triton_per_fused__log_softmax_div_mul_neg_sum_1[grid(1)](buf2, buf0,
            arg0_1, 1, 256, num_warps=2, num_stages=1)
        del arg0_1
        del buf0
    return buf2,


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]