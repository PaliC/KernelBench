import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused__softmax_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp4, xmask)
    tl.store(out_ptr0 + (4096 * dim + x0), tmp10, xmask)


@triton.jit
def triton_red_fused__softmax_div_log_mul_sub_sum_xlogy_1(in_out_ptr0,
    in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK: tl.
    constexpr, RBLOCK: tl.constexpr):
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        r1 = rindex // 16
        tmp0 = tl.load(in_ptr0 + r2, rmask, eviction_policy='evict_first',
            other=0.0)
        tmp1 = tl.load(in_ptr1 + r1, rmask, eviction_policy='evict_last',
            other=0.0)
        tmp4 = tl.load(in_ptr2 + r1, rmask, eviction_policy='evict_last',
            other=0.0)
        tmp6 = tl.load(in_ptr3 + r2, rmask, eviction_policy='evict_first',
            other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl_math.exp(tmp2)
        tmp5 = tmp3 / tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = libdevice.isnan(tmp7).to(tl.int1)
        tmp9 = 0.0
        tmp10 = tmp7 == tmp9
        tmp11 = tl_math.log(tmp7)
        tmp12 = tmp7 * tmp11
        tmp13 = tl.where(tmp10, tmp9, tmp12)
        tmp14 = float('nan')
        tmp15 = tl.where(tmp8, tmp14, tmp13)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp34 + tmp16
        _tmp34 = tl.where(rmask, tmp18, _tmp34)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp35 = 0.25
    tmp36 = tmp34 * tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp36, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096), (4096, 1))
    assert_size_stride(arg1_1, (128, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        buf1 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        get_raw_stream(0)
        triton_per_fused__softmax_0[grid(128)](arg1_1, buf0, buf1, 128, 16,
            XBLOCK=8, num_warps=2, num_stages=1)
        buf2 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        buf3 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        triton_per_fused__softmax_0[grid(128)](arg0_1, buf2, buf3, 128, 16,
            XBLOCK=8, num_warps=2, num_stages=1)
        buf4 = empty_strided_cuda((), (), torch.float32)
        buf5 = buf4
        del buf4
        triton_red_fused__softmax_div_log_mul_sub_sum_xlogy_1[grid(1)](buf5,
            arg1_1, buf0, buf1, arg0_1, buf2, buf3, 1, 4096, XBLOCK=1,
            RBLOCK=2048, num_warps=16, num_stages=1)
        del arg0_1
        del arg1_1
        del buf0
        del buf1
        del buf2
        del buf3
    return buf5,


class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

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