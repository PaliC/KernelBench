import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_0(in_ptr0,
    in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 4096
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), None, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (8192 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (12288 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + x3, None)
    tmp17 = tl.load(in_ptr1 + (x0 + 16384 * x2), None, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr1 + (4096 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (8192 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (12288 + x0 + 16384 * x2), None,
        eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-08
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tmp18 = tmp17 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = triton_helpers.maximum(tmp28, tmp13)
    tmp30 = tmp16 / tmp29
    tmp31 = tmp15 * tmp30
    tl.store(out_ptr0 + x3, tmp31, None)


@triton.jit
def triton_per_fused_mean_rsub_sum_1(in_out_ptr0, in_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex % 32
    r1 = rindex // 32
    tmp0 = tl.load(in_ptr0 + (r0 + 128 * r1), None)
    tmp1 = tl.load(in_ptr0 + (32 + r0 + 128 * r1), None)
    tmp3 = tl.load(in_ptr0 + (64 + r0 + 128 * r1), None)
    tmp5 = tl.load(in_ptr0 + (96 + r0 + 128 * r1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp12 = 128.0
    tmp13 = tmp11 / tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([XBLOCK, 1], 0, tl.int32), tmp13, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4096), (4096, 1))
    assert_size_stride(arg1_1, (128, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clamp_min_div_linalg_vector_norm_mul_0[grid(524288)](
            arg1_1, arg0_1, buf0, 524288, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((), (), torch.float32)
        buf2 = buf1
        del buf1
        triton_per_fused_mean_rsub_sum_1[grid(1)](buf2, buf0, 1, 128,
            XBLOCK=1, num_warps=2, num_stages=1)
        del buf0
    return buf2,


class ModelNew(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.

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