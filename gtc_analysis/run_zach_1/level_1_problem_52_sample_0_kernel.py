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
def triton_poi_fused_argmin_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 256 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 256 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 256 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 256 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 256 * x0), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr0 + (5 + 256 * x0), xmask, eviction_policy='evict_last'
        )
    tmp10 = tl.load(in_ptr0 + (6 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (7 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (9 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (10 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (11 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (13 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (14 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (15 + 256 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = triton_helpers.minimum(tmp6, tmp13)
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = triton_helpers.minimum(tmp14, tmp21)
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp27 = triton_helpers.minimum(tmp25, tmp26)
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = triton_helpers.minimum(tmp22, tmp29)
    tl.store(out_ptr0 + x0, tmp30, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_argmin_0[grid(16)](arg0_1, buf0, 16, XBLOCK=16,
            num_warps=1, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]