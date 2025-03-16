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
def triton_per_fused_smooth_l1_loss_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    rnumel):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    tl.full([1], xoffset, tl.int32)
    tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + r0, None)
    tmp1 = tl.load(in_ptr1 + r0, None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.abs(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 < tmp4
    tmp6 = tmp3 * tmp3
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp4
    tmp10 = tmp3 - tmp7
    tmp11 = tl.where(tmp5, tmp9, tmp10)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 256.0
    tmp16 = tmp14 / tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([1], 0, tl.int32), tmp16, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (batch_size, 4096), (4096, 1))
    assert_size_stride(arg1_1, (batch_size, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_smooth_l1_loss_0[grid(1)](buf1, arg1_1, arg0_1, 1,
            256, num_warps=2, num_stages=1)
        del arg0_1
        del arg1_1
    return buf1,


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

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