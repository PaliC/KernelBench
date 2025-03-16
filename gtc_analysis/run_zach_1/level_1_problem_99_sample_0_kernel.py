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
def triton_per_fused_add_clamp_min_mean_mul_neg_sub_0(in_out_ptr0, in_ptr0,
    in_ptr1, in_ptr2, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + r0, None)
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = 256.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + tl.full([1], 0, tl.int32), tmp13, None)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 1), (1, 1))
    assert_size_stride(arg1_1, (128, 1), (1, 1))
    assert_size_stride(arg2_1, (128, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_per_fused_add_clamp_min_mean_mul_neg_sub_0[grid(1)](buf1,
            arg1_1, arg0_1, arg2_1, 1, 256, num_warps=2, num_stages=1)
        del arg0_1
        del arg1_1
        del arg2_1
    return buf1,


class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, input_0, input_1, input_2):
        arg0_1 = input_0
        arg1_1 = input_1
        arg2_1 = input_2
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]