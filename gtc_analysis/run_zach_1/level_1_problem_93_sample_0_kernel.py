import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0


@triton.jit
def triton_per_fused_cumsum_masked_fill_mul_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    rnumel = 4000
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * r2 + 512 * x1), rmask & xmask,
        other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128 * r2 + 512 * x1), rmask & xmask,
        other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4000), (4000, 1))
    assert_size_stride(arg1_1, (128, 4000), (4000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128,), (1,), torch.float32)
        get_raw_stream(0)
        triton_per_fused_cumsum_masked_fill_mul_0[grid(128)](arg0_1, arg1_1,
            buf0, 128, 4000, XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]