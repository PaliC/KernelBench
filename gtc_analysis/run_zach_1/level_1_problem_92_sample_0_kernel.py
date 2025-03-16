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
def triton_per_fused_cumsum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl
    .constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4000 * x0), xmask, other=0.0)
    tmp1 = 0.0
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp0 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8, = tl.associative_scan((tmp6,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r1 + 4000 * x0), tmp8, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4000), (4000, 1))
    buf0 = empty_strided_cuda((128, 4000), (4000, 1), torch.float32)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((128, 4000), (4000, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_cumsum_0[grid(128)](arg0_1, buf1, 128, 4000,
            XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
    return buf1,


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]