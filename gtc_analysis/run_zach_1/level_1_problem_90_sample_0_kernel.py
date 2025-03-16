import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def _triton_helper_fn_mul(arg0_0, arg1_0):
    tmp0 = arg0_0 * arg1_0
    return tmp0


@triton.jit
def triton_per_fused_cumprod_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl
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
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp3, = tl.associative_scan((tmp2,), 1, _triton_helper_fn_mul)
    tl.store(out_ptr0 + (r1 + 4000 * x0), tmp3, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 4000), (4000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 4000), (4000, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_cumprod_0[grid(128)](arg0_1, buf0, 128, 4000,
            XBLOCK=1, num_warps=2, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]