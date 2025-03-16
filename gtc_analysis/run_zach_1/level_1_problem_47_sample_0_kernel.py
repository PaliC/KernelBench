import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + 1)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp5 = tl.load(in_ptr0 + 2)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tmp4 + tmp6
    tmp8 = tl.load(in_ptr0 + 3)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tmp7 + tmp9
    tl.store(out_ptr0 + tl.full([XBLOCK], 0, tl.int32), tmp10, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 1, 1), (), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_sum_0[grid(1)](arg0_1, buf0, 1, XBLOCK=1,
            num_warps=1, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]