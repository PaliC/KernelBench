import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_softplus_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(out_ptr0 + x0, tmp5, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 16384), (16384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 16384), (16384, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_softplus_0[grid(262144)](arg0_1, buf0, 262144,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]