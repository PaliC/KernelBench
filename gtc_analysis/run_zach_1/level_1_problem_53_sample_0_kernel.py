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
def triton_poi_fused_min_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), None)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 64 * x1), None)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 64 * x1), None)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 64 * x1), None)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tl.store(out_ptr0 + x2, tmp6, None)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_min_0[grid(64)](arg0_1, buf0, 64, XBLOCK=64,
            num_warps=1, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]