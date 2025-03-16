import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (N + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_0[grid(16384)](arg0_1, buf0, 16384, XBLOCK
            =128, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        triton_poi_fused_add_div_0[grid(16384)](arg1_1, buf1, 16384, XBLOCK
            =128, num_warps=4, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(buf0, buf1, out=buf2)
        del buf0
        del buf1
    return buf2,


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]