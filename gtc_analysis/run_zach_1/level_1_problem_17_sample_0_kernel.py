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
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = yindex // 4096
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4096 * x2 + 16777216 * y1), xmask &
        ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4096 * y3), tmp0, xmask & ymask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg1_1, (2048, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(2048, 4096)](arg1_1, buf0, 2048, 
            4096, XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)
        extern_kernels.mm(arg0_1, buf0, out=buf1)
        del arg0_1
        del buf0
    return reinterpret_tensor(buf1, (1024, 2048), (1, 1024), 0),


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]