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
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1024
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x2), None, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + x3, tmp0, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 1024, 2048), (2097152, 2048, 1))
    assert_size_stride(arg1_1, (2048, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1024, 2048), (2097152, 2048, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(33554432)](arg0_1, buf0, 33554432,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((16, 1024, 768), (786432, 768, 1), torch
            .float32)
        extern_kernels.bmm(buf0, reinterpret_tensor(arg1_1, (16, 2048, 768
            ), (0, 1, 2048), 0), out=buf1)
        del arg1_1
        del buf0
    return reinterpret_tensor(buf1, (16, 768, 1024), (786432, 1, 768), 0),


class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]