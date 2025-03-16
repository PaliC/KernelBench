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
def triton_poi_fused_diag_embed_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % N
    x1 = xindex // N
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + x2, tmp5, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (N,), (1,))
    assert_size_stride(arg1_1, (N, M), (M, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((N, N), (N, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_diag_embed_0[grid(N)](arg0_1, buf0, N, XBLOCK=N,
            num_warps=1, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((N, M), (M, 1), torch.float32)
        extern_kernels.mm(buf0, arg1_1, out=buf1)
        del arg1_1
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]