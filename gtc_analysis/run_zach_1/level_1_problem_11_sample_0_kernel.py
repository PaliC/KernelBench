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
    x0 = xindex % 512
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * (x1 % 4) + 2048 * (x1 // 4)), None)
    tl.store(out_ptr0 + x2, tmp0, None)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * (x1 % 4) + 1024 * (x1 // 4)), None)
    tl.store(out_ptr0 + x2, tmp0, None)


@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 768
    x1 = xindex // 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768 * (x1 % 4) + 3072 * (x1 // 4)), None)
    tl.store(out_ptr0 + x2, tmp0, None)


@triton.jit
def triton_poi_fused_add_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 4 % 768
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 256, 512, 256), (4194304, 131072, 256, 1))
    assert_size_stride(arg1_1, (256, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 256, 512, 256), (4194304, 512, 1, 
            2048), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(2048, 512)](arg0_1, buf0, 2048, 512,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del arg0_1
        buf1 = empty_strided_cuda((256, 256), (256, 1), torch.float32)
        triton_poi_fused_clone_1[grid(256, 256)](arg1_1, buf1, 256, 256,
            XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((4, 768, 768), (589824, 768, 1), torch.
            float32)
        triton_poi_fused_clone_2[grid(2359296)](buf0, buf2, 2359296, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((1, 589824, 4), (236928, 4, 1), torch.
            float32)
        extern_kernels.bmm(reinterpret_tensor(buf2, (1, 589824, 768), (0, 
            768, 1), 0), reinterpret_tensor(buf1, (1, 768, 4), (0, 4, 1), 0
            ), out=buf3)
        del buf1
        del buf2
        buf4 = reinterpret_tensor(buf3, (4, 768, 768), (589824, 768, 1), 0)
        del buf3
        triton_poi_fused_add_3[grid(2359296)](buf4, buf4, 2359296, XBLOCK=
            1024, num_warps=4, num_stages=1)
    return buf4,


class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]