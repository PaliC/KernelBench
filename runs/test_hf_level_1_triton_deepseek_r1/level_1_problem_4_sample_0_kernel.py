import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_b1,
    stride_cm, stride_c1,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    sum = 0.0
    for i in range(0, K, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < K

        a = tl.load(A_ptr + row * stride_am + offsets * stride_ak, mask=mask, other=0.0)
        b = tl.load(B_ptr + offsets * stride_bk + 0 * stride_b1, mask=mask, other=0.0)
        sum += tl.sum(a * b)

    tl.store(C_ptr + row * stride_cm + 0 * stride_c1, sum)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    C = torch.empty(M, 1, device=A.device, dtype=A.dtype)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_b1 = B.stride()
    stride_cm, stride_c1 = C.stride()

    BLOCK_SIZE = 1024  # Tunable parameter
    grid = (M,)
    matvec_kernel[grid](A, B, C, M, K,
                        stride_am, stride_ak,
                        stride_bk, stride_b1,
                        stride_cm, stride_c1,
                        BLOCK_SIZE=BLOCK_SIZE)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)