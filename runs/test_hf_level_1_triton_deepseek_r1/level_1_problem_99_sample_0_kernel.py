import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triplet_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    output_ptr,
    margin,
    eps,
    feature_dim,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    dim_offsets = pid * feature_dim + offsets

    mask = offsets < feature_dim

    a = tl.load(anchor_ptr + dim_offsets, mask=mask, other=0.0)
    p = tl.load(positive_ptr + dim_offsets, mask=mask, other=0.0)
    n = tl.load(negative_ptr + dim_offsets, mask=mask, other=0.0)

    diff_ap = a - p
    diff_an = a - n

    sum_ap = tl.sum(diff_ap * diff_ap, axis=0)
    sum_an = tl.sum(diff_an * diff_an, axis=0)

    d_ap = tl.sqrt(sum_ap) + eps
    d_an = tl.sqrt(sum_an) + eps

    loss = tl.maximum(d_ap - d_an + margin, 0.0)
    tl.store(output_ptr + pid, loss)


def triton_triplet_margin_loss(anchor, positive, negative, margin, eps=1e-6):
    assert all(t.is_cuda and t.is_contiguous() for t in [anchor, positive, negative])
    batch_size, feature_dim = anchor.shape

    losses = torch.empty(batch_size, device=anchor.device)
    grid = (triton.cdiv(batch_size, 1),)
    
    BLOCK_SIZE = triton.next_power_of_2(feature_dim)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096

    triplet_loss_kernel[grid](
        anchor, positive, negative,
        losses,
        margin,
        eps,
        feature_dim,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return losses.mean()


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)