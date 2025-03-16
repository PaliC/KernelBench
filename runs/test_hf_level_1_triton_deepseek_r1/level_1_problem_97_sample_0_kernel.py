import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    pred_ptr,
    target_ptr,
    output_ptr,
    feature_dim,
    batch_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    feature_start = 0
    dot_acc = 0.0
    pred_norm_acc = 0.0
    target_norm_acc = 0.0

    for feature_start in range(0, feature_dim, BLOCK_SIZE):
        feature_offsets = tl.arange(0, BLOCK_SIZE)
        feature_idx = feature_start + feature_offsets
        mask = feature_idx < feature_dim

        pred = tl.load(pred_ptr + batch_idx * feature_dim + feature_idx, mask=mask, other=0.0)
        target = tl.load(target_ptr + batch_idx * feature_dim + feature_idx, mask=mask, other=0.0)

        dot_acc += tl.sum(pred * target)
        pred_norm_acc += tl.sum(pred * pred)
        target_norm_acc += tl.sum(target * target)

    pred_norm = tl.sqrt(pred_norm_acc + eps)
    target_norm = tl.sqrt(target_norm_acc + eps)
    cos_sim = dot_acc / (pred_norm * target_norm + eps)
    output = 1.0 - cos_sim
    tl.store(output_ptr + batch_idx, output)

def triton_cosine_similarity(predictions, targets, eps=1e-8):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, feature_dim = predictions.shape
    output = torch.empty(batch_size, device=predictions.device)

    BLOCK_SIZE = 128
    grid = (batch_size,)
    cosine_similarity_kernel[grid](
        predictions, targets, output,
        feature_dim, batch_size, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(triton_cosine_similarity(predictions, targets))