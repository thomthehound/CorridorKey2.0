"""Inference-only, source-equivalent region attention lowering for Vulkan.

CorridorKey's reference groups visible tokens within shifted windows and
scatter-writes each group's SDPA result. The importer can instead represent
the identical grouping as a shape-constant boolean attention mask. This is
a correctness-oriented Stage2 import path, not the eventual high-performance
implementation: large windows must use fused or explicitly grouped kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_sdpa_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_windows: int,
    full_idx: tuple[int, ...],
    groups: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> torch.Tensor:
    """Apply the source's exact region membership without indexed writes.

    Only region geometry is specialized. Every participating query attends
    to exactly the source's key set. Queries in padding-only positions
    produce zero, as in the original preallocated output tensor.
    """
    if num_windows <= 0:
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False,
        )
    total, heads, tokens, head_dim = q.shape
    if total % num_windows:
        raise ValueError("Region window count does not divide attention batch")
    batches = total // num_windows

    # Static metadata from _shifted_region_groups_2d_cached. Materialize
    # constants before graph lowering; never infer membership from Q/K/V.
    allowed = [
        [[False] * tokens for _ in range(tokens)]
        for _ in range(num_windows)
    ]
    valid = [[False] * tokens for _ in range(num_windows)]
    for window in full_idx:
        for token in range(tokens):
            allowed[window][token] = [True] * tokens
            valid[window][token] = True
    for token_group, window_group in groups:
        for window in window_group:
            for query in token_group:
                valid[window][query] = True
                for key in token_group:
                    allowed[window][query][key] = True

    # All-masked SDPA rows would have undefined softmax. Give each unused
    # padding query a harmless key, then discard its output explicitly.
    for window in range(num_windows):
        for query in range(tokens):
            if not valid[window][query]:
                allowed[window][query][0] = True

    mask = torch.tensor(allowed, dtype=torch.bool, device=q.device)
    valid_mask = torch.tensor(valid, dtype=q.dtype, device=q.device)
    query = q.reshape(batches, num_windows, heads, tokens, head_dim)
    key = k.reshape(batches, num_windows, heads, tokens, head_dim)
    value = v.reshape(batches, num_windows, heads, tokens, head_dim)
    result = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=mask.unsqueeze(0).unsqueeze(2),
        dropout_p=0.0, is_causal=False,
    )
    result = result * valid_mask.view(1, num_windows, 1, tokens, 1)
    return result.reshape(total, heads, tokens, head_dim)
