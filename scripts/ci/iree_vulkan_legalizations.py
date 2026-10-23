"""Exact grouped-SDPA scatter legalization for AOT GPU compilation.

The upstream reference writes attention results through a transposed view
using advanced indexed assignment. Re-express those same writes as index_copy
into contiguous (window, token, head, channel) storage. This is an import-time
lowering, not a substitute attention model; learned operations and grouping
remain unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_sdpa_contiguous(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_windows: int,
    full_idx: tuple[int, ...],
    groups: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> torch.Tensor:
    if num_windows <= 0:
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False,
        )
    if q.shape[0] % num_windows != 0:
        raise ValueError(
            f"cached window count {num_windows} does not divide attention "
            f"window batch {q.shape[0]}"
        )

    total_windows, heads, tokens, head_dim = q.shape
    batch_windows = total_windows // num_windows
    token_slots = torch.arange(tokens, dtype=torch.long, device=q.device)
    out_flat = q.new_zeros((total_windows * tokens, heads, head_dim))

    def batched_windows(base):
        indices = torch.tensor(base, device=q.device, dtype=torch.long)
        if batch_windows == 1 or indices.numel() == 0:
            return indices
        offsets = (torch.arange(
            batch_windows, device=q.device, dtype=torch.long
        ) * num_windows)
        return (indices.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

    def write_regions(window_index, token_index, result):
        # This is exactly the element map used by the original
        # out.permute(0,2,1,3)[win_idx[:,None],token_idx[None,:]] = attn.
        row_index = (
            window_index[:, None] * tokens + token_index[None, :]
        ).reshape(-1)
        rows = result.permute(0, 2, 1, 3).reshape(-1, heads, head_dim)
        out_flat.index_copy_(0, row_index, rows)

    if full_idx:
        windows = batched_windows(full_idx)
        result = F.scaled_dot_product_attention(
            q.index_select(0, windows),
            k.index_select(0, windows),
            v.index_select(0, windows),
            dropout_p=0.0, is_causal=False,
        )
        write_regions(windows, token_slots, result)

    for token_tuple, window_tuple in groups:
        windows = batched_windows(window_tuple)
        token_index = torch.tensor(
            token_tuple, device=q.device, dtype=torch.long,
        )
        result = F.scaled_dot_product_attention(
            q.index_select(0, windows).index_select(2, token_index),
            k.index_select(0, windows).index_select(2, token_index),
            v.index_select(0, windows).index_select(2, token_index),
            dropout_p=0.0, is_causal=False,
        )
        write_regions(windows, token_index, result)

    return out_flat.reshape(
        total_windows, tokens, heads, head_dim,
    ).permute(0, 2, 1, 3).contiguous()
