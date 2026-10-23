"""Source-equivalent grouped-SDPA lowering for Vulkan AOT compilation.

The original attention outputs are computed on exactly the same per-region
Q/K/V tensors. Reassembly uses a compile-time geometry routing table and
a contiguous gather instead of indexed writes through a tensor view.

GPU working storage is linear in the number of output tokens, not quadratic
in the attention window size. No model weights or group semantics change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F



def headwise_gate_contiguous(self, query_tokens: torch.Tensor,
                             attn: torch.Tensor) -> torch.Tensor:
    """Exact gate applied in token/head order, avoiding strided vector writes.

    Source computes the same sigmoid gate and applies it to each attention
    head through a heads-first view. Token-major scaling provides an
    equivalent layout for the compiler, restoring source ordering afterward.
    """
    gate = torch.sigmoid(self.proj(query_tokens)).unsqueeze(-1)
    return (attn.transpose(-3, -2) * gate).transpose(-3, -2).contiguous()


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

    # Route positions are source-geometry constants for a compiled shape
    # profile. The initial full-window zero region represents unassigned
    # padding tokens. Using a regular window-sized allocation also avoids
    # an irregular one-row vector constant in the Vulkan software backend.
    route = [0] * (total_windows * tokens)
    pieces = [q.new_zeros((tokens, heads, head_dim))]
    cursor = tokens

    def windows_for(base: tuple[int, ...]):
        # Window-major ordering is identical to _batched_windows upstream.
        window_ids = [
            batch * num_windows + window
            for batch in range(batch_windows)
            for window in base
        ]
        return (
            window_ids,
            torch.tensor(window_ids, device=q.device, dtype=torch.long),
        )

    def add_output(window_ids, token_ids, result):
        nonlocal cursor
        rows = result.permute(0, 2, 1, 3).reshape(-1, heads, head_dim)
        pieces.append(rows)
        for window in window_ids:
            for token in token_ids:
                route[window * tokens + token] = cursor
                cursor += 1

    if full_idx:
        window_ids, indices = windows_for(full_idx)
        result = F.scaled_dot_product_attention(
            q.index_select(0, indices),
            k.index_select(0, indices),
            v.index_select(0, indices),
            dropout_p=0.0, is_causal=False,
        )
        add_output(window_ids, range(tokens), result)

    for token_tuple, window_tuple in groups:
        window_ids, window_index = windows_for(window_tuple)
        token_index = torch.tensor(
            token_tuple, device=q.device, dtype=torch.long,
        )
        result = F.scaled_dot_product_attention(
            q.index_select(0, window_index).index_select(2, token_index),
            k.index_select(0, window_index).index_select(2, token_index),
            v.index_select(0, window_index).index_select(2, token_index),
            dropout_p=0.0, is_causal=False,
        )
        add_output(window_ids, token_tuple, result)

    packed = torch.cat(pieces, dim=0)
    output_rows = packed.index_select(
        0, torch.tensor(route, device=q.device, dtype=torch.long),
    )
    return output_rows.reshape(
        total_windows, tokens, heads, head_dim,
    ).permute(0, 2, 1, 3).contiguous()
