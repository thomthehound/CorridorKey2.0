"""Source-equivalent segmented linear operators for Stage1 Vulkan export.

The original nn.Linear owns every trained parameter. The wrapper changes only
the computation's reduction grouping during export. No scaling correction,
parameter conversion, or alternative learned operation is introduced.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentedSourceLinear(nn.Module):
    """Represent source F.linear as batched at-most-128-wide dot products."""

    def __init__(self, source: nn.Linear, segment: int = 128):
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError("Expected a source nn.Linear")
        if not 1 <= segment <= 128:
            raise ValueError("Segment size must be in [1, 128]")
        self.source_linear = source
        self.segment = int(segment)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = self.source_linear
        flat = x.reshape(-1, source.in_features)
        weight = source.weight
        width = source.in_features
        segments = (width + self.segment - 1) // self.segment
        padded_width = segments * self.segment
        if padded_width != width:
            flat = F.pad(flat, (0, padded_width - width))
            weight = F.pad(weight, (0, padded_width - width))

        lhs = flat.reshape(
            flat.shape[0], segments, self.segment
        ).transpose(0, 1)
        rhs = weight.reshape(
            source.out_features, segments, self.segment
        ).permute(1, 2, 0)
        partials = torch.bmm(lhs, rhs)

        # Hierarchical sum avoids recreating a wide reduction for very large
        # linear layers. Bias is added once, after the entire dot product.
        while partials.shape[0] > 128:
            n = partials.shape[0]
            blocks = (n + 127) // 128
            excess = blocks * 128 - n
            if excess:
                partials = F.pad(partials, (0, 0, 0, 0, 0, excess))
            partials = partials.reshape(
                blocks, 128, partials.shape[1], source.out_features
            ).sum(dim=1)
        values = partials.sum(dim=0)
        if source.bias is not None:
            values = values + source.bias
        return values.reshape(*x.shape[:-1], source.out_features)


FUSION_PROJECTIONS = ("linear_shortcut", "w_gate", "w_up", "w_down")


def wrap_fusion_projections(fuse: nn.Module):
    """Temporarily substitute only the original fusion's four nn.Linears.

    Caller is responsible for restoring the returned (name, source) pairs.
    The source module itself is not replaced or reconstructed.
    """
    if fuse is None:
        raise ValueError("Stage1 has no learned fusion module")
    originals = []
    try:
        for name in FUSION_PROJECTIONS:
            source = getattr(fuse, name)
            if not isinstance(source, nn.Linear):
                raise TypeError(f"Unexpected Stage1 fusion projection: {name}")
            originals.append((name, source))
            setattr(fuse, name, SegmentedSourceLinear(source))
    except BaseException:
        for name, source in originals:
            setattr(fuse, name, source)
        raise
    return originals
