"""Source-equivalent segmented GEMM lowering for Stage1 downsampling.

This export-only module preserves trained convolution weights, stride, padding,
dilation, and bias. It tiles the reduction axis for Vulkan compilers that
miscompile wide floating-point accumulations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentedStage1DownConv(nn.Module):
    """Lower a Conv2d to im2col + batched 128-wide dot products.

    The source Conv2d is registered as a child and retains ownership of
    every trained parameter. Only its inference computation is transformed.
    The implementation handles arbitrary positive H/W and output grid sizes.
    """

    def __init__(self, conv: nn.Conv2d, segment: int = 128):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("Expected the source Stage1 Conv2d")
        if conv.groups != 1 or conv.padding_mode != "zeros":
            raise ValueError("Segmented convolution requires groups=1, zero padding")
        if segment < 1 or segment > 128:
            raise ValueError("Segmented convolution width must be within [1, 128]")
        self.source_conv = conv
        self.segment = int(segment)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.source_conv
        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        ph, pw = conv.padding
        dh, dw = conv.dilation
        oh = (x.shape[-2] + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (x.shape[-1] + 2 * pw - dw * (kw - 1) - 1) // sw + 1

        padded = F.pad(x, (pw, pw, ph, ph))
        taps = [
            padded[
                :, :,
                ky * dh:ky * dh + oh * sh:sh,
                kx * dw:kx * dw + ow * sw:sw,
            ]
            for ky in range(kh)
            for kx in range(kw)
        ]
        # Stack tap-major, then flatten channel-major to match Conv2d weights
        # (output_channel, input_channel, kernel_y, kernel_x).
        patches = torch.stack(taps, dim=2).permute(0, 3, 4, 1, 2).reshape(
            x.shape[0] * oh * ow, -1
        )
        weights = conv.weight.flatten(1)
        width = weights.shape[1]
        segments = (width + self.segment - 1) // self.segment
        remaining = segments * self.segment - width
        if remaining:
            patches = F.pad(patches, (0, remaining))
            weights = F.pad(weights, (0, remaining))

        lhs = patches.reshape(
            patches.shape[0], segments, self.segment
        ).transpose(0, 1)
        rhs = weights.reshape(
            conv.out_channels, segments, self.segment
        ).permute(1, 2, 0)
        partials = torch.bmm(lhs, rhs)

        # Keep every intermediate reduction at 128 or fewer terms.
        while partials.shape[0] > 128:
            n = partials.shape[0]
            blocks = (n + 127) // 128
            excess = blocks * 128 - n
            if excess:
                partials = F.pad(partials, (0, 0, 0, 0, 0, excess))
            partials = partials.reshape(
                blocks, 128, partials.shape[1], conv.out_channels
            ).sum(dim=1)
        values = partials.sum(dim=0)
        if conv.bias is not None:
            values = values + conv.bias
        return values.reshape(
            x.shape[0], oh, ow, conv.out_channels
        ).permute(0, 3, 1, 2).contiguous()
