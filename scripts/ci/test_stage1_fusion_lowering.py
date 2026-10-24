#!/usr/bin/env python3
"""Check that segmented Stage1 fusion preserves source weights and arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from corridorkey.blocks import FusionLinSwiglu
from stage1_linear_lowering import (
    FUSION_PROJECTIONS, SegmentedSourceLinear, wrap_fusion_projections,
)


def check_linear(width: int, outputs: int):
    source = nn.Linear(width, outputs).eval()
    x = torch.randn(1, 1, 1, 1, width)
    with torch.inference_mode():
        expected = source(x)
        wrapped = SegmentedSourceLinear(source).eval()
        actual = wrapped(x)
        exported = torch.export.export(wrapped, (x,), strict=True).module()(x)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(exported, expected, atol=2e-5, rtol=2e-5)


def check_source_fusion():
    model = FusionLinSwiglu(
        vfm_dim=384, aux_dim=128, out_dim=256, hidden=768
    ).eval()
    x = torch.randn(1, 1, 1, 1, 512)
    keys = tuple(model.state_dict().keys())
    weights = tuple((name, getattr(model, name)) for name in FUSION_PROJECTIONS)
    with torch.inference_mode():
        reference = model(x)
    saved = wrap_fusion_projections(model)
    try:
        with torch.inference_mode():
            actual = model(x)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
        exported = torch.export.export(model, (x,), strict=True)
        with torch.inference_mode():
            result = exported.module()(x)
        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    finally:
        for name, original in saved:
            setattr(model, name, original)
    assert tuple(model.state_dict().keys()) == keys
    assert all(getattr(model, name) is original for name, original in weights)
    with torch.inference_mode():
        restored = model(x)
    torch.testing.assert_close(restored, reference, atol=0, rtol=0)


def main():
    torch.manual_seed(20261024)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    for width, outputs in (
        (129, 17), (512, 256), (2816, 1536),
        (2816, 6080), (6080, 1536), (16513, 13),
    ):
        check_linear(width, outputs)
        print(f"[Stage1 fusion] linear {width}->{outputs}: CPU/export passed",
              flush=True)
    check_source_fusion()
    print("[Stage1 fusion] source block and parameter restoration passed",
          flush=True)


if __name__ == "__main__":
    main()
