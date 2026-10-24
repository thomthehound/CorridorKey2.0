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
    wrap_wide_stage1_linears, restore_source_linears,
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


def check_nested_wide_linears():
    class NestedSource(nn.Module):
        def __init__(self):
            super().__init__()
            self.fuse = nn.Module()
            self.fuse.w_gate = nn.Linear(1536, 16)
            self.body_64 = nn.ModuleList([nn.Module()])
            self.body_64[0].attn = nn.Module()
            self.body_64[0].attn.qkv = nn.Module()
            self.body_64[0].attn.qkv.qkv = nn.Linear(1536, 4608)
            self.body_64[0].attn.attn_gate = nn.Module()
            self.body_64[0].attn.attn_gate.proj = nn.Linear(
                1536, 16, bias=False
            )
            self.body_64[0].attn.proj = nn.Linear(1536, 1536)
            self.body_64[0].small = nn.Linear(64, 64)

        def forward(self, x):
            attn = self.body_64[0].attn
            return (attn.qkv.qkv(x),
                    attn.attn_gate.proj(x),
                    attn.proj(x))

    source = NestedSource().eval()
    x = torch.randn(1, 1, 1, 1, 1536)
    saved_parameters = {
        name: param for name, param in source.named_parameters()
    }
    original_fuse = source.fuse.w_gate
    original_small = source.body_64[0].small
    original_modules = {
        name: module for name, module in source.named_modules()
        if type(module) is nn.Linear
    }
    with torch.inference_mode():
        expected = source(x)

    wrapped = wrap_wide_stage1_linears(source)
    try:
        assert len(wrapped) == 3, len(wrapped)
        assert source.fuse.w_gate is original_fuse
        assert source.body_64[0].small is original_small
        assert isinstance(source.body_64[0].attn.qkv.qkv, SegmentedSourceLinear)
        with torch.inference_mode():
            actual = source(x)
            exported = torch.export.export(source, (x,), strict=True)
            exported_values = exported.module()(x)
        for reference, result, cpu_export in zip(expected, actual, exported_values):
            torch.testing.assert_close(
                result, reference, atol=2e-5, rtol=2e-5
            )
            torch.testing.assert_close(
                cpu_export, reference, atol=2e-5, rtol=2e-5
            )
    finally:
        restore_source_linears(wrapped)

    for name, original in original_modules.items():
        assert dict(source.named_modules())[name] is original
    for name, param in saved_parameters.items():
        assert dict(source.named_parameters())[name] is param
    with torch.inference_mode():
        restored = source(x)
    for reference, value in zip(expected, restored):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


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
    check_nested_wide_linears()
    print("[Stage1 fusion] nested wide linears and source restoration passed",
          flush=True)
    print("[Stage1 fusion] source block and parameter restoration passed",
          flush=True)


if __name__ == "__main__":
    main()
