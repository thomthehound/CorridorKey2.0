#!/usr/bin/env python3
"""Source-derived complete Stage1 temporal graph Vulkan proof.

The fixture instantiates CorridorKey's *original* Stage1 class with reduced
channel widths and two frames. Frozen foundation networks are not replaced:
Stage1's documented normalized feature tensor input is supplied explicitly.
This is not trained checkpoint execution and does not cover recurrent sites.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from prove_iree_vulkan import (
    _read_device_result, _rms_norm_inference_math, _sdpa_inference_math,
)


class FlattenedStage1RMSNorm(torch.nn.Module):
    """Preserve the source RMSNorm on a two-dimensional token-major view.

    Normalization is strictly over the last axis in the original class,
    so flattening leading dimensions is an exact, source-weight-preserving
    tensor layout legalization. It avoids high-rank RMSNorm fusion during
    Vulkan code generation.
    """

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        shape = x.shape
        return self.source(x.reshape(-1, shape[-1])).reshape(shape)


class Stage1OutputABI(torch.nn.Module):
    """Expose unchanged source baton and aux as channel-last GPU buffers."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, rgb, hint, moge, hint_quality):
        result = self.source(
            rgb, hint, moge, hint_quality=hint_quality, t_stride=1.0,
        )
        return (
            result["baton"].permute(0, 1, 3, 4, 2).contiguous(),
            result["aux"].permute(0, 1, 3, 4, 2).contiguous(),
        )


def create_stage1():
    from corridorkey.stage1 import Stage1

    torch.manual_seed(20261023)
    model = Stage1(
        d_16=64, d_64=128, cradio_dim=0, moge_dim=24, rvm_dim=0,
        n_enc_16=1, n_body_64=2, n_dec_16=6,
        num_heads_16=4, num_heads_64=8,
        window_HW=4, local_mix_kernel=0, body64_local_mix_kernel=0,
        hint_adaln_blocks=1, mid_16_blocks=2,
        recurrent_64_sites=(),
        gradient_checkpoint=False, mlp_mult=2.0,
        mlp_dw_kernel=0, qkv_dw_kernel=0,
        overlap_mult=1, bottleneck_head_stride=None,
    ).eval()
    with torch.no_grad():
        # Make zero-initialized attention/MLP outputs nontrivial to prevent
        # successful zero-branch execution from hiding bad GPU attention.
        for item in model.modules():
            if hasattr(item, "attn") and hasattr(item.attn, "proj"):
                torch.nn.init.normal_(item.attn.proj.weight, std=0.025)
                if item.attn.proj.bias is not None:
                    torch.nn.init.normal_(item.attn.proj.bias, std=0.003)
            if hasattr(item, "mlp") and hasattr(item.mlp, "w_down"):
                torch.nn.init.normal_(item.mlp.w_down.weight, std=0.015)
                if item.mlp.w_down.bias is not None:
                    torch.nn.init.normal_(item.mlp.w_down.bias, std=0.002)
    rgb = torch.randn(1, 2, 3, 64, 64)
    hint = torch.rand(1, 2, 1, 64, 64)
    moge = torch.randn(1, 2, 24, 1, 1)
    hint_quality = torch.tensor([[1.0, -1.0]])
    args = (rgb, hint, moge, hint_quality)
    with torch.inference_mode():
        reference = model(*args[:3], hint_quality=hint_quality)
    if set(reference).difference({"baton", "aux", "bottleneck"}) or (
        reference["baton"].shape != (1, 2, 64, 4, 4)
        or reference["aux"].shape != (1, 2, 4, 32, 32)
        or not np.isfinite(reference["baton"].numpy()).all()
        or not np.isfinite(reference["aux"].numpy()).all()
    ):
        raise AssertionError("Stage1 source output contract mismatch")
    expected = (
        reference["baton"].detach().numpy(),
        reference["aux"].detach().numpy(),
    )
    return Stage1OutputABI(model).eval(), args, expected


def export_stage1(model, args):
    import corridorkey.blocks as blocks

    # Evaluate the source's static shifted-region cache on CPU for this
    # shape profile. Never synthesize grouping from an alternative model.
    original = blocks._shifted_region_groups_3d_cached
    rgb = args[0]
    _, T, _, H, W = rgb.shape
    token_h, token_w = H // 16, W // 16
    signatures = {}
    for block in (*model.source.mid_16, *model.source.dec_16):
        if not isinstance(block, blocks.SwinBlock3D):
            continue
        if not (block.shift_h or block.shift_w):
            continue
        wh, ww = block.window_H, block.window_W
        padded_h = ((token_h + wh - 1) // wh) * wh
        padded_w = ((token_w + ww - 1) // ww) * ww
        sig = (T, token_h, token_w, padded_h, padded_w,
               wh, ww, block.shift_h, block.shift_w)
        signatures[sig] = original(*sig)
    if not signatures:
        raise AssertionError("Stage1 source is missing shifted temporal windows")

    def fixed(*signature):
        if signature not in signatures:
            raise ValueError(f"Unexpected Stage1 shifted region {signature}")
        return signatures[signature]

    # Replace only the *presentation* of last-axis RMSNorm during export.
    # The original modules/weights remain owned by the source class and
    # are restored immediately after graph capture.
    with torch.inference_mode():
        authoritative = tuple(t.detach().numpy().copy() for t in model(*args))
    saved = []
    for parent in list(model.named_modules()):
        module = parent[1]
        for name, child in list(module.named_children()):
            if isinstance(child, blocks.RMSNorm):
                saved.append((module, name, child))
                setattr(module, name, FlattenedStage1RMSNorm(child))
    if not saved:
        raise AssertionError("Expected source Stage1 RMSNorm layers")
    blocks._shifted_region_groups_3d_cached = fixed
    try:
        with torch.inference_mode():
            flattened = tuple(t.detach().numpy() for t in model(*args))
        for index, (reference, actual) in enumerate(zip(authoritative, flattened)):
            np.testing.assert_allclose(
                actual, reference, atol=0.0, rtol=0.0,
                err_msg=f"Flattened Stage1 RMSNorm changed source output {index}",
            )
        exported = torch.export.export(model, args, strict=True)
    finally:
        for parent, name, original_norm in saved:
            setattr(parent, name, original_norm)
        blocks._shifted_region_groups_3d_cached = original

    exported = exported.run_decompositions({
        torch.ops.aten.scaled_dot_product_attention.default:
            _sdpa_inference_math,
        torch.ops.aten.rms_norm.default:
            _rms_norm_inference_math,
    })
    unconverted = [
        str(node.target) for node in exported.graph.nodes
        if "scaled_dot_product_attention" in str(node.target)
        or "aten.rms_norm" in str(node.target)
    ]
    if unconverted:
        raise AssertionError(f"Unconverted Stage1 operators: {unconverted}")
    return exported, len(signatures)


def check_results(actual, expected, prefix, atol, rtol):
    if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
        raise AssertionError(f"{prefix}: expected baton and aux outputs")
    metrics = []
    for index, (result, reference) in enumerate(zip(actual, expected)):
        if result.ndim != 5:
            raise AssertionError(f"{prefix} output {index} must be five-dimensional")
        cpu = result.transpose(0, 1, 4, 2, 3).copy()
        if cpu.shape != reference.shape or not np.isfinite(cpu).all():
            raise AssertionError(f"{prefix} output {index} shape/finiteness mismatch")
        diff = np.abs(cpu - reference)
        metrics.append({"output": index, "maximum": float(diff.max()),
                        "mean": float(diff.mean())})
        np.testing.assert_allclose(cpu, reference, atol=atol, rtol=rtol,
                                   err_msg=f"{prefix} output {index}")
    return metrics


def run(args):
    report = {"scope": "original Stage1; reduced untrained two-frame profile",
              "checkpoint": False, "recurrent_sites": False,
              "import": False, "compile": False, "vulkan_execute": False}
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        model, samples, expected = create_stage1()
        report["source"] = True
        for i, value in enumerate(expected):
            np.save(args.output / f"source_{i}.npy", value)

        exported, signatures = export_stage1(model, samples)
        report["region_signatures"] = signatures
        with torch.inference_mode():
            cpu = tuple(t.detach().numpy() for t in exported.module()(*samples))
        report["export_errors"] = check_results(cpu, expected, "CPU export",
                                                 atol=2e-5, rtol=2e-5)
        report["import"] = True
        print("[stage1] exported complete source graph and matched CPU", flush=True)

        if args.compile:
            import iree.turbine.aot as aot
            graph = aot.export(exported)
            graph.verify()
            graph.save_mlir(args.output / "stage1.mlir")
            graph.session.set_flags(f"--iree-vulkan-target={args.target}")
            graph.session.set_flags("--iree-opt-level=O2")
            vmfb = args.output / "stage1.vmfb"
            graph.compile(save_to=vmfb, target_backends=["vulkan-spirv"])
            if not vmfb.exists() or vmfb.stat().st_size < 100:
                raise AssertionError("Missing compiled Stage1 VMFB")
            report["compile"] = True
            print("[stage1] compiled complete source graph to Vulkan", flush=True)
            if args.execute_vulkan:
                import iree.runtime as rt
                config = rt.Config("vulkan")
                module = rt.VmModule.copy_buffer(
                    config.vm_instance, vmfb.read_bytes(),
                )
                program = rt.load_vm_module(module, config)
                inputs = [rt.asdevicearray(config.device, x.numpy())
                          for x in samples]
                outputs = program.main(*inputs)
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                actual = tuple(_read_device_result(config, o) for o in outputs)
                report["vulkan_errors"] = check_results(
                    actual, expected, "Vulkan Stage1", atol=3e-3, rtol=3e-3,
                )
                report["vulkan_execute"] = True
                print("[stage1] complete source graph matches software Vulkan",
                      flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    return 1 if "error" in report else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="llvmpipe")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--execute-vulkan", action="store_true")
    args = parser.parse_args()
    if args.execute_vulkan and not args.compile:
        parser.error("--execute-vulkan requires --compile")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
