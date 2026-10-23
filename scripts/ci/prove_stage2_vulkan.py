#!/usr/bin/env python3
"""Export CorridorKey's complete Stage2 graph and validate Vulkan execution.

A reduced-width, deterministic source-model profile makes export tractable
on GitHub-hosted runners. It retains the *actual* Stage2 topology (8 blocks,
including Swin and CSwin; patchify, late RGB skip, depatchify), but it is
not a production checkpoint or a throughput benchmark.
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
import iree.turbine.aot as aot

from prove_iree_vulkan import (
    _read_device_result,
    _rms_norm_inference_math,
    _sdpa_inference_math,
)
from iree_vulkan_legalizations import grouped_sdpa_masked


class ChannelLastStage2Result(torch.nn.Module):
    """Expose Stage2's B,T,C,H,W tensor via a Vulkan-friendly B,T,H,W,C ABI.

    The authoritative source model remains unchanged. The caller reconstructs
    the original channel-first output before comparing or consuming it.
    """

    def __init__(self, stage2):
        super().__init__()
        self.stage2 = stage2

    def forward(self, rgb, baton):
        return self.stage2(rgb, baton).permute(0, 1, 3, 4, 2).contiguous()


def create_stage2():
    from corridorkey.stage2 import Stage2

    torch.manual_seed(1937)
    model = Stage2(
        d=48,
        d_baton=16,
        num_heads=4,
        num_blocks=8,
        num_final_blocks=2,
        window_size=4,
        mlp_mult=2.0,
        gradient_checkpoint=False,
        overlap_mult=1,
        local_mix_kernel=0,
        mlp_dw_kernel=0,
        qkv_dw_kernel=0,
        refine4_blocks=0,
    ).eval()

    # The source initializes output and residual projections to zero for
    # training stability. Activate them without changing module structure
    # so CI cannot pass by testing a trivial zero-output network.
    with torch.no_grad():
        for block in [*model.pre_blocks, *model.final_blocks]:
            if hasattr(block.attn, "proj"):
                torch.nn.init.normal_(block.attn.proj.weight, std=0.025)
                torch.nn.init.normal_(block.attn.proj.bias, std=0.005)
            torch.nn.init.normal_(block.mlp.w_down.weight, std=0.02)
            torch.nn.init.normal_(block.mlp.w_down.bias, std=0.003)
        torch.nn.init.normal_(model.patch_2.conv.weight, std=0.002)
        torch.nn.init.normal_(model.depatchify.linear.weight, std=0.01)
        torch.nn.init.normal_(model.depatchify.linear.bias, std=0.001)

    rgb = torch.randn(1, 1, 3, 32, 32)
    baton = torch.randn(1, 1, 16, 2, 2)
    with torch.inference_mode():
        reference = model(rgb, baton).detach().cpu().numpy()
    if reference.shape != (1, 1, 4, 32, 32):
        raise AssertionError(f"Unexpected Stage2 source output {reference.shape}")
    if not np.isfinite(reference).all() or reference.std() < 1e-4:
        raise AssertionError("Stage2 reference is non-finite or trivial")
    return ChannelLastStage2Result(model).eval(), rgb, baton, reference


def export_stage2(model, rgb, baton):
    import corridorkey.blocks as blocks

    # The region IDs depend only on Stage2's known token-grid geometry.
    # Evaluate the existing upstream geometry cache on CPU, then specialize
    # the *cache lookup*, not the model's attention or learned weights.
    original = blocks._shifted_region_groups_2d_cached
    source = model.stage2
    window = source.window_size
    height, width = rgb.shape[-2] // 16, rgb.shape[-1] // 16
    pad_h = ((height + window - 1) // window) * window
    pad_w = ((width + window - 1) // window) * window
    signatures = {}
    for block in [*source.pre_blocks, *source.final_blocks]:
        if hasattr(block, "shift_h") and (block.shift_h or block.shift_w):
            sig = (height, width, pad_h, pad_w, window,
                   block.shift_h, block.shift_w)
            signatures[sig] = original(*sig)

    if not signatures:
        raise AssertionError("Expected shifted Stage2 attention sites")

    def specialize(*args):
        if args not in signatures:
            raise ValueError(f"Unexpected Stage2 geometry signature {args}")
        return signatures[args]

    # In eval, RoPE tables depend on the shape profile, not on source
    # pixels. Bake the ORIGINAL implementation's CPU tables into module
    # buffers and lift them as read-only constants during export. This
    # avoids re-evaluating trig functions on the GPU for every attention
    # layer and retains the exact reference angle convention.
    from corridorkey.rope import RoPE2D

    original_rope = RoPE2D.get_cos_sin
    rope_shapes = ((height, width), (window, window),
                   (height, 1), (1, width))
    unique_rope_modules = {
        id(module): module for module in source.modules()
        if isinstance(module, RoPE2D)
    }
    if not unique_rope_modules:
        raise AssertionError("Stage2 RoPE modules are missing")
    for rope in unique_rope_modules.values():
        for h, w in rope_shapes:
            cos, sin = original_rope(rope, h, w, device=torch.device("cpu"))
            rope.register_buffer(f"_aot_cos_{h}_{w}", cos.detach().clone(),
                                 persistent=False)
            rope.register_buffer(f"_aot_sin_{h}_{w}", sin.detach().clone(),
                                 persistent=False)

    def static_rope(self, num_h, num_w, device=None, h_offset=0, w_offset=0):
        if (num_h, num_w) not in rope_shapes:
            raise ValueError(f"RoPE shape {(num_h, num_w)} not specialized")
        return (getattr(self, f"_aot_cos_{num_h}_{num_w}"),
                getattr(self, f"_aot_sin_{num_h}_{num_w}"))

    # Capture the authoritative CPU output before changing the temporary
    # import-time group-scatter operator. This is a differential lowering,
    # not a modification to the learned network.
    with torch.inference_mode():
        source_result = model(rgb, baton).detach().cpu().numpy()
    original_grouped = blocks._sdpa_with_region_groups
    blocks._shifted_region_groups_2d_cached = specialize
    blocks._sdpa_with_region_groups = grouped_sdpa_masked
    RoPE2D.get_cos_sin = static_rope
    try:
        with torch.inference_mode():
            lowered_result = model(rgb, baton).detach().cpu().numpy()
        np.testing.assert_allclose(
            lowered_result, source_result, atol=1e-6, rtol=1e-6,
            err_msg="Masked region attention changed source Stage2",
        )
        exported = torch.export.export(model, (rgb, baton), strict=True)
    finally:
        blocks._shifted_region_groups_2d_cached = original
        blocks._sdpa_with_region_groups = original_grouped
        RoPE2D.get_cos_sin = original_rope

    exported = exported.run_decompositions({
        torch.ops.aten.scaled_dot_product_attention.default: _sdpa_inference_math,
        torch.ops.aten.rms_norm.default: _rms_norm_inference_math,
    })
    bad = [
        str(n.target) for n in exported.graph.nodes
        if any(s in str(n.target) for s in
               ("scaled_dot_product_attention", "aten.rms_norm"))
    ]
    if bad:
        raise AssertionError(f"Stage2 graph includes unsupported operators: {bad}")
    return exported, len(signatures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--execute-vulkan", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = {
        "scope": "original complete Stage2 topology, reduced deterministic weights",
        "production_checkpoint": False,
        "shape": [1, 1, 4, 32, 32],
        "import": False,
        "compile": False,
        "vulkan_execute": False,
        "target": args.target,
    }
    try:
        model, rgb, baton, expected = create_stage2()
        np.save(args.output / "reference.npy", expected)
        exported, profile_count = export_stage2(model, rgb, baton)
        status["geometry_signatures"] = profile_count
        with torch.inference_mode():
            imported_output = exported.module()(rgb, baton).detach().cpu().numpy()
            imported_output = imported_output.transpose(0, 1, 4, 2, 3).copy()
        np.save(args.output / "exported_cpu_output.npy", imported_output)
        cpu_diff = np.abs(imported_output - expected)
        status["export_max_absolute_error"] = float(cpu_diff.max())
        status["export_mean_absolute_error"] = float(cpu_diff.mean())
        np.testing.assert_allclose(imported_output, expected,
                                   atol=2e-5, rtol=2e-5,
                                   err_msg="Source vs decomposed exported Stage2 mismatch")
        compiled = aot.export(exported)
        compiled.verify()
        compiled.save_mlir(args.output / "stage2.mlir")
        status["import"] = True
        print("[stage2] exported complete Stage2 graph to MLIR", flush=True)

        compiled.session.set_flags(f"--iree-vulkan-target={args.target}")
        compiled.session.set_flags("--iree-opt-level=O2")
        vmfb = args.output / "stage2.vmfb"
        compiled.compile(save_to=vmfb, target_backends=["vulkan-spirv"])
        if not vmfb.is_file() or vmfb.stat().st_size < 100:
            raise AssertionError("Stage2 VMFB missing")
        status["compile"] = True
        print("[stage2] compiled Stage2 to Vulkan", flush=True)

        if args.execute_vulkan:
            import iree.runtime as rt

            config = rt.Config("vulkan")
            module = rt.VmModule.copy_buffer(config.vm_instance, vmfb.read_bytes())
            program = rt.load_vm_module(module, config)
            output = program.main(
                rt.asdevicearray(config.device, rgb.numpy()),
                rt.asdevicearray(config.device, baton.numpy()),
            )
            if isinstance(output, (tuple, list)):
                if len(output) != 1:
                    raise AssertionError("Stage2 should return one output tensor")
                output = output[0]
            actual = _read_device_result(config, output)
            if actual.shape != (1, 1, 32, 32, 4):
                raise AssertionError(f"Unexpected Stage2 GPU ABI shape: {actual.shape}")
            actual = actual.transpose(0, 1, 4, 2, 3).copy()
            np.save(args.output / "vulkan_output.npy", actual)
            if actual.shape != expected.shape:
                raise AssertionError(f"shape mismatch {actual.shape} vs {expected.shape}")
            diff = np.abs(actual - expected)
            status["vulkan_max_absolute_error"] = float(diff.max())
            status["vulkan_mean_absolute_error"] = float(diff.mean())
            status["vulkan_p99_absolute_error"] = float(np.percentile(diff, 99))
            status["vulkan_channel_max_absolute_errors"] = [
                float(diff[:, :, ch].max()) for ch in range(diff.shape[2])
            ]
            failed = np.argwhere(~np.isclose(actual, expected, rtol=3e-3, atol=3e-3))
            if len(failed):
                print(f"[stage2] {len(failed)} mismatched output values, "
                      f"max={diff.max():.7g} mean={diff.mean():.7g}", flush=True)
                for position in failed[:16]:
                    idx = tuple(map(int, position))
                    print(f"  {idx}: Vulkan={actual[idx]:.8g}, "
                          f"source={expected[idx]:.8g}, "
                          f"CPU-export={imported_output[idx]:.8g}", flush=True)
            np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)
            status["max_absolute_error"] = float(np.max(np.abs(actual - expected)))
            status["vulkan_execute"] = True
            print("[stage2] Vulkan output numerically matches complete Stage2 source",
                  flush=True)
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        (args.output / "report.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if "error" in status else 0


if __name__ == "__main__":
    raise SystemExit(main())
