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
    return model, rgb, baton, reference


def export_stage2(model, rgb, baton):
    import corridorkey.blocks as blocks

    # The region IDs depend only on Stage2's known token-grid geometry.
    # Evaluate the existing upstream geometry cache on CPU, then specialize
    # the *cache lookup*, not the model's attention or learned weights.
    original = blocks._shifted_region_groups_2d_cached
    window = model.window_size
    height, width = rgb.shape[-2] // 16, rgb.shape[-1] // 16
    pad_h = ((height + window - 1) // window) * window
    pad_w = ((width + window - 1) // window) * window
    signatures = {}
    for block in [*model.pre_blocks, *model.final_blocks]:
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

    blocks._shifted_region_groups_2d_cached = specialize
    try:
        exported = torch.export.export(model, (rgb, baton), strict=True)
    finally:
        blocks._shifted_region_groups_2d_cached = original

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
            np.save(args.output / "vulkan_output.npy", actual)
            if actual.shape != expected.shape:
                raise AssertionError(f"shape mismatch {actual.shape} vs {expected.shape}")
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
