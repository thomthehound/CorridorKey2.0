#!/usr/bin/env python3
"""Pinpoint the first divergent tensor in a compiled Stage2 Vulkan graph.

Diagnostics operate on the exact exported Stage2 graph: intermediate values
are added as outputs to an FX graph, then the resulting graph is re-exported.
No learned operation is replaced and no numerical tolerance is widened.
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
from torch.utils import _pytree as pytree
import iree.turbine.aot as aot

from prove_iree_vulkan import _read_device_result
from prove_stage2_vulkan import create_stage2, export_stage2


# Each path refers to a module in the unchanged upstream Stage2 topology.
# All outputs are structural tensor boundaries, not independent test models.
SYNTHETIC_SITES = (
    ("patchify", "patch_1", (1, 32, 2, 2)),
    # A block's final residual addition can occur *outside* its tagged
    # submodule operations. Inspect the input to the next block's first
    # normalization instead: this is the actual inter-block tensor.
    ("input_pre_0", "input:pre_blocks.0.norm1", (1, 2, 2, 48)),
    ("input_pre_1", "input:pre_blocks.1.norm1", (1, 2, 2, 48)),
    ("input_pre_2", "input:pre_blocks.2.norm1", (1, 2, 2, 48)),
    ("input_pre_3", "input:pre_blocks.3.norm1", (1, 2, 2, 48)),
    ("input_pre_4", "input:pre_blocks.4.norm1", (1, 2, 2, 48)),
    ("input_pre_5", "input:pre_blocks.5.norm1", (1, 2, 2, 48)),
    ("input_final_0", "input:final_blocks.0.norm1", (1, 2, 2, 48)),
    ("input_final_1", "input:final_blocks.1.norm1", (1, 2, 2, 48)),
    ("pre_block_0", "pre_blocks.0", (1, 2, 2, 48)),
    ("pre_block_1", "pre_blocks.1", (1, 2, 2, 48)),
    ("pre_block_2_norm1", "pre_blocks.2.norm1", (1, 2, 2, 48)),
    ("pre_block_2_attention", "pre_blocks.2.attn", (1, 2, 2, 48)),
    ("pre_block_2", "pre_blocks.2", (1, 2, 2, 48)),
    ("pre_block_3", "pre_blocks.3", (1, 2, 2, 48)),
    ("pre_block_4", "pre_blocks.4", (1, 2, 2, 48)),
    ("pre_block_5", "pre_blocks.5", (1, 2, 2, 48)),
    ("final_block_0", "final_blocks.0", (1, 2, 2, 48)),
    ("final_block_1", "final_blocks.1", (1, 2, 2, 48)),
    ("depatchify", "depatchify", (1, 4, 32, 32)),
)


def trained_sites(stage2, rgb):
    """Stage2 source-module probes sized from the checkpoint configuration."""
    batch_frames = rgb.shape[0] * rgb.shape[1]
    height, width = rgb.shape[-2] // 16, rgb.shape[-1] // 16
    channels = stage2.d
    token_shape = (batch_frames, height, width, channels)
    patch_shape = (batch_frames, stage2.d_rgb, height, width)
    sites = [("patchify", "patch_1", patch_shape)]
    sites.extend(
        (f"input_pre_{index}", f"input:pre_blocks.{index}.norm1",
         token_shape)
        for index in range(len(stage2.pre_blocks))
    )
    if stage2.use_patch_2:
        sites.append(("patch2", "patch_2", patch_shape))
    sites.extend(
        (f"input_final_{index}", f"input:final_blocks.{index}.norm1",
         token_shape)
        for index in range(len(stage2.final_blocks))
    )
    sites.append(
        ("depatchify", "depatchify",
         (batch_frames, 4, rgb.shape[-2], rgb.shape[-1]))
    )
    return tuple(sites)



def first_block_sites(stage2, rgb):
    """Expose the operators inside the first trained full-spatial block.

    Probe only unchanged FX graph tensors, without replacing any learned
    operation. Module-path and tensor-shape matching is checked strictly.
    """
    block = stage2.pre_blocks[0]
    from corridorkey.blocks import FullSpatialBlock2D
    if not isinstance(block, FullSpatialBlock2D):
        raise ValueError("Stage2 pre-block 0 is not FullSpatialBlock2D")
    bt = rgb.shape[0] * rgb.shape[1]
    height, width = rgb.shape[-2] // 16, rgb.shape[-1] // 16
    width_channels = stage2.d
    token_shape = (bt, height, width, width_channels)
    tokens = height * width
    heads = block.attn.num_heads
    head_width = block.attn.head_dim
    prefix = "pre_blocks.0"
    sites = [
        ("input_pre_0", f"input:{prefix}.norm1", token_shape),
        ("pre0_norm1", f"{prefix}.norm1", token_shape),
        ("pre0_qkv_linear", f"{prefix}.attn.qkv.qkv",
         (bt, tokens, 3 * width_channels)),
        ("pre0_q_norm", f"{prefix}.attn.q_norm",
         (bt, heads, tokens, head_width)),
        ("pre0_k_norm", f"{prefix}.attn.k_norm",
         (bt, heads, tokens, head_width)),
        ("pre0_attention_gate", f"{prefix}.attn.attn_gate",
         (bt, heads, tokens, head_width)),
        ("pre0_attention_proj", f"{prefix}.attn.proj",
         (bt, tokens, width_channels)),
        ("pre0_attention", f"{prefix}.attn",
         (bt, 1, height, width, width_channels)),
    ]
    if block.local_mix is not None:
        sites.extend([
            ("pre0_local_norm", f"{prefix}.local_mix.norm", token_shape),
            ("pre0_local_conv", f"{prefix}.local_mix",
             (bt, 2 * width_channels, height, width)),
            ("pre0_local_mix", f"{prefix}.local_mix", token_shape),
        ])
    sites.extend([
        ("pre0_norm2", f"{prefix}.norm2", token_shape),
        ("pre0_mlp_gate", f"{prefix}.mlp.w_gate",
         (bt, height, width, block.mlp.hidden)),
        ("pre0_mlp_up", f"{prefix}.mlp.w_up",
         (bt, height, width, block.mlp.hidden)),
        ("pre0_mlp", f"{prefix}.mlp", token_shape),
        ("input_pre_1", "input:pre_blocks.1.norm1", token_shape),
    ])
    return tuple(sites)


def _stack_paths(node):
    stack = node.meta.get("nn_module_stack") or {}
    return [
        (entry[0] if isinstance(entry, (tuple, list)) else str(entry))
        for entry in stack.values()
    ]


def expose_intermediates(exported, sites=SYNTHETIC_SITES):
    gm = exported.module()
    diagnostics = []

    for name, path, shape in sites:
        candidates = []
        is_block_input = path.startswith("input:")
        if is_block_input:
            path = path[len("input:"):]
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if not any(path in p for p in _stack_paths(node)):
                continue
            if is_block_input:
                # The first RMSNorm op is square(x). Its operand x is the
                # exact tensor passed into the block, after prior residuals.
                if "square" not in str(node.target):
                    continue
                candidate = node.args[0]
                if not isinstance(candidate, torch.fx.Node):
                    continue
            else:
                candidate = node
            value = candidate.meta.get("val")
            if not hasattr(value, "shape") or tuple(value.shape) != shape:
                continue
            candidates.append(candidate)
        if not candidates:
            available = [
                (node.name, tuple(node.meta["val"].shape),
                 str(node.target), _stack_paths(node))
                for node in gm.graph.nodes
                if hasattr(node.meta.get("val"), "shape")
                and any(path.rsplit(".", 1)[0] in p for p in _stack_paths(node))
            ]
            raise RuntimeError(
                f"No exported tensor at {name} ({path}, shape {shape}). "
                f"Nearby graph nodes: {available[:30]}"
            )
        diagnostics.append((name, candidates[-1]))

    out = next(node for node in gm.graph.nodes if node.op == "output")
    previous = out.args[0]
    if not isinstance(previous, (list, tuple)):
        previous = (previous,)
    output_nodes = tuple(previous) + tuple(node for _, node in diagnostics)
    out.args = (output_nodes,)
    gm.graph.lint()
    gm.recompile()
    # Recompile resets the output tree spec from the original module.
    gm._out_spec = pytree.tree_flatten(tuple(range(len(output_nodes))))[1]
    return gm, [name for name, _ in diagnostics], len(previous)


def run(directory: Path, target: str, *, checkpoint=None,
        height: int = 64, width: int = 64, probe_only: bool = False,
        first_block: bool = False):
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": ("trained checkpoint Stage2" if checkpoint is not None
                  else "reduced source-derived Stage2"),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "target": target,
        "first_block": first_block,
        "image_shape": [height, width] if checkpoint is not None else [32, 32],
        "samples": [],
        "stage": "start",
    }
    try:
        if checkpoint is None:
            # A ten-block Stage2 starts with FullSpatialBlock2D; the eight-
            # block fixture starts with Swin. Match the production block
            # family for the focused probes without changing default tests.
            source, rgb, baton, reference = create_stage2(
                num_blocks=10 if first_block else 8
            )
            reference_output = (reference.transpose(0, 1, 3, 4, 2).copy(),)
            sites = (first_block_sites(source.stage2, rgb)
                     if first_block else SYNTHETIC_SITES)
        else:
            from prove_checkpoint_stage2 import load_checkpoint_stage2
            from prove_stage2_vulkan import ChannelLastStage2Result
            torch.manual_seed(1942)
            stage2, config, manifest, count = load_checkpoint_stage2(checkpoint)
            report["state_tensor_count"] = count
            report["model_config"] = manifest["architecture_config"]
            rgb = torch.randn(1, 1, 3, height, width)
            baton = torch.randn(
                1, 1, config["d_16"], height // 16, width // 16,
            )
            with torch.inference_mode():
                source_result = stage2(rgb, baton)
            reference_output = (
                source_result if isinstance(source_result, (tuple, list))
                else (source_result,)
            )
            reference_output = tuple(
                t.detach().numpy().transpose(0, 1, 3, 4, 2).copy()
                for t in reference_output
            )
            source = ChannelLastStage2Result(stage2).eval()
            sites = (first_block_sites(stage2, rgb)
                     if first_block else trained_sites(stage2, rgb))

        report["sites"] = [name for name, _, _ in sites]
        exported, _ = export_stage2(source, rgb, baton)
        gm, names, final_count = expose_intermediates(exported, sites)
        if final_count != len(reference_output):
            raise AssertionError(
                f"Model has {len(reference_output)} source outputs, "
                f"diagnostic graph has {final_count}"
            )
        with torch.inference_mode():
            expected = gm(rgb, baton)
        if len(expected) != len(names) + final_count:
            raise AssertionError("Diagnostic output count mismatch")
        report["stage"] = "probe-selected"

        for index, (actual, reference) in enumerate(
            zip(expected[:final_count], reference_output)
        ):
            np.testing.assert_allclose(
                actual.detach().cpu().numpy(), reference,
                rtol=2e-5, atol=2e-5,
                err_msg=f"Instrumented CPU graph changed source output {index}",
            )

        reexport = torch.export.export(gm, (rgb, baton), strict=True)
        with torch.inference_mode():
            cpu_exported = reexport.module()(rgb, baton)
        for name, a, b in zip(
            (*[f"final_{i}" for i in range(final_count)], *names),
            expected, cpu_exported,
        ):
            np.testing.assert_allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(),
                rtol=2e-5, atol=2e-5,
                err_msg=f"Diagnostic re-export changed {name}",
            )
        report["stage"] = "cpu-validated"
        print(
            f"[stage2 diagnostic] {len(names)} probe sites; "
            f"{final_count} final output(s); CPU differential passed",
            flush=True,
        )
        if probe_only:
            return 0

        compiled = aot.export(reexport)
        compiled.verify()
        compiled.save_mlir(directory / "stage2_diagnostic.mlir")
        compiled.session.set_flags(f"--iree-vulkan-target={target}")
        compiled.session.set_flags("--iree-opt-level=O2")
        vmfb = directory / "stage2_diagnostic.vmfb"
        compiled.compile(save_to=vmfb, target_backends=["vulkan-spirv"])
        report["stage"] = "compiled"
        print("[stage2 diagnostic] compiled instrumented Stage2", flush=True)

        import iree.runtime as rt
        cfg = rt.Config("vulkan")
        module = rt.VmModule.copy_buffer(cfg.vm_instance, vmfb.read_bytes())
        program = rt.load_vm_module(module, cfg)
        outputs = program.main(
            rt.asdevicearray(cfg.device, rgb.numpy()),
            rt.asdevicearray(cfg.device, baton.numpy()),
        )
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        if len(outputs) != len(expected):
            raise AssertionError(
                f"Expected {len(expected)} result buffers, received {len(outputs)}"
            )
        report["stage"] = "executed"
        compare = (
            [(name, a, b) for name, a, b in
             zip(names, expected[final_count:], outputs[final_count:])]
            + [(f"final_{i}", expected[i], outputs[i])
               for i in range(final_count)]
        )
        first_mismatch = None
        for name, source_value, device_value in compare:
            cpu = source_value.detach().cpu().numpy()
            gpu = _read_device_result(cfg, device_value)
            if cpu.shape != gpu.shape:
                raise AssertionError(
                    f"{name}: shape mismatch CPU {cpu.shape}, Vulkan {gpu.shape}"
                )
            if not np.isfinite(gpu).all():
                raise AssertionError(f"{name}: nonfinite Vulkan values")
            delta = np.abs(cpu - gpu)
            mismatched = int(np.sum(~np.isclose(
                cpu, gpu, atol=3e-3, rtol=3e-3,
            )))
            entry = {
                "name": name,
                "shape": list(cpu.shape),
                "max_absolute_error": float(delta.max()),
                "mean_absolute_error": float(delta.mean()),
                "p99_absolute_error": float(np.percentile(delta, 99)),
                "mismatched": mismatched,
            }
            report["samples"].append(entry)
            if mismatched and first_mismatch is None:
                first_mismatch = name
            print(
                f"[stage2 diagnostic] {name}: "
                f"max={entry['max_absolute_error']:.8g}, "
                f"mean={entry['mean_absolute_error']:.8g}, "
                f"mismatched={mismatched}",
                flush=True,
            )
            np.save(directory / f"{name}_cpu.npy", cpu)
            np.save(directory / f"{name}_vulkan.npy", gpu)
        report["first_mismatch"] = first_mismatch
        report["numerical_match"] = first_mismatch is None
        report["stage"] = "complete"
        if first_mismatch is None:
            print("[stage2 diagnostic] all probe outputs match", flush=True)
        else:
            print(
                f"[stage2 diagnostic] first divergent boundary: "
                f"{first_mismatch}",
                flush=True,
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        (directory / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if "error" in report or report.get("numerical_match") is False else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="llvmpipe")
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Use the trained Stage2 checkpoint instead of a reduced fixture",
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument(
        "--probe-only", action="store_true",
        help="Verify FX probe sites and CPU re-export without compiling",
    )
    parser.add_argument(
        "--first-block", action="store_true",
        help="Probe the first full-spatial block's QKV, attention, local mix "
             "and MLP instead of the full Stage2 block boundaries",
    )
    args = parser.parse_args()
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            parser.error(f"Checkpoint not found: {args.checkpoint}")
        if min(args.height, args.width) <= 0 or (
            args.height % 16 or args.width % 16
        ):
            parser.error("Trained Stage2 image dimensions must be multiples of 16")
    return run(
        args.output, args.target, checkpoint=args.checkpoint,
        height=args.height, width=args.width, probe_only=args.probe_only,
        first_block=args.first_block,
    )


if __name__ == "__main__":
    raise SystemExit(main())
