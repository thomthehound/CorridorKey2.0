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
SITES = (
    ("patchify", "patch_1", (1, 32, 2, 2)),
    ("pre_block_0", "pre_blocks.0", (1, 2, 2, 48)),
    ("pre_block_3", "pre_blocks.3", (1, 2, 2, 48)),
    ("pre_block_7", "pre_blocks.7", (1, 2, 2, 48)),
    ("final_block_0", "final_blocks.0", (1, 2, 2, 48)),
    ("final_block_1", "final_blocks.1", (1, 2, 2, 48)),
    ("depatchify", "depatchify", (1, 4, 32, 32)),
)


def _stack_paths(node):
    stack = node.meta.get("nn_module_stack") or {}
    return [
        (entry[0] if isinstance(entry, (tuple, list)) else str(entry))
        for entry in stack.values()
    ]


def expose_intermediates(exported):
    gm = exported.module()
    diagnostics = []

    for name, path, shape in SITES:
        candidates = []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            value = node.meta.get("val")
            if not hasattr(value, "shape") or tuple(value.shape) != shape:
                continue
            if any(path in p for p in _stack_paths(node)):
                candidates.append(node)
        if not candidates:
            available = [
                (node.name, tuple(node.meta["val"].shape), _stack_paths(node))
                for node in gm.graph.nodes
                if hasattr(node.meta.get("val"), "shape")
                and _stack_paths(node)
            ]
            raise RuntimeError(
                f"No exported tensor at {name} ({path}, shape {shape}). "
                f"Available sample: {available[:20]}"
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
    return gm, [name for name, _ in diagnostics]


def run(directory: Path, target: str):
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "source-derived exported Stage2 graph boundary comparisons",
        "target": target,
        "samples": [],
        "stage": "start",
    }
    try:
        source, rgb, baton, _ = create_stage2()
        exported, _ = export_stage2(source, rgb, baton)
        gm, names = expose_intermediates(exported)
        with torch.inference_mode():
            expected = gm(rgb, baton)
        if len(expected) != len(names) + 1:
            raise AssertionError("Diagnostic output count mismatch")

        reexport = torch.export.export(gm, (rgb, baton), strict=True)
        with torch.inference_mode():
            cpu_exported = reexport.module()(rgb, baton)
        for name, a, b in zip(("final", *names), expected, cpu_exported):
            np.testing.assert_allclose(
                a.detach().numpy(), b.detach().numpy(),
                rtol=2e-5, atol=2e-5,
                err_msg=f"Diagnostic re-export changed {name}",
            )
        compiled = aot.export(reexport)
        compiled.verify()
        compiled.save_mlir(directory / "stage2_diagnostic.mlir")
        compiled.session.set_flags(f"--iree-vulkan-target={target}")
        compiled.session.set_flags("--iree-opt-level=O2")
        vmfb = directory / "stage2_diagnostic.vmfb"
        compiled.compile(save_to=vmfb, target_backends=["vulkan-spirv"])
        report["stage"] = "compiled"

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
        for name, source_value, device_value in zip(("final", *names), expected, outputs):
            cpu = source_value.detach().numpy()
            gpu = _read_device_result(cfg, device_value)
            if cpu.shape != gpu.shape:
                raise AssertionError(
                    f"{name}: shape mismatch CPU {cpu.shape}, Vulkan {gpu.shape}"
                )
            if not np.isfinite(gpu).all():
                raise AssertionError(f"{name}: nonfinite Vulkan values")
            delta = np.abs(cpu - gpu)
            entry = {
                "name": name,
                "shape": list(cpu.shape),
                "max_absolute_error": float(delta.max()),
                "mean_absolute_error": float(delta.mean()),
                "p99_absolute_error": float(np.percentile(delta, 99)),
                "mismatched": int(np.sum(~np.isclose(
                    cpu, gpu, atol=3e-3, rtol=3e-3
                ))),
            }
            report["samples"].append(entry)
            print(f"[stage2 diagnostic] {name}: max={entry['max_absolute_error']:.8g}, "
                  f"mean={entry['mean_absolute_error']:.8g}, "
                  f"mismatched={entry['mismatched']}", flush=True)
            np.save(directory / f"{name}_cpu.npy", cpu)
            np.save(directory / f"{name}_vulkan.npy", gpu)
        report["stage"] = "complete"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        (directory / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return 1 if "error" in report else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="llvmpipe")
    args = parser.parse_args()
    return run(args.output, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
