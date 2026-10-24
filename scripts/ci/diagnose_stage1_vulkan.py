#!/usr/bin/env python3
"""Locate the first incorrect operator boundary in trained Stage1 Vulkan.

Instrument the exact exported source graph and compare the unchanged
operators' intermediate outputs on CPU and Vulkan. No checkpoint tensors,
module parameters, arithmetic semantics, or numerical tolerances are changed.
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
from torch.export.graph_signature import OutputKind, OutputSpec, TensorArgument
from torch.utils import _pytree as pytree


# Scope names follow the ORIGINAL Stage1 modules. Selecting scopes, rather
# than reimplementing intermediate operators, preserves source semantics.
CORE_SCOPES = (
    "stem",
    "enc_16.0.norm1",
    "enc_16.0.attn.qkv",
    "enc_16.0.attn",
    "enc_16.0.norm2",
    "enc_16.0.mlp",
    "enc_16.0",
    "down",
    "fuse.norm",
    "fuse",
    "body_64.0.norm1",
    "body_64.0.attn",
    "body_64.0.norm2",
    "body_64.0.mlp",
    "body_64.0",
    "mid_up",
    "mid_down",
    "up",
    "dec_16.0.norm1",
    "dec_16.0.attn",
    "dec_16.0",
    "aux_depatchify",
)


def module_scope_names(node):
    stack = node.meta.get("nn_module_stack") or {}
    scopes = []
    for entry in stack.values():
        if isinstance(entry, (tuple, list)):
            scopes.append(str(entry[0]))
        else:
            scopes.append(str(entry))
    return tuple(scopes)


def _matches_scope(node, name):
    return any(scope.endswith("." + name)
               or scope == name or (".source." + name) in scope
               for scope in module_scope_names(node))


def _eligible(node):
    value = node.meta.get("val")
    return (node.op == "call_function" and
            hasattr(value, "shape") and len(value.shape) > 0
            and all(isinstance(dim, int) and dim > 0
                    for dim in value.shape))


def expose_stage1_boundaries(exported, scopes=CORE_SCOPES):
    """Select one original graph tensor per present high-level module.

    A module's final tensor-producing FX op is its output when its forward
    returns a single tensor. Narrower branches such as qkv and norm are
    captured independently to distinguish the input, attention, and MLP.
    Never silently accept an absent scope: report it in the manifest.
    """
    # Instrument the source ExportedProgram directly. Re-exporting its
    # unlifted module generates an unsupported copy_ for hint_quality.
    gm = exported.graph_module
    selected = []
    missed = []
    for scope in scopes:
        candidates = [
            node for node in gm.graph.nodes
            if _eligible(node) and _matches_scope(node, scope)
        ]
        if not candidates:
            missed.append(scope)
            continue
        selected.append((scope, candidates[-1]))
    if not selected:
        available = sorted({
            path for node in gm.graph.nodes for path in module_scope_names(node)
        })
        raise RuntimeError(
            f"No Stage1 probe scopes matched the export. Samples: {available[:50]}"
        )

    out = next(node for node in gm.graph.nodes if node.op == "output")
    previous = out.args[0]
    if not isinstance(previous, (list, tuple)):
        previous = (previous,)
    output_specs = exported.graph_signature.output_specs
    if len(previous) != len(output_specs):
        raise AssertionError("Stage1 exported graph output/signature mismatch")
    final_count = sum(
        spec.kind == OutputKind.USER_OUTPUT for spec in output_specs
    )
    if final_count == 0:
        raise AssertionError("Stage1 export has no user-visible outputs")

    names = [f"probe_{scope}" for scope, _ in selected]
    out.args = (tuple(previous) + tuple(node for _, node in selected),)
    output_specs.extend(
        OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(node.name), None)
        for _, node in selected
    )
    # Preserve existing mutation outputs and parameter/buffer lifts. The
    # call-spec tree covers user outputs only, excluding mutation outputs.
    root = exported.module_call_graph[0]
    if root.fqn != "" or root.signature.out_spec is None:
        raise AssertionError("Unexpected Stage1 exported call signature")
    root.signature.out_spec = pytree.tree_flatten(
        tuple(range(final_count + len(selected)))
    )[1]
    gm.graph.lint()
    gm.recompile()
    exported.validate()
    return exported, names, final_count, missed


def run(args):
    import iree.turbine.aot as aot

    from prove_stage1_vulkan import create_stage1, export_stage1
    from prove_iree_vulkan import _read_device_result

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "original source Stage1 exported intermediate tensors",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "target": args.target,
        "rms_reduction": args.rms_reduction,
        "down_reduction": args.down_reduction,
        "probes": [],
        "unmatched_scopes": [],
        "first_divergence": None,
        "probe_only": args.probe_only,
        "import_only": args.import_only,
        "imported": False,
    }
    try:
        if args.checkpoint is None:
            model, inputs, reference = create_stage1()
            report["fixture"] = "reduced source-derived Stage1"
        else:
            from prove_checkpoint_stage1 import (
                load_checkpoint_stage1, make_inputs, TrainedStage1ABI,
                reference_outputs,
            )
            stage1, cfg, _, dims, count = load_checkpoint_stage1(args.checkpoint)
            report["trained_tensors"] = count
            inputs = make_inputs(cfg, dims, args.frames, args.height, args.width)
            reference = reference_outputs(stage1, inputs)
            model = TrainedStage1ABI(stage1).eval()
            report["fixture"] = "original trained Stage1"
            report["feature_dimensions"] = dims

        exported, _ = export_stage1(
            model, inputs, rms_reduction=args.rms_reduction,
            down_reduction=args.down_reduction,
        )
        present = {name for name, _ in model.named_modules()}
        active = tuple(
            name for name in CORE_SCOPES
            if "source." + name in present
        )
        report["requested_scopes"] = list(active)
        probed, names, final_count, missed = expose_stage1_boundaries(
            exported, active
        )
        report["unmatched_scopes"] = missed
        report["probe_names"] = names
        if final_count != len(reference):
            raise AssertionError("Trained Stage1 source/export output arity changed")
        with torch.inference_mode():
            expected = tuple(probed.module()(*inputs))
        if len(expected) != final_count + len(names):
            raise AssertionError("Probe graph output arity mismatch")
        for i, (t, ref) in enumerate(zip(expected[:final_count], reference)):
            actual = t.detach().numpy().transpose(0, 1, 4, 2, 3).copy()
            np.testing.assert_allclose(
                actual, ref, atol=2e-5, rtol=2e-5,
                err_msg=f"Instrumented Stage1 source output {i} differs",
            )
        print(
            f"[Stage1 differential] {len(names)} source scopes, "
            f"{len(missed)} unavailable, CPU probe graph passed",
            flush=True,
        )
        if args.probe_only:
            report["cpu_proof"] = True
            return 0
        compiled = aot.export(probed)
        compiled.verify()
        compiled.save_mlir(args.output / "stage1_probes.mlir")
        report["imported"] = True
        if args.import_only:
            print("[Stage1 differential] imported probed Stage1 to Torch-MLIR", flush=True)
            return 0
        compiled.session.set_flags(f"--iree-vulkan-target={args.target}")
        compiled.session.set_flags("--iree-opt-level=O2")
        vmfb = args.output / "stage1_probes.vmfb"
        compiled.compile(save_to=vmfb, target_backends=["vulkan-spirv"])
        report["compiled"] = True
        print("[Stage1 differential] compiled instrumented trained Stage1", flush=True)

        import iree.runtime as rt

        config = rt.Config("vulkan")
        module = rt.VmModule.copy_buffer(
            config.vm_instance, vmfb.read_bytes()
        )
        runtime = rt.load_vm_module(module, config)
        device_inputs = [
            rt.asdevicearray(config.device, tensor.numpy())
            for tensor in inputs
        ]
        result = runtime.main(*device_inputs)
        if not isinstance(result, (list, tuple)):
            result = (result,)
        if len(result) != len(expected):
            raise AssertionError(
                f"Vulkan diagnostic output count {len(result)} != {len(expected)}"
            )
        for index, (name, expected_value, device_value) in enumerate(zip(
            [f"final_{i}" for i in range(final_count)] + names, expected, result
        )):
            actual = _read_device_result(config, device_value)
            reference_value = expected_value.detach().numpy()
            if actual.shape != reference_value.shape:
                raise AssertionError(
                    f"{name}: Vulkan {actual.shape} != CPU {reference_value.shape}"
                )
            if not np.isfinite(actual).all():
                raise AssertionError(f"{name}: nonfinite Vulkan output")
            delta = np.abs(actual - reference_value)
            count = int((~np.isclose(
                actual, reference_value, atol=3e-3, rtol=3e-3
            )).sum())
            info = {
                "name": name, "shape": list(actual.shape),
                "max_absolute_error": float(delta.max()),
                "mean_absolute_error": float(delta.mean()),
                "mismatched": count,
                "total": int(actual.size),
            }
            report["probes"].append(info)
            if index >= final_count and count and report["first_divergence"] is None:
                report["first_divergence"] = name
            np.save(args.output / f"{name.replace('.', '_')}_vulkan.npy", actual)
            print(
                f"[Stage1 differential] {name}: max={delta.max():.8g}, "
                f"mean={delta.mean():.8g}, mismatched={count}/{actual.size}",
                flush=True,
            )
        if report["first_divergence"]:
            print(
                "[Stage1 differential] earliest probed divergence: "
                + report["first_divergence"], flush=True,
            )
        else:
            print(
                "[Stage1 differential] no divergence at sampled boundaries",
                flush=True,
            )
        return int(bool(report["first_divergence"]))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="rdna3")
    parser.add_argument("--rms-reduction", choices=("native", "segmented128"),
                        default="segmented128")
    parser.add_argument("--down-reduction", choices=("native", "segmented128"),
                        default="native")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe-only", action="store_true")
    modes.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.frames <= 0 or min(args.height, args.width) <= 0:
        parser.error("Frame dimensions must be positive")
    if args.height % 64 or args.width % 64:
        parser.error("Height and width must be 64-aligned")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
