#!/usr/bin/env python3
"""Isolate the original Stage2 first-block RMSNorm for IREE Vulkan validation.

Run the exact learned first-block normalization and its actual CPU input when
--checkpoint is given. Compare native-rank, flattened token-major, and
reassociated 256-channel reduction implementations with unchanged source
weights. Emit intermediate arithmetic results, and optionally evaluate alternative
source-equivalent reduction lowerings without relaxing numerical tolerances.
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
    _read_device_result, _rms_norm_inference_math,
    _rms_norm_segmented_128_math,
)


class FlatRMSNorm(torch.nn.Module):
    """Reshape only the leading dimensions of the unchanged source RMSNorm."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        return self.source(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class ChunkedRMSNorm(torch.nn.Module):
    """Reassociate only the square-sum reduction, retaining source weights.

    Splitting the reduction into contiguous 256-channel groups preserves
    RMSNorm's normalization domain and learned affine transform. This probes
    whether Vulkan errors are specific to a wide 1280-element reduction.
    """

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        channel_count = x.shape[-1]
        if channel_count % 256:
            raise ValueError("Chunked RMSNorm expects 256-aligned channel count")
        accumulation = x.float() if x.dtype in (
            torch.float16, torch.bfloat16
        ) else x
        square = accumulation.square().reshape(
            *x.shape[:-1], channel_count // 256, 256
        )
        variance = square.sum(dim=-1).sum(dim=-1, keepdim=True) / channel_count
        normalized = (accumulation * torch.rsqrt(
            variance + self.source.eps
        )).to(x.dtype)
        return normalized * self.source.weight.to(x.dtype)




class ContractedRMSNorm(torch.nn.Module):
    """Compute the original channel mean-square as a matrix-vector product.

    A constant 1280x1 coefficient vector contracts the original squared
    channels. This is mathematically RMSNorm, not an alternate learned model:
    weights, epsilon, channel domain and per-token normalization are intact.
    """

    def __init__(self, source):
        super().__init__()
        self.source = source
        channels = source.normalized_shape[0]
        self.register_buffer(
            "mean_coefficients",
            torch.full((channels, 1), 1.0 / channels),
            persistent=False,
        )

    def forward(self, x):
        accum = x.float() if x.dtype in (
            torch.float16, torch.bfloat16
        ) else x
        width = accum.shape[-1]
        squared = accum.square().reshape(-1, width)
        variance = (
            squared @ self.mean_coefficients.to(dtype=accum.dtype)
        ).reshape(*x.shape[:-1], 1)
        scale = torch.rsqrt(variance + self.source.eps)
        return (accum * scale).to(x.dtype) * self.source.weight.to(x.dtype)


class PaddedRMSNorm(torch.nn.Module):
    """Reduce over an explicitly zero-padded power-of-two channel dimension.

    Divide by the original channel count, never the padded extent.
    This tests the Vulkan reduction's tail handling without relaxing source
    semantics or modifying checkpoint weights.
    """

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        import torch.nn.functional as F

        accum = x.float() if x.dtype in (
            torch.float16, torch.bfloat16
        ) else x
        width = accum.shape[-1]
        extent = 1 << (width - 1).bit_length()
        square = F.pad(accum.square(), (0, extent - width))
        variance = square.sum(dim=-1, keepdim=True) / width
        return (
            (accum * torch.rsqrt(variance + self.source.eps)).to(x.dtype)
            * self.source.weight.to(x.dtype)
        )




class RMSNormTerms(torch.nn.Module):
    """Expose the original normalized input, reduction and affine operands."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        weight = self.source.weight.to(dtype=x.dtype)
        square = x.square()
        variance = square.mean(dim=-1, keepdim=True)
        inverse_scale = torch.rsqrt(variance + self.source.eps)
        unweighted = x * inverse_scale
        weighted = unweighted * weight
        return x, square, variance, inverse_scale, unweighted, weight, weighted


TERM_NAMES = (
    "input", "square", "mean_square", "inverse_scale",
    "unweighted", "affine_weight", "weighted",
)


def run_terms(module, sample, reference, args):
    """Compile and compare every term without substituting another network."""
    import iree.turbine.aot as aot
    import iree.runtime as rt

    model = RMSNormTerms(module).eval()
    with torch.inference_mode():
        expected = tuple(v.detach().numpy() for v in model(sample))
    np.testing.assert_allclose(
        expected[-1], reference, atol=2e-5, rtol=2e-5,
        err_msg="Explicit terms do not reproduce the source RMSNorm",
    )

    exported = torch.export.export(model, (sample,), strict=True)
    with torch.inference_mode():
        cpu_exported = tuple(v.detach().numpy()
                             for v in exported.module()(sample))
    for name, a, b in zip(TERM_NAMES, cpu_exported, expected):
        np.testing.assert_allclose(
            a, b, atol=2e-5, rtol=2e-5,
            err_msg=f"CPU re-export changed term {name}",
        )

    compiled = aot.export(exported)
    compiled.verify()
    compiled.save_mlir(args.output / "terms.mlir")
    compiled.session.set_flags(f"--iree-vulkan-target={args.target}")
    compiled.session.set_flags("--iree-opt-level=O2")
    binary_path = args.output / "terms.vmfb"
    compiled.compile(save_to=binary_path, target_backends=["vulkan-spirv"])
    print("[rmsnorm terms] compiled original RMSNorm algebra", flush=True)
    result = {"compile": True, "first_mismatch": None, "values": []}
    if not args.execute_vulkan:
        return result

    cfg = rt.Config("vulkan")
    binary = rt.VmModule.copy_buffer(cfg.vm_instance, binary_path.read_bytes())
    program = rt.load_vm_module(binary, cfg)
    device_outputs = program.main(
        rt.asdevicearray(cfg.device, sample.detach().numpy()),
    )
    if not isinstance(device_outputs, (tuple, list)):
        raise AssertionError("Expected multiple RMSNorm term outputs")
    if len(device_outputs) != len(TERM_NAMES):
        raise AssertionError(
            f"Expected {len(TERM_NAMES)} RMSNorm outputs, got {len(device_outputs)}"
        )

    for name, cpu, device_result in zip(TERM_NAMES, expected, device_outputs):
        gpu = _read_device_result(cfg, device_result)
        np.save(args.output / f"term_{name}_cpu.npy", cpu)
        np.save(args.output / f"term_{name}_vulkan.npy", gpu)
        if gpu.shape != cpu.shape:
            raise AssertionError(
                f"{name}: CPU shape {cpu.shape}, device shape {gpu.shape}"
            )
        if not np.isfinite(gpu).all():
            raise AssertionError(f"{name}: nonfinite device output")
        delta = np.abs(gpu - cpu)
        mismatched = int((~np.isclose(gpu, cpu, atol=3e-3, rtol=3e-3)).sum())
        entry = {
            "name": name, "shape": list(cpu.shape),
            "max_absolute_error": float(delta.max()),
            "mean_absolute_error": float(delta.mean()),
            "mismatched": mismatched,
        }
        if name == "mean_square":
            # Preserve the scalar reduction values and their ratio so an
            # incorrect padded divisor can be distinguished from missing
            # or repeated channels in the generated GPU reduction.
            denominator = np.where(cpu != 0, cpu, 1.0)
            ratios = (gpu / denominator).reshape(-1)
            entry["cpu"] = cpu.reshape(-1).tolist()
            entry["vulkan"] = gpu.reshape(-1).tolist()
            entry["vulkan_over_cpu_ratio"] = ratios.tolist()
            entry["median_vulkan_over_cpu_ratio"] = float(np.median(ratios))
            print(
                "[rmsnorm terms] mean_square: median Vulkan/CPU ratio="
                f"{entry['median_vulkan_over_cpu_ratio']:.8g}",
                flush=True,
            )
        result["values"].append(entry)
        if mismatched and result["first_mismatch"] is None:
            result["first_mismatch"] = name
        print(
            f"[rmsnorm terms] {name}: "
            f"max={entry['max_absolute_error']:.8g}, "
            f"mean={entry['mean_absolute_error']:.8g}, "
            f"mismatched={mismatched}",
            flush=True,
        )
    return result



def build_source(checkpoint: Path | None, height: int, width: int):
    if checkpoint is None:
        from corridorkey.blocks import RMSNorm

        torch.manual_seed(1942)
        normalized = RMSNorm(1280).eval()
        with torch.no_grad():
            normalized.weight.copy_(
                torch.linspace(0.6, 1.4, normalized.weight.numel())
            )
        sample = torch.randn(1, height // 16, width // 16, 1280)
        return normalized, sample

    # Capture the trained input at the beginning of the exact original block.
    from prove_checkpoint_stage2 import load_checkpoint_stage2

    torch.manual_seed(1942)
    stage2, config, _manifest, _count = load_checkpoint_stage2(checkpoint)
    rgb = torch.randn(1, 1, 3, height, width)
    baton = torch.randn(1, 1, config["d_16"], height // 16, width // 16)
    inputs = []

    def capture(_module, args):
        inputs.append(args[0].detach().clone())

    handle = stage2.pre_blocks[0].norm1.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            stage2(rgb, baton)
    finally:
        handle.remove()
    if len(inputs) != 1:
        raise AssertionError(
            f"Expected one first-block RMSNorm invocation, found {len(inputs)}"
        )
    return stage2.pre_blocks[0].norm1.eval(), inputs[0]


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "original Stage2 first-block RMSNorm source module",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "shape": [1, args.height // 16, args.width // 16, 1280],
        "target": args.target,
        "execute_vulkan": args.execute_vulkan,
        "cases": [],
    }
    try:
        module, sample = build_source(args.checkpoint, args.height, args.width)
        report["shape"] = list(sample.shape)
        np.save(args.output / "input.npy", sample.detach().numpy())
        with torch.inference_mode():
            reference = module(sample).detach().numpy()
            flattened_reference = FlatRMSNorm(module)(sample).detach().numpy()
        np.testing.assert_allclose(
            flattened_reference, reference, atol=1e-6, rtol=1e-6,
            err_msg="Reshaping the original RMSNorm changed source inference",
        )
        np.save(args.output / "source.npy", reference)

        import iree.turbine.aot as aot

        if args.trace_terms:
            try:
                report["terms"] = run_terms(module, sample, reference, args)
            except Exception as exc:
                report["terms"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(report["terms"]["traceback"], file=sys.stderr, flush=True)
        else:
            report["terms"] = {
                "skipped": True,
                "reason": "instrumented term graph disabled for this run",
            }

        variants = [
            ("native_rank", module, "O2"),
            ("flat_tokens", FlatRMSNorm(module), "O2"),
            ("chunked_256", ChunkedRMSNorm(module), "O2"),
            ("segmented128", module, "O2"),
        ]
        if args.explore_reductions:
            # Distinct lowered graph families, including an unoptimized
            # compilation of the original source operation. None changes
            # the checkpoint weights, epsilon, or normalized channels.
            variants.extend([
                ("contracted", ContractedRMSNorm(module), "O2"),
                ("padded_pow2", PaddedRMSNorm(module), "O2"),
                ("native_rank_O0", module, "O0"),
            ])
        for kind, variant, opt_level in variants:
            entry = {"kind": kind, "compile": False, "numerical_match": None}
            report["cases"].append(entry)
            try:
                exported = torch.export.export(
                    variant, (sample,), strict=True,
                ).run_decompositions({
                    torch.ops.aten.rms_norm.default: (
                        _rms_norm_segmented_128_math
                        if kind == "segmented128"
                        else _rms_norm_inference_math
                    ),
                })
                with torch.inference_mode():
                    cpu_exported = exported.module()(sample).detach().numpy()
                np.testing.assert_allclose(
                    cpu_exported, reference, rtol=2e-5, atol=2e-5,
                    err_msg=f"{kind}: source/export CPU disagreement",
                )
                compiled = aot.export(exported)
                compiled.verify()
                compiled.save_mlir(args.output / f"{kind}.mlir")
                compiled.session.set_flags(
                    f"--iree-vulkan-target={args.target}"
                )
                compiled.session.set_flags(f"--iree-opt-level={opt_level}")
                entry["optimization"] = opt_level
                vmfb = args.output / f"{kind}.vmfb"
                compiled.compile(
                    save_to=vmfb, target_backends=["vulkan-spirv"],
                )
                entry["compile"] = True
                print(f"[rmsnorm] {kind}: Vulkan compile passed", flush=True)
                if not args.execute_vulkan:
                    continue

                import iree.runtime as rt

                config = rt.Config("vulkan")
                binary = rt.VmModule.copy_buffer(
                    config.vm_instance, vmfb.read_bytes(),
                )
                program = rt.load_vm_module(binary, config)
                result = program.main(
                    rt.asdevicearray(config.device, sample.detach().numpy()),
                )
                if isinstance(result, (tuple, list)):
                    if len(result) != 1:
                        raise AssertionError("Expected one normalization result")
                    result = result[0]
                actual = _read_device_result(config, result)
                np.save(args.output / f"{kind}_vulkan.npy", actual)
                if actual.shape != reference.shape or not np.isfinite(actual).all():
                    raise AssertionError("Unexpected shape or nonfinite Vulkan tensor")
                difference = np.abs(actual - reference)
                mismatched = int(np.count_nonzero(~np.isclose(
                    actual, reference, atol=3e-3, rtol=3e-3,
                )))
                entry.update({
                    "max_absolute_error": float(difference.max()),
                    "mean_absolute_error": float(difference.mean()),
                    "mismatched": mismatched,
                    "numerical_match": mismatched == 0,
                })
                print(
                    f"[rmsnorm] {kind}: max={entry['max_absolute_error']:.8g} "
                    f"mean={entry['mean_absolute_error']:.8g} "
                    f"mismatched={mismatched}",
                    flush=True,
                )
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()
                print(entry["traceback"], file=sys.stderr, flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
    return int(
        "error" in report
        or "error" in report.get("terms", {})
        or report.get("terms", {}).get("first_mismatch") is not None
        or any(
            "error" in item or item["numerical_match"] is False
            for item in report["cases"]
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--target", default="rdna1")
    parser.add_argument("--execute-vulkan", action="store_true")
    parser.add_argument(
        "--no-trace-terms", dest="trace_terms", action="store_false",
        help="Run source normalization variants without the optional "
             "seven-output RMSNorm arithmetic instrumentation",
    )
    parser.set_defaults(trace_terms=True)
    parser.add_argument(
        "--explore-reductions", action="store_true",
        help="Also compile source-equivalent matmul and padded reductions, "
             "plus the native source at IREE optimization level O0",
    )
    parser.add_argument("--output", type=Path, default=Path("build/rmsnorm-proof"))
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or (
        args.height % 16 or args.width % 16
    ):
        parser.error("Image dimensions must be positive multiples of 16")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
