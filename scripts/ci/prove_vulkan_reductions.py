#!/usr/bin/env python3
"""Probe IREE Vulkan sum/mean-square reductions across channel widths.

Run source-equivalent isolated reductions on the same Vulkan device used
by the trained Stage2 checkpoint. Record exact CPU/GPU reduction values,
ratios and compilation artifacts without changing trained inference.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prove_iree_vulkan import _read_device_result


class Reduction(torch.nn.Module):
    def __init__(self, operation: str):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        squares = x.square()
        if self.operation == "sum_square":
            return squares.sum(dim=-1, keepdim=True)
        if self.operation == "mean_square":
            return squares.mean(dim=-1, keepdim=True)
        if self.operation in ("split128_sum_square", "split128_mean_square"):
            # Each source sum is over at most 128 channels, followed by
            # an exact sum of the partial results. This is a diagnostic
            # layout for the same mathematical channel reduction.
            chunks = [
                squares[..., start:start + 128].sum(dim=-1, keepdim=True)
                for start in range(0, squares.shape[-1], 128)
            ]
            partials = torch.cat(chunks, dim=-1)
            result = partials.sum(dim=-1, keepdim=True)
            if self.operation == "split128_mean_square":
                return result / squares.shape[-1]
            return result
        raise ValueError(f"Unsupported reduction: {self.operation}")


def parse_widths(raw: str) -> list[int]:
    widths = [int(value.strip()) for value in raw.split(",")]
    if not widths or len(set(widths)) != len(widths):
        raise ValueError("Expected a nonempty set of distinct channel widths")
    if any(value <= 0 or value > 16384 for value in widths):
        raise ValueError("Channel widths must be in 1..16384")
    return widths


def create_input(widths, input_path):
    maximum = max(widths)
    torch.manual_seed(1942)
    if input_path is None:
        tensor = torch.randn(1, 4, 4, maximum)
    else:
        original = np.load(input_path, allow_pickle=False)
        if original.ndim != 4 or original.shape[:3] != (1, 4, 4):
            raise ValueError(
                "Expected checkpoint RMSNorm input shape (1, 4, 4, channels)"
            )
        if original.dtype != np.float32 or not np.isfinite(original).all():
            raise ValueError("Input must be finite float32")
        tensor = torch.from_numpy(original.copy())
        if maximum > tensor.shape[-1]:
            extra = torch.randn(1, 4, 4, maximum - tensor.shape[-1])
            tensor = torch.cat((tensor, extra), dim=-1)
    return tensor


def run(args):
    import iree.turbine.aot as aot

    args.output.mkdir(parents=True, exist_ok=True)
    widths = parse_widths(args.widths)
    operations = args.operations.split(",")
    supported = {
        "sum_square", "mean_square",
        "split128_sum_square", "split128_mean_square",
    }
    if not operations or len(operations) != len(set(operations)):
        raise ValueError("Specify distinct reduction operations")
    if not set(operations).issubset(supported):
        raise ValueError(f"Supported operations: {sorted(supported)}")
    inputs = create_input(widths, args.source_input)
    dependency_versions = {}
    for package in ("iree-base-compiler", "iree-base-runtime", "iree-turbine"):
        try:
            dependency_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            dependency_versions[package] = None
    report = {
        "dependency_versions": dependency_versions,
        "scope": "unchanged FP32 mean/sum-square reductions on Vulkan",
        "target": args.target,
        "source_input": str(args.source_input) if args.source_input else None,
        "device_names": [],
        "device_selection": "default IREE Vulkan device",
        "cases": [],
    }
    import iree.runtime as rt
    if args.execute_vulkan:
        driver = rt.get_driver("vulkan")
        devices = driver.query_available_devices()
        report["device_names"] = [
            str(item.get("name", "unknown")) for item in devices
        ]
        print(
            f"[reduction sweep] Vulkan devices: {report['device_names']}",
            flush=True,
        )
        if len(devices) > 1:
            print(
                "[reduction sweep] multiple devices detected: IREE will "
                "use its default Vulkan device; verify that it is the "
                "intended AMD GPU", flush=True,
            )
    try:
        for width in widths:
            sample = inputs[..., :width].contiguous()
            for operation in operations:
                name = f"{operation}_{width}"
                case = {
                    "name": name, "width": width, "operation": operation,
                    "compile": False, "numerical_match": None,
                }
                report["cases"].append(case)
                try:
                    model = Reduction(operation).eval()
                    with torch.inference_mode():
                        expected = model(sample).detach().numpy()
                    exported = torch.export.export(
                        model, (sample,), strict=True
                    )
                    with torch.inference_mode():
                        cpu_exported = exported.module()(sample).detach().numpy()
                    np.testing.assert_allclose(
                        cpu_exported, expected, atol=2e-5, rtol=2e-5,
                        err_msg=f"{name}: CPU export changed reduction",
                    )
                    compiled = aot.export(exported)
                    compiled.verify()
                    compiled.save_mlir(args.output / f"{name}.mlir")
                    compiled.session.set_flags(
                        f"--iree-vulkan-target={args.target}"
                    )
                    compiled.session.set_flags("--iree-opt-level=O2")
                    if args.dump_executables:
                        dump = args.output / f"{name}_executables"
                        dump.mkdir(exist_ok=True)
                        compiled.session.set_flags(
                            f"--iree-hal-dump-executable-files-to={dump.resolve()}"
                        )
                    vmfb = args.output / f"{name}.vmfb"
                    compiled.compile(
                        save_to=vmfb, target_backends=["vulkan-spirv"]
                    )
                    case["compile"] = True
                    if not args.execute_vulkan:
                        print(f"[reduction sweep] {name}: compiled", flush=True)
                        continue

                    config = rt.Config("vulkan")
                    binary = rt.VmModule.copy_buffer(
                        config.vm_instance, vmfb.read_bytes()
                    )
                    program = rt.load_vm_module(binary, config)
                    result = program.main(
                        rt.asdevicearray(config.device, sample.numpy())
                    )
                    if isinstance(result, (tuple, list)):
                        if len(result) != 1:
                            raise AssertionError(
                                f"{name}: expected one Vulkan output"
                            )
                        result = result[0]
                    actual = _read_device_result(config, result)
                    if actual.shape != expected.shape:
                        raise AssertionError(
                            f"{name}: GPU shape {actual.shape}, "
                            f"CPU shape {expected.shape}"
                        )
                    if not np.isfinite(actual).all():
                        raise AssertionError(f"{name}: nonfinite GPU output")
                    delta = np.abs(actual - expected)
                    ratios = actual / np.where(expected != 0, expected, 1)
                    mismatch = int((~np.isclose(
                        actual, expected, atol=3e-3, rtol=3e-3,
                    )).sum())
                    case.update({
                        "numerical_match": mismatch == 0,
                        "mismatched": mismatch,
                        "max_absolute_error": float(delta.max()),
                        "mean_absolute_error": float(delta.mean()),
                        "min_ratio": float(ratios.min()),
                        "median_ratio": float(np.median(ratios)),
                        "max_ratio": float(ratios.max()),
                        "cpu": expected.reshape(-1).tolist(),
                        "vulkan": actual.reshape(-1).tolist(),
                        "source_expression": (
                            "two_stage_128" if operation.startswith("split128")
                            else "native"
                        ),
                    })
                    np.save(args.output / f"{name}_vulkan.npy", actual)
                    print(
                        f"[reduction sweep] {name}: "
                        f"mismatched={mismatch}/{actual.size}, "
                        f"max_error={case['max_absolute_error']:.8g}, "
                        f"median_gpu_cpu={case['median_ratio']:.8g}",
                        flush=True,
                    )
                except Exception as exc:
                    case["error"] = f"{type(exc).__name__}: {exc}"
                    case["traceback"] = traceback.format_exc()
                    print(case["traceback"], file=sys.stderr, flush=True)
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return int(any(
        "error" in case or case["numerical_match"] is False
        for case in report["cases"]
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input", type=Path)
    parser.add_argument("--widths", default="32,64,128,256,512,768,1024,1280,2048")
    parser.add_argument("--operations", default="sum_square,mean_square")
    parser.add_argument("--target", default="rdna1")
    parser.add_argument("--execute-vulkan", action="store_true")
    parser.add_argument("--dump-executables", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("build/reduction-sweep")
    )
    args = parser.parse_args()
    if args.source_input is not None and not args.source_input.is_file():
        parser.error(f"Source input not found: {args.source_input}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
