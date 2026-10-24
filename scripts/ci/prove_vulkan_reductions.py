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
import re
import shutil
import subprocess
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


def collect_vulkan_properties(directory: Path):
    """Record loader-reported properties, including subgroup size controls."""
    command = shutil.which("vulkaninfo")
    if not command:
        return {"available": False, "reason": "vulkaninfo is not on PATH"}
    try:
        result = subprocess.run(
            [command], text=True, capture_output=True,
            encoding="utf-8", errors="replace", timeout=45,
            check=False,
        )
        (directory / "vulkaninfo.txt").write_text(
            result.stdout + "\n" + result.stderr, encoding="utf-8"
        )
        expressions = (
            "deviceName", "driverName", "driverInfo", "driverVersion",
            "subgroupSize", "minSubgroupSize", "maxSubgroupSize",
            "requiredSubgroupSizeStages", "subgroupSizeControl",
            "computeFullSubgroups", "maxComputeWorkgroupSubgroups",
        )
        properties = [
            line.strip() for line in result.stdout.splitlines()
            if any(re.match(r"^" + re.escape(name) + r"\s*=", line.strip())
                   for name in expressions)
        ]
        return {
            "available": True, "returncode": result.returncode,
            "properties": properties, "full_output": "vulkaninfo.txt",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}


def probe_one_hot(runtime, config, width, operation):
    """Map original input channel positions to measured GPU multiplicities.

    Preserve the compiled (1,4,4,width) shape, feed runtime one-hot inputs
    in batches of 16, and measure outputs without changing the reduction.
    """
    if operation not in ("sum_square", "mean_square"):
        return None
    from prove_iree_vulkan import _read_device_result
    group = 16
    observed = []
    ideal = 1.0 / width if operation == "mean_square" else 1.0
    for start in range(0, width, group):
        hot = np.zeros((1, 4, 4, width), dtype=np.float32)
        count = min(group, width - start)
        for index in range(count):
            hot.reshape(group, width)[index, start + index] = 1.0
        result = runtime.main(rt_asdevice(config, hot))
        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise AssertionError("Unexpected one-hot output count")
            result = result[0]
        values = _read_device_result(config, result).reshape(-1)
        observed.extend((values[:count] / ideal).astype(float).tolist())
    multiples = np.asarray(observed, dtype=np.float64)
    return {
        "count": len(observed),
        "expected_multiplier": 1.0,
        "min": float(multiples.min()), "max": float(multiples.max()),
        "median": float(np.median(multiples)),
        "roughly_once": int(np.count_nonzero(
            np.isclose(multiples, 1.0, atol=3e-3, rtol=3e-3)
        )),
        "roughly_twice": int(np.count_nonzero(
            np.isclose(multiples, 2.0, atol=3e-3, rtol=3e-3)
        )),
        "by_channel": multiples.tolist(),
    }


def rt_asdevice(config, tensor):
    import iree.runtime as rt
    return rt.asdevicearray(config.device, tensor)


def collect_executable_subgroup_metadata(directory: Path):
    """Extract complete dispatch configurations and SPIR-V entry-point ABIs.

    These are compiler artifacts, not proof of the subgroup size that the
    Vulkan driver actually used. Do not infer a requested subgroup from the
    presence of `subgroup_size_choices` in the target capability set.
    """
    dispatches = []
    abi = []
    choices = set()
    selected_sizes = set()
    for path in sorted(directory.rglob("*.mlir")):
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        name = str(path.relative_to(directory))
        for match in re.finditer(
            r"subgroup_size_choices\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]",
            content,
        ):
            choices.add((int(match.group(1)), int(match.group(2))))
        for match in re.finditer(
            r"(?<!\w)subgroup_size\s*=\s*(\d+)", content
        ):
            selected_sizes.add(int(match.group(1)))
        for index, line in enumerate(lines):
            if "iree_codegen.dispatch_config" in line:
                # A subgroup_size value is commonly nested in this block,
                # several lines after the opening declaration.
                block = [line.strip()]
                for following in lines[index + 1:index + 22]:
                    block.append(following.strip())
                    if following.strip() == "}":
                        break
                dispatches.append({
                    "file": name,
                    "line": index + 1,
                    "config": "\n".join(block),
                })
            if ("spirv.entry_point_abi" in line
                    or "spirv.entry_point" in line
                    and "abi" in line):
                context = " ".join(
                    value.strip() for value in lines[index:index + 4]
                )
                abi.append({
                    "file": name,
                    "line": index + 1,
                    "text": context[:1400],
                })
    binaries = []
    for path in sorted(directory.rglob("*.spv")):
        try:
            header = path.read_bytes()[:8]
            if len(header) < 8:
                continue
            magic, version = __import__("struct").unpack("<II", header)
            if magic != 0x07230203:
                continue
            binaries.append({
                "file": str(path.relative_to(directory)),
                "spirv_version": (
                    f"{(version >> 16) & 255}.{(version >> 8) & 255}"
                ),
            })
        except OSError:
            continue
    return {
        "target_subgroup_size_choices": [list(choice) for choice in sorted(choices)],
        "selected_subgroup_sizes_in_ir": sorted(selected_sizes),
        "dispatch_configurations": dispatches,
        "spirv_entry_point_abis": abi,
        "spirv_binaries": binaries,
        "interpretation": "Subgroup size choices are hardware capabilities, "
            "not a pipeline requirement. An explicit subgroup_size in the "
            "dispatch config must propagate to the SPIR-V entry-point ABI "
            "and serialized Vulkan PipelineDef to request wave32 at runtime. "
            "Absent metadata here is not proof of the final pipeline state.",
    }


def inspect_existing_dumps(output: Path) -> int:
    """Refresh evidence using previously dumped executables; no Vulkan calls."""
    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for case in report.get("cases", []):
        directory = output / f"{case['name']}_executables"
        if directory.is_dir():
            case["codegen_subgroup_metadata"] = (
                collect_executable_subgroup_metadata(directory)
            )
        else:
            case["codegen_subgroup_metadata"] = {
                "error": f"Missing executable dump directory: {directory}"
            }
    out = output / "subgroup-inspection.json"
    out.write_text(json.dumps({
        "device_names": report.get("device_names", []),
        "vulkan_properties": report.get("vulkan_properties", {}),
        "cases": [
            {
                "name": c["name"],
                "width": c["width"],
                "numerical_match": c.get("numerical_match"),
                "one_hot_summary": {
                    key: c.get("one_hot", {}).get(key)
                    for key in ("min", "max", "roughly_once", "roughly_twice")
                },
                "codegen_subgroup_metadata": c.get("codegen_subgroup_metadata"),
            }
            for c in report["cases"]
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[reduction sweep] Compiler subgroup inspection: {out}", flush=True)
    for case in report["cases"]:
        metadata = case["codegen_subgroup_metadata"]
        print(
            f"[reduction sweep] {case['name']}: "
            f"target choices={metadata.get('target_subgroup_size_choices')}; "
            f"explicit IR subgroup sizes="
            f"{metadata.get('selected_subgroup_sizes_in_ir')}; "
            f"entry-point ABIs={len(metadata.get('spirv_entry_point_abis', []))}",
            flush=True,
        )
    return int(any(
        "error" in case.get("codegen_subgroup_metadata", {})
        for case in report["cases"]
    ))


def run(args):
    if args.inspect_only:
        return inspect_existing_dumps(args.output)
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
    if args.one_hot and not args.execute_vulkan:
        raise ValueError("--one-hot requires --execute-vulkan")
    if args.require_subgroup_size and not args.dump_executables:
        raise ValueError("--require-subgroup-size needs --dump-executables")
    if args.record_subgroups:
        report["vulkan_properties"] = collect_vulkan_properties(args.output)
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
                    if args.dump_executables:
                        case["codegen_subgroup_metadata"] = (
                            collect_executable_subgroup_metadata(dump)
                        )
                        if args.require_subgroup_size:
                            chosen = case["codegen_subgroup_metadata"][
                                "selected_subgroup_sizes_in_ir"
                            ]
                            if chosen != [args.require_subgroup_size]:
                                raise AssertionError(
                                    f"{name}: expected explicit subgroup size "
                                    f"{args.require_subgroup_size}, got {chosen}; "
                                    "compiler output is not proven wave32"
                                )
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
                    if args.one_hot and operation in ("sum_square", "mean_square"):
                        case["one_hot"] = probe_one_hot(
                            program, config, width, operation
                        )
                        print(
                            f"[reduction sweep] {name}: one-hot "
                            f"range={case['one_hot']['min']:.6g}.."
                            f"{case['one_hot']['max']:.6g}; "
                            f"twice={case['one_hot']['roughly_twice']}/"
                            f"{width}", flush=True,
                        )
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
    parser.add_argument("--one-hot", action="store_true",
                        help="Measure per-channel contribution multiplicity "
                             "with identical runtime reduction shapes")
    parser.add_argument("--record-subgroups", action="store_true",
                        help="Capture vulkaninfo device/subgroup properties")
    parser.add_argument("--require-subgroup-size", type=int, choices=(32, 64),
                        help="Require explicit subgroup-size selection in "
                             "emitted IREE IR; needs --dump-executables")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Inspect previously dumped executables without "
                             "recompiling or executing any GPU kernels")
    parser.add_argument(
        "--output", type=Path, default=Path("build/reduction-sweep")
    )
    args = parser.parse_args()
    if args.source_input is not None and not args.source_input.is_file():
        parser.error(f"Source input not found: {args.source_input}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
