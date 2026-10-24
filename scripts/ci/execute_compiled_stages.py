#!/usr/bin/env python3
"""Execute compiled trained Stage1 -> Stage2 on Vulkan without PyTorch.

Inputs are the six normalized Stage1 boundary tensors produced by the
checkpoint exporter. Frozen foundation model execution is outside this
partition. Stage1's channel-last baton is read back and transposed to the
Stage2 channel-first input ABI; this explicit staging cost is recorded.
The source models are NOT imported in this runtime process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def load_array(path: Path) -> np.ndarray:
    result = np.load(path, allow_pickle=False)
    if result.dtype != np.float32 or not result.flags.c_contiguous:
        raise ValueError(f"{path}: expected contiguous float32 .npy input")
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: input contains nonfinite values")
    return result


def compare(actual, expected, label, atol=3e-3, rtol=3e-3):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"{label}: invalid device result shape/finiteness")
    delta = np.abs(actual - expected)
    mismatched = int(np.count_nonzero(~np.isclose(
        actual, expected, atol=atol, rtol=rtol
    )))
    report = {
        "label": label, "shape": list(actual.shape),
        "max_absolute_error": float(delta.max()),
        "mean_absolute_error": float(delta.mean()),
        "mismatched": mismatched,
    }
    if mismatched:
        raise AssertionError(
            f"{label}: {mismatched}/{actual.size} values differ; "
            f"max={report['max_absolute_error']:.8g}"
        )
    return report


def readback(value) -> np.ndarray:
    """Read IREE Vulkan device-array data through its public host API."""
    result = np.asarray(value.to_host())
    if result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("Nonfinite or non-float32 Vulkan result")
    return result


def load_vmfb(config, path: Path):
    import iree.runtime as rt

    if not path.is_file():
        raise FileNotFoundError(f"Compiled Vulkan VMFB not found: {path}")
    module = rt.VmModule.copy_buffer(config.vm_instance, path.read_bytes())
    return rt.load_vm_module(module, config)


def normalize_outputs(output):
    if isinstance(output, (tuple, list)):
        return tuple(output)
    return (output,)


def execute(args):
    import iree.runtime as rt

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "compiled Stage1 -> Stage2 Vulkan, frozen foundation inputs",
        "stage1_vmfb": str(args.stage1_vmfb),
        "stage2_vmfb": str(args.stage2_vmfb),
        "runs": args.runs,
        "warmup": args.warmup,
        "torch_runtime": False,
        "interstage_transfer": "Vulkan readback -> CPU NHWC/NCHW -> upload",
        "success": False,
    }
    try:
        data = tuple(load_array(args.inputs / f"input_{i}.npy")
                     for i in range(6))
        rgb, hint, moge, cradio, rvm, quality = data
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("Stage1 RGB must have B,T,3,H,W layout")
        batch, frames, _, height, width = rgb.shape
        if height % 64 or width % 64:
            raise ValueError("Stage1 frame resolution must be 64-aligned")
        if hint.shape != (batch, frames, 1, height, width):
            raise ValueError("Stage1 hint shape mismatch")
        for tensor in (moge, cradio, rvm):
            if tensor.ndim != 5 or tensor.shape[:2] != (batch, frames) or (
                tensor.shape[-2:] != (height // 64, width // 64)
            ):
                raise ValueError("Stage1 foundation feature ABI mismatch")
        if quality.shape != (batch, frames):
            raise ValueError("Stage1 hint-quality ABI mismatch")
        report["input_shapes"] = [list(x.shape) for x in data]

        config = rt.Config("vulkan")
        stage1 = load_vmfb(config, args.stage1_vmfb)
        stage2 = load_vmfb(config, args.stage2_vmfb)
        device_inputs = tuple(
            rt.asdevicearray(config.device, value) for value in data
        )
        report["input_upload_bytes"] = sum(x.nbytes for x in data)

        def run_once():
            tic = time.perf_counter_ns()
            stage1_output = normalize_outputs(stage1.main(*device_inputs))
            if len(stage1_output) not in (2, 3):
                raise AssertionError(
                    f"Stage1 returned {len(stage1_output)} outputs, expected "
                    "baton, aux and optional bottleneck"
                )
            baton_nhwc = readback(stage1_output[0])
            aux_nhwc = readback(stage1_output[1])
            if baton_nhwc.shape[:2] != (batch, frames) or (
                baton_nhwc.shape[2:4] != (height // 16, width // 16)
            ):
                raise AssertionError("Stage1 baton shape differs from contract")
            if aux_nhwc.shape != (batch, frames, height // 2, width // 2, 4):
                raise AssertionError("Stage1 auxiliary output shape differs")
            baton_nchw = baton_nhwc.transpose(0, 1, 4, 2, 3).copy()
            bridge_done = time.perf_counter_ns()
            stage2_output = normalize_outputs(stage2.main(
                device_inputs[0],
                rt.asdevicearray(config.device, baton_nchw),
            ))
            if len(stage2_output) not in (1, 2):
                raise AssertionError("Stage2 output count differs from source")
            final_outputs = tuple(
                readback(o).transpose(0, 1, 4, 2, 3).copy()
                for o in stage2_output
            )
            for final in final_outputs:
                if final.shape != (batch, frames, 4, height, width):
                    raise AssertionError("Stage2 output shape differs from source")
            end = time.perf_counter_ns()
            return (baton_nchw, aux_nhwc, final_outputs,
                    (bridge_done - tic) / 1e6, (end - tic) / 1e6)

        for _ in range(args.warmup):
            run_once()
        elapsed = []
        for _ in range(args.runs):
            baton, aux, outputs, stage1_ms, total_ms = run_once()
            elapsed.append({
                "stage1_with_readback_ms": stage1_ms,
                "stage1_stage2_total_ms": total_ms,
            })
        report["latencies_ms"] = elapsed
        report["wall_clock_total_median_ms"] = float(np.median([
            x["stage1_stage2_total_ms"] for x in elapsed
        ]))
        report["wall_clock_total_p95_ms"] = float(np.percentile([
            x["stage1_stage2_total_ms"] for x in elapsed
        ], 95))
        report["baton_readback_bytes"] = baton.nbytes
        report["baton_reupload_bytes"] = baton.nbytes
        report["final_readback_bytes"] = sum(x.nbytes for x in outputs)
        report["stage1_aux_readback_bytes"] = aux.nbytes
        # These timings include device execution, synchronization, array
        # readbacks and host staging; they are not GPU kernel timings.

        np.save(args.output / "stage1_baton.npy", baton)
        np.save(args.output / "stage1_aux.npy",
                aux.transpose(0, 1, 4, 2, 3).copy())
        for index, output in enumerate(outputs):
            np.save(args.output / f"stage2_{index}.npy", output)
        if args.stage1_reference is not None:
            report["stage1_errors"] = [
                compare(baton, load_array(
                    args.stage1_reference / "source_0.npy"
                ), "trained Stage1 baton"),
                compare(aux.transpose(0, 1, 4, 2, 3).copy(), load_array(
                    args.stage1_reference / "source_1.npy"
                ), "trained Stage1 aux"),
            ]
        if args.stage2_reference is not None:
            report["stage2_errors"] = [
                compare(output, load_array(
                    args.stage2_reference / f"source_{index}.npy"
                ), f"trained Stage2 output {index}")
                for index, output in enumerate(outputs)
            ]
        report["success"] = True
        print(
            "[native Vulkan partitions] trained Stage1 -> Stage2 executed; "
            f"median wall-clock={report['wall_clock_total_median_ms']:.3f} ms "
            f"(includes interstage readback/reupload)",
            flush=True,
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return int(not report["success"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-vmfb", type=Path, required=True)
    parser.add_argument("--stage2-vmfb", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--stage1-reference", type=Path)
    parser.add_argument("--stage2-reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/two-stage-vulkan"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.runs <= 0 or args.warmup < 0:
        parser.error("Runs must be positive and warmup nonnegative")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
