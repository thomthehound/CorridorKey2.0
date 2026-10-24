#!/usr/bin/env python3
"""Build and execute the real checkpoint's Stage1 -> Stage2 Vulkan pipeline.

The importer uses CPU PyTorch and source model definitions; the *execution*
process receives only compiled Vulkan VMFBs and NumPy arrays. Foundation
feature extraction and video scheduling are separate, still-unported
partitions. This proves the two trained neural stages, not full video matting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent


def invoke(label, argv):
    print(f"[trained pipeline] {label}", flush=True)
    subprocess.run([sys.executable, *map(str, argv)], check=True)


def read_report(path):
    if not path.is_file():
        raise FileNotFoundError(f"Expected proof report missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("error"):
        raise AssertionError(f"{path}: {payload['error']}")
    return payload


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    stage1_dir = args.output / "stage1"
    stage2_dir = args.output / "stage2"
    execution_dir = args.output / "execution"
    report = {
        "checkpoint": str(args.checkpoint),
        "scope": "real trained Stage1+Stage2, normalized foundation features",
        "gpu_target": args.target,
        "rms_reduction": "segmented128",
        "down_reduction": args.down_reduction,
        "foundation_networks_executed": False,
        "native_cpp_runtime": False,
        "stage1": False,
        "stage2": False,
        "stages_executed_without_pytorch": False,
    }
    try:
        try:
            invoke("Compile and prove real trained Stage1 on Vulkan", [
                SCRIPTS / "prove_checkpoint_stage1.py",
                args.checkpoint, "--frames", "1",
                "--height", args.height, "--width", args.width,
                "--target", args.target, "--rms-reduction", "segmented128",
                "--down-reduction", args.down_reduction,
                "--compile", "--execute-vulkan", "--output", stage1_dir,
            ])
        except subprocess.CalledProcessError:
            if args.diagnose_stage1:
                # The diagnostic recompiles the identical trained graph with
                # intermediate tensor readbacks. Its nonzero exit is expected
                # when it successfully identifies an incorrect GPU boundary.
                diagnostic_dir = args.output / "stage1_diagnostic"
                diagnostic_cmd = [
                    sys.executable, str(SCRIPTS / "diagnose_stage1_vulkan.py"),
                    "--checkpoint", str(args.checkpoint),
                    "--frames", "1",
                    "--height", str(args.height), "--width", str(args.width),
                    "--target", str(args.target),
                    "--rms-reduction", "segmented128",
                    "--down-reduction", args.down_reduction,
                    "--output", str(diagnostic_dir),
                ]
                print(
                    "[trained pipeline] Stage1 GPU proof failed; collecting "
                    "original-graph intermediate differences", flush=True,
                )
                diagnostic = subprocess.run(diagnostic_cmd, check=False)
                report["stage1_diagnostic_exit_code"] = diagnostic.returncode
                if (diagnostic_dir / "report.json").is_file():
                    result = read_report(diagnostic_dir / "report.json") if (
                        diagnostic.returncode == 0
                    ) else json.loads(
                        (diagnostic_dir / "report.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    report["stage1_first_divergence"] = result.get(
                        "first_divergence"
                    )
                    report["stage1_diagnostic_report"] = str(
                        diagnostic_dir / "report.json"
                    )
            raise
        stage1 = read_report(stage1_dir / "report.json")
        if not stage1.get("vulkan_execute"):
            raise AssertionError("Stage1 hardware proof did not complete")
        report["stage1"] = True
        report["stage1_tensors"] = stage1["tensor_count"]
        report["feature_channels"] = stage1["foundation_feature_dimensions"]

        invoke("Compile and prove Stage2 against actual trained Stage1 baton", [
            SCRIPTS / "prove_checkpoint_stage2.py",
            args.checkpoint, "--height", args.height, "--width", args.width,
            "--target", args.target, "--rms-reduction", "segmented128",
            "--rgb-input", stage1_dir / "input_0.npy",
            "--baton-input", stage1_dir / "source_0.npy",
            "--compile", "--execute-vulkan", "--output", stage2_dir,
        ])
        stage2 = read_report(stage2_dir / "report.json")
        if not stage2.get("vulkan_execute"):
            raise AssertionError("Stage2 hardware proof did not complete")
        report["stage2"] = True
        report["stage2_tensors"] = stage2["state_tensor_count"]

        invoke("Run trained Stage1 then Stage2 with IREE-only host", [
            SCRIPTS / "execute_compiled_stages.py",
            "--stage1-vmfb", stage1_dir / "stage1.vmfb",
            "--stage2-vmfb", stage2_dir / "stage2.vmfb",
            "--inputs", stage1_dir, "--stage1-reference", stage1_dir,
            "--stage2-reference", stage2_dir,
            "--runs", args.runs, "--warmup", args.warmup,
            "--output", execution_dir,
        ])
        execution = read_report(execution_dir / "report.json")
        if not execution.get("success") or not execution.get("torch_runtime") is False:
            raise AssertionError("IREE-only chain failed or imported PyTorch")
        report["stages_executed_without_pytorch"] = True
        report["execution_wall_clock_median_ms"] = (
            execution["wall_clock_total_median_ms"]
        )
        report["host_staging"] = execution["interstage_transfer"]
        print(
            "[trained pipeline] source-faithful trained Stage1 -> Stage2 "
            "Vulkan execution passed",
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
    return int("error" in report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--target", default="rdna3")
    parser.add_argument("--down-reduction", choices=("native", "segmented128"),
                        default="native")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--no-stage1-diagnostic", dest="diagnose_stage1",
        action="store_false",
        help="Skip automatic source-graph boundary localization after a "
             "trained Stage1 GPU failure",
    )
    parser.set_defaults(diagnose_stage1=True)
    parser.add_argument(
        "--output", type=Path, default=Path("build/trained-two-stage")
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    if min(args.height, args.width) <= 0 or (
        args.height % 64 or args.width % 64
    ):
        parser.error("Frame dimensions must be positive multiples of 64")
    if args.runs <= 0 or args.warmup < 0:
        parser.error("--runs must be positive and --warmup nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
