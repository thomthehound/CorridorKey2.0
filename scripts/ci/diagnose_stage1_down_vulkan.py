#!/usr/bin/env python3
"""Differential Vulkan proof for the checkpoint's original Stage1 /16->/64 convolution.

Runs the trained convolution with its actual source encoder activation. Compares
native Conv2d, a reduced-kernel Conv2d, and matrix-multiply formulations
without modifying checkpoint parameters or source-model inference.
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
import torch.nn.functional as F


class DownsampleVariant(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, mode: str):
        super().__init__()
        if mode not in ("native", "im2col", "crop4", "gemm4", "gemm4runtime",
                         "seg128", "spatial", "unfold"):
            raise ValueError(f"Unknown convolution variant: {mode}")
        if mode in ("crop4", "gemm4", "gemm4runtime", "seg128"):
            # This exact reduction holds only for one output cell: at 4x4
            # input, the padded k8/s4 convolution visits only its center taps.
            if (conv.kernel_size != (8, 8) or conv.stride != (4, 4)
                    or conv.padding != (2, 2) or conv.dilation != (1, 1)):
                raise ValueError("4x4 variants require k8/s4/p2/d1")
        if mode in ("crop4", "gemm4", "gemm4runtime", "seg128",
                    "spatial", "unfold") and conv.groups != 1:
            raise ValueError("Matmul variants require groups=1")
        self.conv = conv
        self.mode = mode

    def forward(self, x, runtime_weight=None, runtime_bias=None):
        if self.mode in ("native", "im2col"):
            return self.conv(x)
        conv = self.conv
        if self.mode in ("crop4", "gemm4", "gemm4runtime", "seg128"):
            if x.shape[-2:] != (4, 4):
                raise ValueError("4x4 diagnostic variants require 4x4 activations")
            # For k8/s4/p2 and 4x4 input, only taps [2:6,2:6]
            # overlap the image. These are slices of the trained weights.
            if self.mode == "gemm4runtime":
                if runtime_weight is None or runtime_bias is None:
                    raise ValueError("Runtime GEMM needs weight and bias inputs")
                weight = runtime_weight
                bias = runtime_bias
            else:
                weight = conv.weight[:, :, 2:6, 2:6]
                bias = conv.bias
            if self.mode == "crop4":
                return F.conv2d(x, weight, bias, stride=4)
            flattened = x.reshape(x.shape[0], -1)
            matrix = weight.reshape(conv.out_channels, -1)
            if self.mode == "seg128":
                # Partition the original 8192-element dot product into
                # independent <=128-element GEMMs, then sum partial values.
                # No compensation factors or modifications of trained weights.
                reduction_width = flattened.shape[-1]
                segment_width = min(128, reduction_width)
                if reduction_width % segment_width:
                    raise ValueError("Segment width must exactly divide features")
                segments = reduction_width // segment_width
                lhs = flattened.reshape(
                    x.shape[0], segments, segment_width
                ).transpose(0, 1)
                rhs = matrix.reshape(
                    conv.out_channels, segments, segment_width
                ).permute(1, 2, 0)
                output = torch.bmm(lhs, rhs).sum(dim=0)
                if bias is not None:
                    output = output + bias
            else:
                output = F.linear(flattened, matrix, bias)
            return output.reshape(x.shape[0], conv.out_channels, 1, 1)

        height = (
            x.shape[-2] + 2 * conv.padding[0]
            - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1
        ) // conv.stride[0] + 1
        width = (
            x.shape[-1] + 2 * conv.padding[1]
            - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1
        ) // conv.stride[1] + 1
        if self.mode == "unfold":
            # This native aten.im2col path remains selectable to track IREE
            # compiler support; it currently fails compilation.
            patches = F.unfold(
                x, kernel_size=conv.kernel_size, dilation=conv.dilation,
                padding=conv.padding, stride=conv.stride,
            )
        else:
            # Explicit padded spatial gathers, not aten.im2col. This
            # formulation is exact for all supported input resolutions.
            padded = F.pad(x, (
                conv.padding[1], conv.padding[1],
                conv.padding[0], conv.padding[0],
            ))
            taps = [
                padded[:, :, ky * conv.dilation[0]:
                       ky * conv.dilation[0] + height * conv.stride[0]:
                       conv.stride[0],
                       kx * conv.dilation[1]:
                       kx * conv.dilation[1] + width * conv.stride[1]:
                       conv.stride[1]]
                for ky in range(conv.kernel_size[0])
                for kx in range(conv.kernel_size[1])
            ]
            patches = torch.stack(taps, dim=2).permute(
                0, 3, 4, 1, 2
            ).reshape(x.shape[0], height * width, -1).transpose(1, 2)
        values = torch.matmul(
            patches.transpose(1, 2), conv.weight.flatten(1).transpose(0, 1)
        )
        if conv.bias is not None:
            values = values + conv.bias
        return values.transpose(1, 2).reshape(
            x.shape[0], conv.out_channels, height, width
        )



def variant_inputs(mode, conv, activation):
    if mode == "gemm4runtime":
        if conv.bias is None:
            raise ValueError("Runtime GEMM currently requires bias")
        # Clone parameters into independent, explicit runtime tensors.
        return (
            activation,
            conv.weight.detach()[:, :, 2:6, 2:6].contiguous().clone(),
            conv.bias.detach().clone(),
        )
    return (activation,)


def self_test_spatial():
    """Verify spatial gather order against the original convolution geometry."""
    torch.manual_seed(20261024)
    conv = torch.nn.Conv2d(8, 8, kernel_size=8, stride=4, padding=2).eval()
    cases = ((4, 4), (4, 8), (8, 4), (8, 8), (12, 12))
    for height, width in cases:
        x = torch.randn(2, 8, height, width)
        with torch.inference_mode():
            expected = conv(x).numpy()
        modes = (
            "spatial", "crop4", "gemm4", "gemm4runtime", "seg128"
        ) if (height, width) == (4, 4) else ("spatial",)
        for mode in modes:
            variant = DownsampleVariant(conv, mode).eval()
            with torch.inference_mode():
                call_args = variant_inputs(mode, conv, x)
                direct = variant(*call_args).numpy()
                exported = torch.export.export(
                    variant, call_args, strict=True
                ).module()(*call_args).detach().numpy()
            np.testing.assert_allclose(
                direct, expected, atol=2e-5, rtol=2e-5,
                err_msg=f"{mode}: shape {(height, width)}",
            )
            np.testing.assert_allclose(
                exported, expected, atol=2e-5, rtol=2e-5,
                err_msg=f"{mode}: exported shape {(height, width)}",
            )
    print("[Stage1 down] source-equivalent spatial layouts pass CPU/export",
          flush=True)


def make_fixture():
    torch.manual_seed(20261024)
    conv = torch.nn.Conv2d(8, 8, kernel_size=8, stride=4, padding=2).eval()
    return conv, torch.randn(1, 8, 4, 4)


def load_trained_downsample(checkpoint, height, width, frames):
    from prove_checkpoint_stage1 import (
        load_checkpoint_stage1, make_inputs, reference_outputs,
    )

    stage1, config, _, dims, count = load_checkpoint_stage1(checkpoint)
    inputs = make_inputs(config, dims, frames, height, width)
    captured = []

    def capture_input(_module, args):
        captured.append(args[0].detach().clone())

    handle = stage1.down.conv.register_forward_pre_hook(capture_input)
    try:
        reference_outputs(stage1, inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise AssertionError(
            f"Expected exactly one Stage1 down convolution, got {len(captured)}"
        )
    return stage1.down.conv, captured[0], count


def accumulation_analysis(actual, expected, bias):
    """Distinguish duplicated MAC accumulation from repeated bias addition."""
    y = np.asarray(expected, dtype=np.float64).reshape(-1)
    gpu = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(bias, dtype=np.float64).reshape(-1)
    if y.size != b.size or gpu.size != b.size:
        return None
    dot = y - b
    gpu_dot = gpu - b
    denominator = np.dot(dot, dot)
    scale = float(np.dot(dot, gpu_dot) / denominator) if denominator else None
    doubled = 2.0 * dot + b
    residual = gpu - doubled
    return {
        "dot_scale_least_squares": scale,
        "twice_dot_plus_bias_max_error": float(np.max(np.abs(residual))),
        "twice_dot_plus_bias_mean_error": float(np.mean(np.abs(residual))),
        "twice_dot_plus_bias_mismatched": int(np.count_nonzero(
            ~np.isclose(gpu, doubled, atol=3e-3, rtol=3e-3)
        )),
        "twice_dot_plus_bias_samples": int(gpu.size),
        "bias_head": [float(v) for v in b[:8]],
        "residual_head": [float(v) for v in residual[:8]],
    }


def stats(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("Downsample result has incorrect shape or nonfinite values")
    delta = np.abs(actual - expected)
    return {
        "max_absolute_error": float(delta.max()),
        "mean_absolute_error": float(delta.mean()),
        "mismatched": int(np.count_nonzero(
            ~np.isclose(actual, expected, atol=3e-3, rtol=3e-3)
        )),
        "total": int(expected.size),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "target": args.target,
        "height": args.height, "width": args.width, "frames": args.frames,
        "modes": args.modes, "variants": {},
    }
    try:
        if args.checkpoint:
            conv, activation, count = load_trained_downsample(
                args.checkpoint, args.height, args.width, args.frames
            )
            report["source"] = "trained Stage1 convolution and encoder activation"
            report["trained_tensors"] = count
        else:
            conv, activation = make_fixture()
            report["source"] = "deterministic synthetic convolution fixture"
        conv.eval()
        with torch.inference_mode():
            reference = conv(activation).detach().numpy().copy()
        np.save(args.output / "down_input_cpu.npy", activation.numpy())
        np.save(args.output / "down_output_cpu.npy", reference)
        report["input_shape"] = list(activation.shape)
        report["output_shape"] = list(reference.shape)
        report["weight_shape"] = list(conv.weight.shape)
        print(
            f"[Stage1 down] input={tuple(activation.shape)} "
            f"weights={tuple(conv.weight.shape)} "
            f"output={tuple(reference.shape)}", flush=True,
        )

        for mode in args.modes:
            entry = report["variants"][mode] = {}
            try:
                variant = DownsampleVariant(conv, mode).eval()
                call_args = variant_inputs(mode, conv, activation)
                with torch.inference_mode():
                    direct = variant(*call_args).detach().numpy().copy()
                entry["cpu"] = stats(direct, reference)
                np.testing.assert_allclose(
                    direct, reference, atol=2e-5, rtol=2e-5,
                    err_msg=f"{mode} changed source convolution on CPU",
                )
                exported = torch.export.export(variant, call_args, strict=True)
                with torch.inference_mode():
                    exported_cpu = exported.module()(*call_args).detach().numpy()
                if mode == "gemm4runtime":
                    from torch.export.graph_signature import InputKind
                    if any(
                        spec.kind == InputKind.PARAMETER
                        for spec in exported.graph_signature.input_specs
                    ):
                        raise AssertionError(
                            "Runtime GEMM accidentally captured module parameters"
                        )
                np.testing.assert_allclose(
                    exported_cpu, reference, atol=2e-5, rtol=2e-5,
                    err_msg=f"{mode} changed source convolution during export",
                )
                entry["cpu_export_passed"] = True
                print(f"[Stage1 down] {mode} CPU/export passed", flush=True)
                if args.probe_only:
                    continue

                import iree.turbine.aot as aot

                compiled = aot.export(exported)
                compiled.verify()
                compiled.save_mlir(args.output / f"{mode}.mlir")
                entry["imported"] = True
                print(f"[Stage1 down] {mode} Torch-MLIR import passed", flush=True)
                if args.import_only:
                    continue
                compiled.session.set_flags(f"--iree-vulkan-target={args.target}")
                compiled.session.set_flags("--iree-opt-level=O2")
                if mode == "im2col":
                    compiled.session.set_flags(
                        "--iree-global-opt-use-im2col-for-convs=true"
                    )
                vmfb = args.output / f"{mode}.vmfb"
                compiled.compile(save_to=vmfb, target_backends=["vulkan-spirv"])
                entry["compiled"] = True
                print(f"[Stage1 down] {mode} Vulkan compiled", flush=True)

                import iree.runtime as rt
                from prove_iree_vulkan import _read_device_result

                config = rt.Config("vulkan")
                module = rt.VmModule.copy_buffer(
                    config.vm_instance, vmfb.read_bytes()
                )
                program = rt.load_vm_module(module, config)
                result = program.main(*(
                    rt.asdevicearray(config.device, tensor.detach().numpy())
                    for tensor in call_args
                ))
                if isinstance(result, (tuple, list)):
                    if len(result) != 1:
                        raise AssertionError("Expected one downsample result")
                    result = result[0]
                actual = _read_device_result(config, result)
                np.save(args.output / f"{mode}_vulkan.npy", actual)
                entry["vulkan"] = stats(actual, reference)
                if conv.bias is not None:
                    entry["accumulation_analysis"] = accumulation_analysis(
                        actual, reference, conv.bias.detach().numpy()
                    )
                    info = entry["accumulation_analysis"]
                    if info is not None:
                        print(
                            f"[Stage1 down] {mode} accumulated scale="
                            f"{info['dot_scale_least_squares']:.8g}, "
                            f"double-dot residual max="
                            f"{info['twice_dot_plus_bias_max_error']:.8g}, "
                            f"mismatched="
                            f"{info['twice_dot_plus_bias_mismatched']}/"
                            f"{info['twice_dot_plus_bias_samples']}",
                            flush=True,
                        )
                entry["vulkan_passed"] = entry["vulkan"]["mismatched"] == 0
                print(
                    f"[Stage1 down] {mode} Vulkan: "
                    f"max={entry['vulkan']['max_absolute_error']:.8g}, "
                    f"mismatched={entry['vulkan']['mismatched']}/"
                    f"{entry['vulkan']['total']}", flush=True,
                )
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()
                print(entry["traceback"], file=sys.stderr, flush=True)
        return int(any(
            "error" in variant or (
                not args.probe_only and not args.import_only
                and not variant.get("vulkan_passed", False)
            )
            for variant in report["variants"].values()
        ))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-spatial", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--target", default="rdna3")
    parser.add_argument(
        "--modes", default="native,gemm4,gemm4runtime,seg128",
        help="Comma-separated variants: native,im2col,crop4,gemm4,"
             "gemm4runtime,seg128,spatial,unfold",
    )
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--probe-only", action="store_true")
    group.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    if args.self_test_spatial:
        self_test_spatial()
        return 0
    args.modes = tuple(part.strip() for part in args.modes.split(","))
    if not args.modes or any(
        part not in ("native", "im2col", "crop4", "gemm4", "gemm4runtime",
                     "seg128", "spatial", "unfold")
        for part in args.modes
    ) or len(set(args.modes)) != len(args.modes):
        parser.error(
            "--modes must select unique members of "
            "native,im2col,crop4,gemm4,gemm4runtime,seg128,spatial,unfold"
        )
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.frames < 1 or args.height < 1 or args.width < 1 or (
        args.height % 64 or args.width % 64
    ):
        parser.error("Height and width must be positive multiples of 64")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
