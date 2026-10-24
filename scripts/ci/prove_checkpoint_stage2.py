#!/usr/bin/env python3
"""Verify the actual CorridorKey Stage2 checkpoint without loading foundations.

A real runtime checkpoint supplies the source model configuration and all
stage2.* tensors. Instantiate exactly that Stage2 network, load every tensor
strictly, and optionally export/compile and execute its inference graph.

PyTorch is a conversion/reference dependency only; the VMFB runs on Vulkan.
This is a Stage2 proof, not end-to-end foreground/alpha video inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inspect_checkpoint import inspect_checkpoint, model_constructor_defaults


MODEL_SOURCE = Path(__file__).resolve().parents[2] / "corridorkey/model.py"


def stage2_from_config(config: dict):
    """Match CorridorKeyV3's Stage2 constructor argument wiring exactly."""
    from corridorkey.stage2 import Stage2

    return Stage2(
        d=config["d_stage2"],
        d_baton=config["d_16"],
        num_heads=config["num_heads_stage2"],
        num_blocks=config["n_blocks_stage2"],
        num_final_blocks=config["n_final_blocks_stage2"],
        window_size=config["window_size_stage2"],
        mlp_mult=config["mlp_mult"],
        gradient_checkpoint=False,
        overlap_patchify=config["overlap_patchify"],
        overlap_mult=config["overlap_mult"],
        depatch_fourier_features=config["depatch_fourier_features"],
        depatch_fourier_kernel=config["depatch_fourier_kernel"],
        enable_mae_mask_tokens=config["enable_mae_mask_tokens"],
        local_mix_kernel=config["stage2_local_mix_kernel"],
        local_mix_init_std=config["stage2_local_mix_init_std"],
        mlp_dw_kernel=config["mlp_dw_kernel"],
        qkv_dw_kernel=config["qkv_dw_kernel"],
        qkv_dw_fraction=config["qkv_dw_fraction"],
        refine4_blocks=config["refine4_blocks"],
        refine4_width=config["refine4_width"],
        refine4_token_channels=config["refine4_token_channels"],
        refine4_kernel=config["refine4_kernel"],
        refine4_expansion=config["refine4_expansion"],
        rope_shift_coords=config["rope_shift_coords"],
        rope_jitter_coords=config["rope_jitter_coords"],
        rope_rescale_coords=config["rope_rescale_coords"],
        rope_base=config["rope_base"],
    ).eval()


def load_checkpoint_stage2(path: Path):
    """Load only trained Stage2 weights; reject every absent/unexpected key."""
    from safetensors import safe_open

    manifest = inspect_checkpoint(path, MODEL_SOURCE)
    config = model_constructor_defaults(MODEL_SOURCE)
    config.update(manifest["architecture_config"])
    model = stage2_from_config(config)

    expected = set(model.state_dict().keys())
    prefix = "stage2."
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        names = {key[len(prefix):] for key in checkpoint.keys()
                 if key.startswith(prefix)}
        missing = sorted(expected - names)
        unexpected = sorted(names - expected)
        if missing or unexpected:
            raise ValueError(
                "Stage2 checkpoint state differs from the source model: "
                f"missing={missing[:12]}, unexpected={unexpected[:12]}"
            )
        weights = {
            name: checkpoint.get_tensor(prefix + name)
            for name in sorted(expected)
        }
    model.load_state_dict(weights, strict=True)
    return model, config, manifest, len(weights)


def _as_arrays(output):
    items = output if isinstance(output, (tuple, list)) else (output,)
    return tuple(t.detach().cpu().numpy() for t in items)


def _assert_reference(actual, expected, label, atol=3e-3, rtol=3e-3):
    if len(actual) != len(expected):
        raise AssertionError(
            f"{label}: {len(actual)} outputs, expected {len(expected)}"
        )
    errors = []
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a.shape != b.shape or not np.isfinite(a).all():
            raise AssertionError(f"{label}[{index}]: shape or finiteness mismatch")
        difference = np.abs(a - b)
        errors.append({
            "output": index,
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        })
        np.testing.assert_allclose(
            a, b, atol=atol, rtol=rtol,
            err_msg=f"{label}[{index}] differs from checkpoint reference",
        )
    return errors


def self_test():
    """Exercise real safetensors loading, source defaults, and strict keys."""
    from safetensors.torch import save_file

    config = {
        "d_16": 16,
        "d_stage2": 48,
        "num_heads_stage2": 4,
        "n_blocks_stage2": 8,
        "n_final_blocks_stage2": 2,
        "window_size_stage2": 4,
        "stage2_local_mix_kernel": 0,
        "qkv_dw_kernel": 0,
        "mlp_dw_kernel": 0,
        "mlp_mult": 2.0,
    }
    resolved = model_constructor_defaults(MODEL_SOURCE)
    resolved.update(config)
    torch.manual_seed(391)
    source = stage2_from_config(resolved)
    with tempfile.TemporaryDirectory(prefix="corridorkey-stage2-") as tmp:
        checkpoint = Path(tmp) / "test.safetensors"
        state = {
            "stage2." + name: t.detach().clone().contiguous()
            for name, t in source.state_dict().items()
        }
        metadata = {
            "format": "corridorkey_v2_runtime_checkpoint",
            "model_config": json.dumps(config),
            "metadata": "{}",
        }
        save_file(state, str(checkpoint), metadata=metadata)
        loaded, checked, _, count = load_checkpoint_stage2(checkpoint)
        assert count == len(source.state_dict())
        assert checked["d_stage2"] == 48
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor,
                                       atol=0.0, rtol=0.0)

        # This path must work with no IREE compiler installed.
        cpu_args = argparse.Namespace(
            checkpoint=checkpoint, output=Path(tmp) / "cpu-only",
            height=32, width=32, compile=False,
            execute_vulkan=False, target="rdna1",
        )
        if run(cpu_args) != 0:
            raise AssertionError("CPU-only checkpoint inference failed")
        if "iree" in sys.modules:
            raise AssertionError("CPU-only checkpoint inference imported IREE")

        missing = dict(state)
        missing.pop(next(iter(missing)))
        save_file(missing, str(checkpoint), metadata=metadata)
        try:
            load_checkpoint_stage2(checkpoint)
        except ValueError as exc:
            assert "missing=" in str(exc)
        else:
            raise AssertionError("An incomplete Stage2 checkpoint was accepted")
    print("Trained Stage2 checkpoint contract self-test passed")


def run(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(args.checkpoint),
        "scope": "trained Stage2 only (no Stage1 or foundations)",
        "source_weights": False,
        "source_execution": False,
        "export": False,
        "compile": False,
        "vulkan_execute": False,
        "image_height": args.height,
        "image_width": args.width,
    }
    try:
        torch.manual_seed(1942)
        stage2, config, manifest, tensor_count = load_checkpoint_stage2(
            args.checkpoint
        )
        report["source_weights"] = True
        report["state_tensor_count"] = tensor_count
        report["model_config"] = manifest["architecture_config"]
        report["stage2_channels"] = config["d_stage2"]
        report["stage2_blocks"] = config["n_blocks_stage2"]

        height, width = args.height, args.width
        rgb = torch.randn(1, 1, 3, height, width)
        baton = torch.randn(1, 1, config["d_16"], height // 16, width // 16)
        with torch.inference_mode():
            source = _as_arrays(stage2(rgb, baton))
        for index, value in enumerate(source):
            if not np.isfinite(value).all():
                raise AssertionError(f"Source Stage2 output {index} is nonfinite")
            np.save(args.output / f"source_{index}.npy", value)
        report["source_execution"] = True
        report["source_shapes"] = [list(t.shape) for t in source]
        print(f"[checkpoint Stage2] loaded {tensor_count} trained tensors; "
              f"source output shapes={report['source_shapes']}", flush=True)

        if args.compile:
            try:
                import iree.turbine.aot as aot
            except ModuleNotFoundError as exc:
                if exc.name == "iree" or exc.name.startswith("iree."):
                    raise RuntimeError(
                        "Vulkan compilation requires iree-turbine. "
                        "Run: python scripts/setup_amd_importer.py --profile vulkan"
                    ) from exc
                raise
            from prove_stage2_vulkan import ChannelLastStage2Result, export_stage2

            model = ChannelLastStage2Result(stage2).eval()

            exported, specialized = export_stage2(model, rgb, baton)
            with torch.inference_mode():
                output = _as_arrays(exported.module()(rgb, baton))
            output = tuple(t.transpose(0, 1, 4, 2, 3).copy() for t in output)
            report["cpu_export_errors"] = _assert_reference(
                output, source, "Exported Stage2", atol=2e-5, rtol=2e-5,
            )
            graph = aot.export(exported)
            graph.verify()
            graph.save_mlir(args.output / "stage2.mlir")
            report["export"] = True
            report["static_region_profiles"] = specialized

            vmfb = args.output / "stage2.vmfb"
            graph.session.set_flags(f"--iree-vulkan-target={args.target}")
            graph.session.set_flags("--iree-opt-level=O2")
            graph.compile(save_to=vmfb, target_backends=["vulkan-spirv"])
            if not vmfb.is_file() or vmfb.stat().st_size < 100:
                raise AssertionError("Checkpoint Stage2 VMFB was not generated")
            report["compile"] = True
            print("[checkpoint Stage2] compiled trained graph for Vulkan",
                  flush=True)

            if args.execute_vulkan:
                import iree.runtime as rt
                from prove_iree_vulkan import _read_device_result

                device = rt.Config("vulkan")
                module = rt.VmModule.copy_buffer(
                    device.vm_instance, vmfb.read_bytes()
                )
                program = rt.load_vm_module(module, device)
                result = program.main(
                    rt.asdevicearray(device.device, rgb.numpy()),
                    rt.asdevicearray(device.device, baton.numpy()),
                )
                result = result if isinstance(result, (tuple, list)) else (result,)
                actual = tuple(
                    _read_device_result(device, t)
                    .transpose(0, 1, 4, 2, 3).copy()
                    for t in result
                )
                # Preserve divergent outputs and error statistics even when
                # the Vulkan numerical assertion fails.
                for index, value in enumerate(actual):
                    np.save(args.output / f"vulkan_{index}.npy", value)
                report["vulkan_error_stats"] = [
                    {
                        "output": index,
                        "max_absolute_error": float(np.max(np.abs(a - b))),
                        "mean_absolute_error": float(np.mean(np.abs(a - b))),
                        "mismatched": int(np.sum(~np.isclose(
                            a, b, atol=3e-3, rtol=3e-3,
                        ))),
                    }
                    for index, (a, b) in enumerate(zip(actual, source))
                ]
                report["vulkan_errors"] = _assert_reference(
                    actual, source, "Vulkan Stage2",
                )
                report["vulkan_execute"] = True
                print("[checkpoint Stage2] trained Vulkan graph matches source",
                      flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 1 if "error" in report else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("build/checkpoint-stage2"))
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--execute-vulkan", action="store_true")
    parser.add_argument("--target", default="rdna1")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.checkpoint is None:
        parser.error("Provide a runtime safetensors checkpoint or --self-test")
    if args.checkpoint.suffix != ".safetensors":
        parser.error("This proof requires a runtime .safetensors checkpoint")
    if args.height <= 0 or args.width <= 0 or (
        args.height % 16 or args.width % 16
    ):
        parser.error("--height and --width must be positive multiples of 16")
    if args.execute_vulkan and not args.compile:
        parser.error("--execute-vulkan requires --compile")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
