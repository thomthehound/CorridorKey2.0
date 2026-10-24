#!/usr/bin/env python3
"""Execute the original trained Stage1 checkpoint and export it to Vulkan.

Stage1 consumes normalized *features* from MoGe/C-RADIO/RVM, not their RGB
foundation-model inputs. This partitions the real Stage1 graph without
loading or approximating the external foundation networks. Deterministic
feature tensors are test inputs, not substitutes for trained Stage1 weights.

CPU-only strict state loading and execution do not require IREE.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from inspect_checkpoint import inspect_checkpoint, model_constructor_defaults


MODEL_SOURCE = Path(__file__).resolve().parents[2] / "corridorkey/model.py"


def stage1_from_config(config, *, cradio_dim, moge_dim, rvm_dim):
    """Mirror CorridorKeyV3's Stage1 construction with explicit feature ABI."""
    from corridorkey.stage1 import Stage1

    return Stage1(
        d_16=config["d_16"],
        d_64=config["d_64"],
        cradio_dim=cradio_dim,
        moge_dim=moge_dim,
        rvm_dim=rvm_dim,
        n_enc_16=config["n_enc_16"],
        n_body_64=config["n_body_64"],
        n_dec_16=config["n_dec_16"],
        num_heads_16=config["num_heads_16"],
        num_heads_64=config["num_heads_64"],
        window_HW=config["window_HW"],
        mlp_mult=config["mlp_mult"],
        local_mix_kernel=config["stage1_local_mix_kernel"],
        local_mix_init_std=config["stage1_local_mix_init_std"],
        body64_local_mix_kernel=config["stage1_body64_local_mix_kernel"],
        body64_local_mix_init_std=config["stage1_body64_local_mix_init_std"],
        body64_mlp_hidden=config["stage1_body64_mlp_hidden"],
        body64_direct_concat=config["stage1_body64_direct_concat"],
        recurrent_64_sites=config["stage1_recurrent_64_sites"],
        hint_adaln_blocks=config["stage1_hint_adaln_blocks"],
        mid_16_blocks=config["stage1_mid_16_blocks"],
        mlp_dw_kernel=config["mlp_dw_kernel"],
        qkv_dw_kernel=config["qkv_dw_kernel"],
        qkv_dw_fraction=config["qkv_dw_fraction"],
        gradient_checkpoint=False,
        overlap_patchify=config["overlap_patchify"],
        overlap_mult=config["overlap_mult"],
        depatch_fourier_features=config["depatch_fourier_features"],
        depatch_fourier_kernel=config["depatch_fourier_kernel"],
        enable_mae_mask_tokens=config["enable_mae_mask_tokens"],
        bottleneck_head_stride=config["bottleneck_head_stride"],
        rope_shift_coords=config["rope_shift_coords"],
        rope_jitter_coords=config["rope_jitter_coords"],
        rope_rescale_coords=config["rope_rescale_coords"],
        rope_base=config["rope_base"],
    ).eval()


def _foundation_dimensions(config, checkpoint):
    """Resolve feature ABI using the checkpoint's *actual* fusion weights.

    MoGe and small RVM have stable source output widths. C-RADIO's width
    depends on the selected snapshot; infer it from the trained fusion
    projection instead of loading remote foundation code or guessing.
    """
    from safetensors import safe_open
    from corridorkey.rvm import RVM_CFG

    moge = 768 if config["use_moge"] else 0
    rvm = (
        int(RVM_CFG[config["rvm_variant"]]["embed_dim"])
        if config["rvm_ckpt"] else 0
    )
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        if config["stage1_body64_direct_concat"]:
            input_width = int(config["d_64"])
        else:
            key = "stage1.fuse.linear_shortcut.weight"
            if key not in handle.keys():
                raise ValueError(f"Missing learned feature fusion tensor {key}")
            dims = handle.get_slice(key).get_shape()
            if len(dims) != 2 or dims[0] != config["d_64"]:
                raise ValueError(f"Invalid trained Stage1 fusion shape: {dims}")
            input_width = dims[1]

    foundation_width = input_width - config["d_16"]
    cradio = foundation_width - moge - rvm
    if cradio < 0 or (config["use_cradio"] and cradio <= 0) or (
        not config["use_cradio"] and cradio != 0
    ):
        raise ValueError(
            "Stage1 feature dimensions inconsistent with configured "
            f"foundations: fused_width={input_width}, d16={config['d_16']}, "
            f"moge={moge}, rvm={rvm}, inferred_cradio={cradio}, "
            f"use_cradio={config['use_cradio']}"
        )
    return {"cradio": cradio, "moge": moge, "rvm": rvm}


def load_checkpoint_stage1(checkpoint):
    """Strictly load original Stage1 tensors without loading foundations."""
    from safetensors import safe_open

    manifest = inspect_checkpoint(checkpoint, MODEL_SOURCE)
    config = model_constructor_defaults(MODEL_SOURCE)
    config.update(manifest["architecture_config"])
    dims = _foundation_dimensions(config, checkpoint)
    stage1 = stage1_from_config(
        config, cradio_dim=dims["cradio"],
        moge_dim=dims["moge"], rvm_dim=dims["rvm"],
    )
    expected = set(stage1.state_dict())
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = {key[len("stage1."):] for key in handle.keys()
                if key.startswith("stage1.")}
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing or extra:
            raise ValueError(
                "Trained Stage1 weights disagree with source architecture: "
                f"missing={missing[:16]}, unexpected={extra[:16]}"
            )
        weights = {
            name: handle.get_tensor("stage1." + name)
            for name in sorted(expected)
        }
    stage1.load_state_dict(weights, strict=True)
    return stage1, config, manifest, dims, len(weights)


def make_inputs(config, dims, frames, height, width):
    """Deterministic Stage1 boundary inputs with source-correct dimensions."""
    torch.manual_seed(1942)
    shape = (1, frames)
    grid = (height // 64, width // 64)
    rgb = torch.randn(*shape, 3, height, width)
    hint = torch.rand(*shape, 1, height, width)
    moge = torch.randn(*shape, dims["moge"], *grid)
    cradio = torch.randn(*shape, dims["cradio"], *grid)
    rvm = torch.randn(*shape, dims["rvm"], *grid)
    quality = torch.ones(*shape)
    return (rgb, hint, moge, cradio, rvm, quality)


class TrainedStage1ABI(torch.nn.Module):
    """Preserve the source Stage1 graph and expose channel-last output tensors."""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, rgb, hint, moge, cradio, rvm, quality):
        outputs = self.source(
            rgb, hint, moge,
            cradio_feat=cradio if self.source.cradio_dim else None,
            rvm_feat=rvm if self.source.rvm_dim else None,
            hint_quality=quality, t_stride=1.0,
        )
        result = [outputs["baton"], outputs["aux"]]
        if outputs.get("bottleneck") is not None:
            result.append(outputs["bottleneck"])
        return tuple(
            value.permute(0, 1, 3, 4, 2).contiguous()
            for value in result
        )


def reference_outputs(stage1, inputs):
    rgb, hint, moge, cradio, rvm, quality = inputs
    with torch.inference_mode():
        result = stage1(
            rgb, hint, moge,
            cradio_feat=cradio if stage1.cradio_dim else None,
            rvm_feat=rvm if stage1.rvm_dim else None,
            hint_quality=quality, t_stride=1.0,
        )
    tensors = [result["baton"], result["aux"]]
    if result.get("bottleneck") is not None:
        tensors.append(result["bottleneck"])
    return tuple(value.detach().numpy().copy() for value in tensors)


def _compare(actual, reference, name, *, atol, rtol):
    if len(actual) != len(reference):
        raise AssertionError(f"{name}: output arity changed")
    errors = []
    for index, (value, expected) in enumerate(zip(actual, reference)):
        value = np.asarray(value)
        if value.shape != expected.shape or not np.isfinite(value).all():
            raise AssertionError(f"{name}[{index}]: invalid shape or NaN/Inf")
        delta = np.abs(value - expected)
        errors.append({
            "output": index, "max_absolute_error": float(delta.max()),
            "mean_absolute_error": float(delta.mean()),
            "mismatched": int(np.count_nonzero(
                ~np.isclose(value, expected, atol=atol, rtol=rtol)
            )),
        })
        np.testing.assert_allclose(
            value, expected, atol=atol, rtol=rtol,
            err_msg=f"{name}[{index}] differs from trained source",
        )
    return errors


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "original trained Stage1 graph; foundation features as inputs",
        "checkpoint": str(args.checkpoint),
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "rms_reduction": args.rms_reduction,
        "down_reduction": args.down_reduction,
        "compiled": False,
        "vulkan_execute": False,
    }
    try:
        stage1, config, _manifest, dims, count = load_checkpoint_stage1(
            args.checkpoint
        )
        report["tensor_count"] = count
        report["foundation_feature_dimensions"] = dims
        print(
            f"[checkpoint Stage1] loaded {count} trained tensors; "
            f"feature channels={dims}", flush=True,
        )
        inputs = make_inputs(
            config, dims, args.frames, args.height, args.width
        )
        for index, tensor in enumerate(inputs):
            np.save(args.output / f"input_{index}.npy", tensor.numpy())
        report["input_shapes"] = [list(t.shape) for t in inputs]
        source = reference_outputs(stage1, inputs)
        for index, value in enumerate(source):
            np.save(args.output / f"source_{index}.npy", value)
        report["source_shapes"] = [list(v.shape) for v in source]
        report["source_execution"] = True
        print(
            f"[checkpoint Stage1] original source output "
            f"shapes={report['source_shapes']}", flush=True,
        )
        if args.compile:
            import iree.turbine.aot as aot
            import iree.runtime as rt

            from prove_stage1_vulkan import export_stage1
            from prove_iree_vulkan import _read_device_result

            abi = TrainedStage1ABI(stage1).eval()
            exported, signatures = export_stage1(
                abi, inputs, rms_reduction=args.rms_reduction,
                down_reduction=args.down_reduction,
            )
            report["shifted_attention_signatures"] = signatures
            with torch.inference_mode():
                output = exported.module()(*inputs)
            cpu_values = tuple(
                t.detach().numpy().transpose(0, 1, 4, 2, 3).copy()
                for t in output
            )
            report["export_errors"] = _compare(
                cpu_values, source, "exported Stage1",
                atol=2e-5, rtol=2e-5,
            )
            compiled = aot.export(exported)
            compiled.verify()
            compiled.save_mlir(args.output / "stage1.mlir")
            compiled.session.set_flags(
                f"--iree-vulkan-target={args.target}"
            )
            compiled.session.set_flags("--iree-opt-level=O2")
            vmfb = args.output / "stage1.vmfb"
            compiled.compile(
                save_to=vmfb, target_backends=["vulkan-spirv"]
            )
            if not vmfb.exists() or vmfb.stat().st_size < 100:
                raise AssertionError("Trained Stage1 VMFB was not generated")
            report["compiled"] = True
            print("[checkpoint Stage1] trained Vulkan graph compiled", flush=True)
            if args.execute_vulkan:
                cfg = rt.Config("vulkan")
                module = rt.VmModule.copy_buffer(
                    cfg.vm_instance, vmfb.read_bytes()
                )
                program = rt.load_vm_module(module, cfg)
                output = program.main(*[
                    rt.asdevicearray(cfg.device, tensor.numpy())
                    for tensor in inputs
                ])
                if not isinstance(output, (tuple, list)):
                    output = (output,)
                gpu_values = tuple(
                    _read_device_result(cfg, tensor)
                    .transpose(0, 1, 4, 2, 3).copy()
                    for tensor in output
                )
                for index, value in enumerate(gpu_values):
                    np.save(args.output / f"vulkan_{index}.npy", value)
                report["vulkan_error_stats"] = [
                    {
                        "output": index,
                        "max_absolute_error": float(
                            np.max(np.abs(value - reference))
                        ),
                        "mean_absolute_error": float(
                            np.mean(np.abs(value - reference))
                        ),
                        "mismatched": int(np.count_nonzero(
                            ~np.isclose(
                                value, reference, atol=3e-3, rtol=3e-3
                            )
                        )),
                    }
                    for index, (value, reference) in enumerate(
                        zip(gpu_values, source)
                    )
                ]
                report["vulkan_errors"] = _compare(
                    gpu_values, source, "Vulkan Stage1",
                    atol=3e-3, rtol=3e-3,
                )
                report["vulkan_execute"] = True
                print("[checkpoint Stage1] trained Vulkan graph matches source",
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
    return int("error" in report)


def self_test():
    from safetensors.torch import save_file

    cfg = model_constructor_defaults(MODEL_SOURCE)
    cfg.update({
        "d_16": 64, "d_64": 128,
        "n_enc_16": 1, "n_body_64": 2, "n_dec_16": 6,
        "num_heads_16": 4, "num_heads_64": 8,
        "window_HW": 4, "stage1_local_mix_kernel": 0,
        "stage1_body64_local_mix_kernel": 0,
        "stage1_mid_16_blocks": 2,
        "stage1_hint_adaln_blocks": 1,
        "qkv_dw_kernel": 0, "mlp_dw_kernel": 0,
        "mlp_mult": 2.0, "overlap_mult": 1,
        "use_moge": False, "use_cradio": False,
        "rvm_ckpt": None, "bottleneck_head_stride": None,
        "stage1_recurrent_64_sites": [],
    })
    torch.manual_seed(391)
    original = stage1_from_config(
        cfg, cradio_dim=0, moge_dim=0, rvm_dim=0
    )
    with tempfile.TemporaryDirectory(prefix="corridorkey-stage1-") as tmp:
        path = Path(tmp) / "weights.safetensors"
        tensors = {
            "stage1." + name: value.detach().contiguous().clone()
            for name, value in original.state_dict().items()
        }
        metadata = {
            "format": "corridorkey_v2_runtime_checkpoint",
            "model_config": json.dumps({
                name: cfg[name] for name in (
                    "d_16", "d_64", "n_enc_16", "n_body_64",
                    "n_dec_16", "num_heads_16", "num_heads_64",
                    "window_HW", "stage1_local_mix_kernel",
                    "stage1_body64_local_mix_kernel", "stage1_mid_16_blocks",
                    "stage1_hint_adaln_blocks", "qkv_dw_kernel",
                    "mlp_dw_kernel", "mlp_mult", "overlap_mult",
                    "use_moge", "use_cradio", "rvm_ckpt",
                    "bottleneck_head_stride", "stage1_recurrent_64_sites"
                )
            }),
            "metadata": "{}",
        }
        save_file(tensors, str(path), metadata=metadata)
        loaded, config, _, dims, count = load_checkpoint_stage1(path)
        if count != len(tensors) or dims != {
            "cradio": 0, "moge": 0, "rvm": 0
        }:
            raise AssertionError("Stage1 self-test lost checkpoint ABI")
        for name, original_value in original.state_dict().items():
            torch.testing.assert_close(
                loaded.state_dict()[name], original_value, rtol=0, atol=0
            )
        inputs = make_inputs(config, dims, 1, 64, 64)
        _compare(
            reference_outputs(loaded, inputs),
            reference_outputs(original, inputs),
            "self-test trained Stage1", atol=0, rtol=0,
        )
        if "iree" in sys.modules:
            raise AssertionError("CPU-only Stage1 test imported IREE")
        broken = dict(tensors)
        broken.pop(next(iter(broken)))
        save_file(broken, str(path), metadata=metadata)
        try:
            load_checkpoint_stage1(path)
        except ValueError as exc:
            if "missing=" not in str(exc):
                raise
        else:
            raise AssertionError("Incomplete Stage1 state was accepted")
    print("[checkpoint Stage1] strict CPU checkpoint contract passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--execute-vulkan", action="store_true")
    parser.add_argument("--target", default="rdna3")
    parser.add_argument(
        "--rms-reduction", choices=("native", "segmented128"),
        default="segmented128",
    )
    parser.add_argument(
        "--down-reduction", choices=("native", "segmented128"),
        default="native", help="Export-only Stage1 downsample lowering",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/checkpoint-stage1")
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.checkpoint is None or args.checkpoint.suffix != ".safetensors":
        parser.error("Provide the original runtime .safetensors checkpoint")
    if min(args.height, args.width, args.frames) <= 0 or (
        args.height % 64 or args.width % 64
    ):
        parser.error("Stage1 input dimensions must be positive 64-multiples")
    if args.execute_vulkan and not args.compile:
        parser.error("--execute-vulkan requires --compile")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
