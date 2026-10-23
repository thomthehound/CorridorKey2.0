#!/usr/bin/env python3
"""Export and execute genuine CorridorKey layers through IREE Vulkan.

Uses CPU-only PyTorch exclusively for source-model import and numerical reference.
Each case instantiates a real module from corridorkey.blocks. No replacement
operators or CPU inference fallback are used for the compiled result.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path

# Direct execution from scripts/ci must resolve the repository's package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

import iree.turbine.aot as aot


def cases():
    from corridorkey.blocks import (
        FullAttention3D,
        FullSpatialAttention2D,
        LayerNorm2d,
        SwiGLU,
    )

    return {
        "rmsnorm2d": (
            lambda: LayerNorm2d(num_channels=16),
            (1, 16, 4, 4),
        ),
        "swiglu": (
            lambda: SwiGLU(dim=16, zero_init_down=False),
            (1, 4, 4, 16),
        ),
        "full-spatial-attention": (
            lambda: FullSpatialAttention2D(
                dim=16, num_heads=2, zero_init_proj=False
            ),
            (1, 1, 4, 4, 16),
        ),
        "full-temporal-attention": (
            lambda: FullAttention3D(dim=16, num_heads=2),
            (1, 2, 2, 2, 16),
        ),
    }


def prepare(name: str):
    make, shape = cases()[name]
    torch.manual_seed(1937)
    model = make().eval()
    # Upstream initializes several residual projections to zero. Change only
    # those initial weights for this random-weight structural test so an
    # incorrect attention implementation cannot appear to pass by producing 0.
    if name == "full-temporal-attention":
        with torch.no_grad():
            torch.nn.init.normal_(model.proj.weight, std=0.03)
            torch.nn.init.normal_(model.proj.bias, std=0.01)
    if name == "rmsnorm2d":
        with torch.no_grad():
            model.weight.copy_(torch.linspace(0.6, 1.4, 16))
    sample = torch.randn(shape, dtype=torch.float32)
    with torch.inference_mode():
        expected = model(sample).detach().numpy()
    if not np.isfinite(expected).all() or np.max(np.abs(expected)) == 0:
        raise AssertionError(f"{name}: invalid or trivial reference")
    return model, sample, expected


def _rms_norm_inference_math(
    input: torch.Tensor,
    normalized_shape,
    weight=None,
    eps=None,
):
    """Lower RMSNorm to reduction, reciprocal square root and multiplication.

    Keep accumulation in fp32 for reduced-precision inputs, and apply the
    original normalized dimensions, epsilon and learned affine weight.
    """
    dims = tuple(range(-len(normalized_shape), 0))
    epsilon = torch.finfo(input.dtype).eps if eps is None else eps
    accumulator = input.float() if input.dtype in (
        torch.float16, torch.bfloat16
    ) else input
    variance = accumulator.square().mean(dim=dims, keepdim=True)
    normalized = accumulator * torch.rsqrt(variance + epsilon)
    normalized = normalized.to(input.dtype)
    if weight is not None:
        normalized = normalized * weight.to(input.dtype)
    return normalized


def _sdpa_inference_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa: bool = False,
):
    """Canonical matmul/softmax decomposition for the unmasked inference SDPA.

    IREE's online_attention Vulkan lowering currently rejects a workgroup
    memref result. Importing mathematically equivalent primitive operators
    permits Vulkan compilation without changing any source model or weights.
    Other SDPA modes must be legalized independently rather than approximated.
    """
    if attn_mask is not None or dropout_p != 0.0 or is_causal or enable_gqa:
        raise NotImplementedError(
            "Only unmasked, non-causal, dropout-free inference SDPA is legalized"
        )
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * factor
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value)


def _read_device_result(config, result) -> np.ndarray:
    """Copy the actual Vulkan result through IREE-owned host staging memory.

    The high-level to_host() import path can allocate an unaligned NumPy
    destination when Vulkan external host allocation requires page alignment.
    The driver's own staging allocation satisfies device memory constraints.
    """
    import iree.runtime as ireert

    source = result._buffer_view.get_buffer()
    mem_type = int(ireert.MemoryType.HOST_LOCAL | ireert.MemoryType.DEVICE_VISIBLE)
    usage = int(ireert.BufferUsage.TRANSFER_TARGET | ireert.BufferUsage.MAPPING_SCOPED)
    staging = config.device.allocator.allocate_buffer(
        memory_type=mem_type,
        allowed_usage=usage,
        allocation_size=source.byte_length(),
    )
    semaphore = config.device.create_semaphore(0)
    config.device.queue_copy(
        source,
        staging,
        wait_semaphores=ireert.HalFence.create_at(semaphore, 0),
        signal_semaphores=ireert.HalFence.create_at(semaphore, 1),
    )
    ireert.HalFence.create_at(semaphore, 1).wait()
    mapping = staging.map()
    host = mapping.asarray(result._buffer_view.shape, result.dtype)
    return np.array(host, copy=True)


def execute_vulkan(binary: Path, sample: torch.Tensor) -> np.ndarray:
    import iree.runtime as ireert

    config = ireert.Config("vulkan")
    binary_bytes = binary.read_bytes()
    module = ireert.VmModule.copy_buffer(config.vm_instance, binary_bytes)
    compiled = ireert.load_vm_module(module, config)
    device_input = ireert.asdevicearray(config.device, sample.numpy())
    output = compiled.main(device_input)
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise AssertionError(f"expected one output, received {len(output)}")
        output = output[0]
    return _read_device_result(config, output)


def run_case(name: str, directory: Path, run_vulkan: bool, target: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    status = {"case": name, "export": False, "compile": False,
              "vulkan_execute": False, "vulkan_requested": run_vulkan}
    model, sample, expected = prepare(name)
    np.save(directory / "reference.npy", expected)
    exported = torch.export.export(model, (sample,), strict=True)
    exported = exported.run_decompositions({
        torch.ops.aten.scaled_dot_product_attention.default: _sdpa_inference_math,
        torch.ops.aten.rms_norm.default: _rms_norm_inference_math,
    })
    unlegalized = [
        str(node.target)
        for node in exported.graph.nodes
        if any(op in str(node.target) for op in (
            "scaled_dot_product_attention", "aten.rms_norm"
        ))
    ]
    if unlegalized:
        raise AssertionError(f"Unlegalized inference operators: {unlegalized}")
    output = aot.export(exported)
    output.verify()
    mlir_path = directory / "import.mlir"
    output.save_mlir(mlir_path)
    if not mlir_path.is_file() or mlir_path.stat().st_size < 100:
        raise AssertionError("MLIR export was empty")
    status["export"] = True
    print(f"[{name}] imported real CorridorKey module to MLIR", flush=True)

    vmfb_path = directory / "vulkan.vmfb"
    output.session.set_flags(f"--iree-vulkan-target={target}")
    output.compile(save_to=vmfb_path, target_backends=["vulkan-spirv"])
    if not vmfb_path.is_file() or vmfb_path.stat().st_size < 100:
        raise AssertionError("Vulkan binary was empty")
    status["compile"] = True
    print(f"[{name}] compiled MLIR to Vulkan VMFB", flush=True)

    if run_vulkan:
        actual = execute_vulkan(vmfb_path, sample)
        if actual.shape != expected.shape:
            raise AssertionError(
                f"shape mismatch: Vulkan {actual.shape}; PyTorch {expected.shape}"
            )
        if not np.isfinite(actual).all():
            raise AssertionError("non-finite Vulkan output")
        np.save(directory / "vulkan_output.npy", actual)
        np.testing.assert_allclose(
            actual, expected, atol=2.0e-3, rtol=2.0e-3,
            err_msg=f"Vulkan output differs from CorridorKey reference ({name})"
        )
        status["vulkan_execute"] = True
        status["max_absolute_error"] = float(np.max(np.abs(actual - expected)))
        print(f"[{name}] Vulkan execution matched source", flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-vulkan", action="store_true")
    parser.add_argument("--vulkan-target", required=True,
                        help="IREE Vulkan target (e.g. llvmpipe, rdna1)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "target": f"vulkan-spirv ({args.vulkan_target})",
        "vulkan_execution_requested": args.execute_vulkan,
        "scope": "upstream source-module graph proof; not full-model inference",
        "cases": [],
    }
    failure = False
    for name in cases():
        try:
            status = run_case(name, args.output / name,
                              args.execute_vulkan, args.vulkan_target)
        except Exception as e:
            status = {"case": name, "failed": True,
                      "error": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()}
            print(f"[{name}] FAILED\n{status['traceback']}", file=sys.stderr, flush=True)
            failure = True
        report["cases"].append(status)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    successes = sum(1 for case in report["cases"] if not case.get("failed"))
    print(f"Vulkan proof: {successes}/{len(report['cases'])} cases passed")
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
