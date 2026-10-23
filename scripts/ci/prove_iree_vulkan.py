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
from iree_vulkan_legalizations import grouped_sdpa_masked


def cases():
    from corridorkey.blocks import (
        FullAttention3D,
        FullSpatialAttention2D,
        FullSpatialBlock2D,
        CSwinBlock2D,
        PatchifyConv,
        Depatchify,
        LayerNorm2d,
        RMSNorm,
        SwiGLU,
        SwinBlock2D,
        SwinBlock3D,
    )

    return {
        "rmsnorm2d": (
            lambda: LayerNorm2d(num_channels=16),
            (1, 16, 4, 4),
        ),
        "rmsnorm-nhwc": (
            lambda: RMSNorm(16),
            (1, 4, 4, 16),
        ),
        # Stage1's exact normalization operand ranks/dimensions; these
        # distinguish a fundamental RMSNorm error from fused graph codegen.
        "rmsnorm-stage1-tokens": (
            lambda: RMSNorm(64),
            (1, 2, 4, 4, 64),
        ),
        "rmsnorm-stage1-head": (
            lambda: RMSNorm(16),
            (2, 4, 16, 16),
        ),
        "rmsnorm2d-small": (
            lambda: LayerNorm2d(num_channels=16),
            (1, 16, 2, 2),
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
        # Real SwinBlock2D wrappers (not a substitute attention network).
        # Irregular sizes exercise masking of shifted, padded edge tokens.
        "swin2d-shift-padded": (
            lambda: SwinBlock2D(dim=16, num_heads=2, window_size=4,
                                shift=(2, 2)),
            (1, 7, 9, 16),
        ),
        "swin2d-shift-batched": (
            lambda: SwinBlock2D(dim=16, num_heads=2, window_size=4,
                                shift=(2, 0)),
            (2, 8, 8, 16),
        ),
        "swin2d-shift-column": (
            lambda: SwinBlock2D(dim=16, num_heads=2, window_size=4,
                                shift=(0, 2)),
            (1, 5, 7, 16),
        ),
        # First Stage-1-style true temporal shifted-window graph. Unlike
        # Swin2D, this exercises upstream grouped index gather/scatter.
        "swin3d-shift-padded": (
            lambda: SwinBlock3D(dim=16, num_heads=2, window_H=4,
                                window_W=4, shift=(2, 2)),
            (1, 2, 5, 7, 16),
        ),
        "swin3d-shift-batched": (
            lambda: SwinBlock3D(dim=16, num_heads=2, window_H=4,
                                window_W=4, shift=(2, 0)),
            (2, 3, 5, 7, 16),
        ),
        # Stage2's actual three spatial transformer families at the same
        # channel count and token grid as the complete Stage2 probe.
        "fullspatial2d-stage2": (
            lambda: FullSpatialBlock2D(dim=48, num_heads=4),
            (1, 2, 2, 48),
        ),
        "cswin2d-stage2": (
            lambda: CSwinBlock2D(dim=48, num_heads=4),
            (1, 2, 2, 48),
        ),
        "swin2d-shift-stage2": (
            lambda: SwinBlock2D(dim=48, num_heads=4, window_size=4,
                                shift=(2, 2)),
            (1, 2, 2, 48),
        ),
        # Unlike the dense-mask Swin proof, these run the original grouped
        # gather/scatter path with an exact contiguous index-copy lowering.
        "swin2d-grouped-stage2": (
            lambda: SwinBlock2D(dim=48, num_heads=4, window_size=4,
                                shift=(2, 2)),
            (1, 2, 2, 48),
        ),
        "swin2d-grouped-batched": (
            lambda: SwinBlock2D(dim=16, num_heads=2, window_size=4,
                                shift=(2, 0)),
            (2, 5, 7, 16),
        ),
        # Original Stage2 image-to-token and token-to-image operators.
        # Test nontrivial weights, not the zero-initialized image head.
        "stage2-patchify": (
            lambda: PatchifyConv(in_channels=3, out_channels=32, patch=16),
            (1, 3, 32, 32),
        ),
        "stage2-depatchify": (
            lambda: Depatchify(in_dim=48, out_channels=4, r=16,
                               zero_init_all=False),
            (1, 48, 2, 2),
        ),
        "stage2-patchify-overlap": (
            lambda: PatchifyConv(in_channels=3, out_channels=32,
                                 patch=16, overlap_mult=2),
            (1, 3, 32, 32),
        ),
        "stage2-depatchify-overlap": (
            lambda: Depatchify(in_dim=48, out_channels=4, r=16,
                               zero_init_all=False, overlap_mult=2),
            (1, 48, 2, 2),
        ),
    }


class GeometryMaskSwin2DAttention(torch.nn.Module):
    """Import-time legalization of an original shifted Swin attention module.

    The source block retains its exact forward, weights, shifts and skips.
    Its own cached grouping represents a shape-constant attention mask;
    express it as a dense bool tensor for graph import instead of exporting
    Python loops, CPU region IDs, and advanced indexed scatter.
    """

    def __init__(self, original, shape, window_size, shift):
        super().__init__()
        from corridorkey.blocks import _shifted_attention_mask_2d

        self.original = original
        _, height, width, _ = shape
        padded_height = ((height + window_size - 1) // window_size) * window_size
        padded_width = ((width + window_size - 1) // window_size) * window_size
        mask = _shifted_attention_mask_2d(
            height, width, padded_height, padded_width,
            window_size, shift[0], shift[1], torch.device("cpu"),
        )
        if mask is None:
            raise ValueError("Shifted block requires a nonempty region mask")
        if not mask.any(dim=-1).all():
            raise ValueError("Shifted mask has a fully masked query")
        self.register_buffer("allowed", mask, persistent=False)

    def forward(self, x, attn_mask=None, region_groups=None, region_ids=None):
        if region_groups is None:
            return self.original(x, attn_mask=attn_mask, region_ids=region_ids)
        return self.original(x, attn_mask=self.allowed)


class ChannelLastResult(torch.nn.Module):
    """Expose the source operator's NCHW result in NHWC GPU tensor layout.

    The original numerical result is reconstructed at the API boundary. This
    avoids materializing a transposed NCHW output inside the Vulkan graph.
    """

    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, x):
        return self.source(x).permute(0, 2, 3, 1).contiguous()


def verify_cswin_layout_equivalence():
    """Prove head-wise gating commutes with stripe flatten/concatenation."""
    generator = torch.Generator().manual_seed(2731)
    a = torch.randn(2, 3, 4, 2, 12, generator=generator)
    b = torch.randn(2, 3, 4, 2, 12, generator=generator)
    gate = torch.sigmoid(torch.randn(2, 3, 4, 4, 1, generator=generator))
    original = (torch.cat([a, b], dim=-2) * gate).reshape(2, 3, 4, 48)
    gate_a, gate_b = gate.split((2, 2), dim=-2)
    flattened = torch.cat(
        [(a * gate_a).reshape(2, 3, 4, 24),
         (b * gate_b).reshape(2, 3, 4, 24)],
        dim=-1,
    )
    torch.testing.assert_close(original, flattened, atol=0.0, rtol=0.0)


def prepare(name: str):
    make, shape = cases()[name]
    torch.manual_seed(1937)
    if name == "cswin2d-stage2":
        verify_cswin_layout_equivalence()
    model = make().eval()
    # Upstream initializes several residual projections to zero. Change only
    # those initial weights for this random-weight structural test so an
    # incorrect attention implementation cannot appear to pass by producing 0.
    if name == "full-temporal-attention":
        with torch.no_grad():
            torch.nn.init.normal_(model.proj.weight, std=0.03)
            torch.nn.init.normal_(model.proj.bias, std=0.01)
    if name.startswith("rmsnorm"):
        with torch.no_grad():
            model.weight.copy_(torch.linspace(0.6, 1.4, 16))
    if (name.startswith(("swin2d-shift", "swin2d-grouped", "swin3d-shift"))
            or name in ("fullspatial2d-stage2", "cswin2d-stage2")):
        with torch.no_grad():
            torch.nn.init.normal_(model.attn.proj.weight, std=0.04)
            torch.nn.init.normal_(model.attn.proj.bias, std=0.02)
            torch.nn.init.normal_(model.mlp.w_down.weight, std=0.04)
            torch.nn.init.normal_(model.mlp.w_down.bias, std=0.02)
    sample = torch.randn(shape, dtype=torch.float32)
    with torch.inference_mode():
        expected = model(sample).detach().numpy()
    if not np.isfinite(expected).all() or np.max(np.abs(expected)) == 0:
        raise AssertionError(f"{name}: invalid or trivial reference")
    if name.startswith("swin2d-shift"):
        source_attention = model.attn
        model.attn = GeometryMaskSwin2DAttention(
            source_attention, shape, model.window_size,
            (model.shift_h, model.shift_w),
        )
        with torch.inference_mode():
            legalized_reference = model(sample).detach().numpy()
        np.testing.assert_allclose(
            legalized_reference, expected, atol=1.0e-5, rtol=1.0e-5,
            err_msg="Dense geometry mask must reproduce original grouped SDPA",
        )
    if name == "rmsnorm2d":
        lowered = ChannelLastResult(model).eval()
        with torch.inference_mode():
            lowered_reference = lowered(sample).permute(0, 3, 1, 2).numpy()
        np.testing.assert_allclose(
            lowered_reference, expected, atol=1.0e-6, rtol=1.0e-6,
            err_msg="Channel-last graph output changed source semantics",
        )
        model = lowered
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
    if dropout_p != 0.0 or is_causal or enable_gqa:
        raise NotImplementedError(
            "Inference SDPA legalization supports optional masks, "
            "but not dropout, causal attention or grouped-query expansion"
        )
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * factor
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # PyTorch SDPA bool masks mean True=allowed. Every mask row
            # must have at least one valid key; checked during preparation.
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
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
    tensor_bytes = int(np.prod(result.shape)) * np.dtype(result.dtype).itemsize
    source_bytes = source.byte_length()
    print(f"[readback] logical_bytes={tensor_bytes}, physical_bytes={source_bytes}, "
          f"shape={result.shape}, dtype={result.dtype}", flush=True)
    if source_bytes < tensor_bytes:
        raise AssertionError(
            f"Result device buffer is smaller than declared tensor: "
            f"{source_bytes} < {tensor_bytes} bytes"
        )
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
              "vulkan_execute": False, "vulkan_requested": run_vulkan,
              "legalization": ("NHWC GPU output, NCHW API semantics" if name == "rmsnorm2d"
                                else "standard import legalization")}
    model, sample, expected = prepare(name)
    np.save(directory / "reference.npy", expected)
    if name.startswith(("swin2d-shift", "swin2d-grouped", "swin3d-shift")):
        # Source Python region caches are shape-constant but use nonzero +
        # tolist, which cannot be traced. Capture their actual output on
        # CPU, then substitute only the cache during torch.export.
        # The original grouped attention remains intact for 3D.
        import corridorkey.blocks as blocks

        is_3d = name.startswith("swin3d")
        cache_name = ("_shifted_region_groups_3d_cached" if is_3d
                      else "_shifted_region_groups_2d_cached")
        source_groups = getattr(blocks, cache_name)
        if is_3d:
            _, frames, height, width, _ = sample.shape
            size_h, size_w = model.window_H, model.window_W
        else:
            _, height, width, _ = sample.shape
            size_h = size_w = model.window_size
        height_pad = ((height + size_h - 1) // size_h) * size_h
        width_pad = ((width + size_w - 1) // size_w) * size_w
        signature = ((frames, height, width, height_pad, width_pad,
                      size_h, size_w, model.shift_h, model.shift_w)
                     if is_3d else
                     (height, width, height_pad, width_pad,
                      size_h, model.shift_h, model.shift_w))
        groups = source_groups(*signature)

        def fixed_groups(*args):
            if args != signature:
                raise ValueError("Unexpected shifted-window export shape")
            return groups

        grouped_case = name.startswith("swin2d-grouped")
        original_sdpa = blocks._sdpa_with_region_groups
        setattr(blocks, cache_name, fixed_groups)
        if grouped_case:
            blocks._sdpa_with_region_groups = grouped_sdpa_masked
        try:
            if grouped_case:
                with torch.inference_mode():
                    lowered_reference = model(sample).detach().numpy()
                np.testing.assert_allclose(
                    lowered_reference, expected, atol=1e-6, rtol=1e-6,
                    err_msg="Region mask legalization differs from source",
                )
                status["legalization"] = "exact static region mask"
            exported = torch.export.export(model, (sample,), strict=True)
        finally:
            setattr(blocks, cache_name, source_groups)
            blocks._sdpa_with_region_groups = original_sdpa
    else:
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
    output.session.set_flags("--iree-opt-level=O2")
    output.compile(save_to=vmfb_path, target_backends=["vulkan-spirv"])
    if not vmfb_path.is_file() or vmfb_path.stat().st_size < 100:
        raise AssertionError("Vulkan binary was empty")
    status["compile"] = True
    print(f"[{name}] compiled MLIR to Vulkan VMFB", flush=True)

    if run_vulkan:
        actual = execute_vulkan(vmfb_path, sample)
        if name == "rmsnorm2d":
            if actual.shape != (1, expected.shape[2], expected.shape[3], expected.shape[1]):
                raise AssertionError(f"unexpected channel-last GPU output {actual.shape}")
            actual = actual.transpose(0, 3, 1, 2).copy()
        if actual.shape != expected.shape:
            raise AssertionError(
                f"shape mismatch: Vulkan {actual.shape}; PyTorch {expected.shape}"
            )
        if not np.isfinite(actual).all():
            raise AssertionError("non-finite Vulkan output")
        np.save(directory / "vulkan_output.npy", actual)
        close = np.isclose(actual, expected, atol=2.0e-3, rtol=2.0e-3)
        if not close.all():
            failed_positions = np.argwhere(~close)
            print(f"[{name}] {failed_positions.shape[0]} divergent values; "
                  f"first 12 positions (actual, expected):", flush=True)
            for position in failed_positions[:12]:
                index = tuple(map(int, position))
                print(f"  {index}: {actual[index]!r}, {expected[index]!r}",
                      flush=True)
            print(f"[{name}] output physical buffer bytes="
                  f"{actual.nbytes}, shape={actual.shape}", flush=True)
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
