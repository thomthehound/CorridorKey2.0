# AMD-native inference architecture

## Objective

Execute CorridorKey's original learned models and checkpoints natively on AMD GPUs under Vulkan, preserving full-resolution foreground/alpha output, HDR color and temporal semantics, and without requiring CUDA, ROCm/HIP, ONNX, or PyTorch in the deployed runtime. The import/compiler architecture must accept subsequent CorridorKey model releases without manual reimplementation of their networks.

The native core uses C++23 on Windows and Linux. Vulkan 1.4 is the initial GPU baseline; extensions are selected by queried device capabilities. The interpreter, import, compiler, scheduling, and execution layers remain independent of any compositor or animation host.

## Architectural boundaries

```text
Model release: source architecture + checkpoint weights + metadata
                             |
                CPU-side acquisition/export
                  (isolated from runtime)
                             |
       PyTorch exporter / torch-mlir / IREE Turbine importer
             (for source models written in PyTorch)
                             |
               Canonical MLIR representation
         (operators, parameters, shapes, states, metadata)
                             |
        Graph verification, legalization and optimization
                             |
              Target-specific compilation
                  /                     \
       Vulkan SPIR-V / IREE          Future AIE/NPU backend
                  \                     /
          Backend-independent execution interface
                             |
        Native C++23 scheduling / buffer ownership / IO
                             |
                 Alpha + linear foreground
```

**No inference-time PyTorch dependency.** Models authored in PyTorch require a source-compatible import tool at **conversion/build time**, normally using CPU-only PyTorch export with torch-mlir/IREE Turbine. Do not use ONNX as a required or intermediate artifact. Do not build an alternate hand-coded imitation of the model. A genuinely PyTorch-free source-to-IR importer may be researched separately; parsing checkpoint weights alone cannot recover arbitrary Python model semantics.

**Compilation contract.** Retain source/checkpoint version and model configuration as semantic inputs to compilation. The compilation pipeline detects unsupported operations or control flow, produces an actionable coverage report, and fails rather than silently replacing operators, weakening temporal context, or falling back to CPU. A new model release may require a new operator legalization or exporter adaptation, but must not require a new hand-written GPU implementation of its entire network.

**Separation of model and application behavior.** Compile well-defined model subgraphs; manage video reading, temporal window scheduling, hint carry, RVM chronological state, Hann blending, and Stage 2 frame processing in explicit native orchestration. Do not freeze release-specific Python scheduler implementation details into unrelated graph kernels. The scheduler must preserve source semantics for each model version.

## Source-of-truth semantics

- Read checkpoint model configuration; model defaults are not authoritative. The source tree includes optional C-RADIOv4-SO400M, MoGe-2 ViT-B, and RVM small foundation models. Determine active components from the actual checkpoint.
- Preserve signed-asinh scene-linear RGB encoding, source color transforms, resolution/padding, per-foundation input resampling, feature normalization, and final inverse transforms.
- Preserve spatiotemporal Stage 1 attention, RoPE, shifted-window masking and routing, CSwin operations, recurrent state, per-window hints, carried hints, and overlapping Hann accumulation.
- Preserve Stage 2 refined alpha and foreground, including unclamped linear/HDR foreground and correct straight/premultiplied output semantics.
- Preserve native frame timestamps and expose a structured in-memory output API without forcing an encoded preview or disk round-trip.
- Treat low-VRAM window length and precision as explicit operating modes because both can affect quality or performance.

## Milestone 0 — Model and importer proof

1. Audit the checkpoint metadata, active networks, input/output tensor interfaces, operator coverage, precision, and dynamic control flow; enumerate graph partitions and shape profiles.
2. Use an isolated CPU-capable importer built on `torch.export` / `torch-mlir` / `iree-turbine` to capture the actual source models as MLIR. PyTorch is import-time tooling only.
3. Diagnose dynamic masking and data-dependent indexing in `corridorkey/blocks.py`; specialize geometry-dependent region maps and masks where semantics permit. Support explicit shape variants rather than inventing a different attention rule.
4. Produce deterministic reference fixtures for the original operations and subgraphs. Include no hint, first-frame hint, full hint, carried hint, overlapping windows, occlusion/motion edges, HDR, and multiple precision profiles.
5. Compile the active foundation graph(s), Stage 1 and Stage 2 using IREE's `vulkan-spirv` backend. Verify the compiled binary executes each partition on an AMD GPU without CUDA/ROCm/ONNX or an implicit CPU kernel.
6. Report unsupported ops and their exact legalization path. Extend the importer/compiler only where necessary, then rerun export. Never substitute a different foundation network.

**Gate:** Successful model capture and Vulkan execution of all required graph partitions with layerwise comparisons against the original reference.

## Milestone 1 — Complete source-faithful Vulkan inference

- Load exact weights and model metadata, compile required shape/window variants, and run a complete clip through foundation inference, Stage 1 and Stage 2.
- Implement common runtime interfaces for stateful inputs, outputs, tensor ownership, feature caches, and versioned scheduler configuration.
- Preserve hint/carry/recurrent semantics, overlapping feature accumulation, padding, output cropping, color transforms, and HDR precision.
- Check alpha and foreground reference agreement per layer and in the final image; investigate visually meaningful or temporally systematic errors instead of relying only on loose mean error thresholds.
- Record the actual GPU execution graph and all host transfers. Host scheduling is expected; hidden CPU neural-operator execution is not an acceptable native-GPU result.

**Gate:** End-to-end inference with actual checkpoints on AMD under Vulkan, with validated original-model behavior and explicit performance/memory accounting.

## Milestone 2 — Native host, persistent memory and extensibility

- Build the native C++23 inference runtime with Vulkan integration, predictable resource ownership, device capabilities, error reporting, and input/output interfaces.
- Preallocate/reuse graph inputs, outputs and scratch buffers; pool per-window storage; minimize host/device round trips.
- Implement asynchronous transfer and execution scheduling where correct and beneficial. Retain bounded-VRAM profiles without silently compromising model semantics.
- Cache compiled variants by model config, shape profile, precision strategy, target feature set and relevant compiler settings. Invalidate incompatible artifacts based on metadata/schema compatibility.
- Provide Windows/Linux builds, CPU-side validation, and hardware AMD test targets.
- Keep IREE as the initial compiler/runtime implementation, not as a public model API that prevents later compiler substitutions. Any custom GPU kernels integrate through compiler lowering or explicit verified runtime operators, rather than forming a shadow network.

**Gate:** The distributed inference path requires neither PyTorch nor ONNX/ROCm/CUDA, and can be driven by native video, image, or memory clients.

## Milestone 3 — Measured GPU performance

Profile before tuning:

- Flash/fused SDPA-style attention, shifted windows, CSwin, RoPE and recurrent execution.
- Linear/conv/MLP fusion, tensor layout, subgroup operations, and optional cooperative matrices based on queried Vulkan capabilities.
- BF16/FP16/FP32 numerical strategies. FP16 is not automatically equivalent to BF16, especially for HDR; quantify error at sensitive points.
- Unnecessary sync, per-frame allocation, CPU staging and duplicate computation across Hann overlaps, without changing learned temporal behavior.
- Decoder/inference/output overlap, throughput versus latency, compilation/startup cost and memory budgets.
- Benchmark identical model/window/quality settings at 720p, 1080p, 1440p and 4K, reporting mean/p95 latency, FPS, memory, transfer bytes and quality metrics.

**Gate:** Measured performance gains with preserved alpha, foreground, edge, identity, and temporal quality. Never claim a speedup from reduced resolution or shorter temporal context without naming the changed setting.

## Future NPU + GPU execution

Vulkan/SPIR-V is the **GPU** target, not a portable NPU driver. The shared MLIR graph is the NPU expansion point. Prototype AMD XDNA compilation through the appropriate MLIR-AIE/IRON/AIE backend and its hardware runtime (for example XRT or supported native alternatives), independently from GPU Vulkan.

- Partition at graph boundaries according to measured execution capabilities, precision, scratchpad and model storage, transfer costs, and dependency scheduling.
- Account for GPU ↔ NPU data movement and synchronization; fused attention or intermediate feature matrices are not assumed NPU-capable.
- Prefer NPU work only when end-to-end quality and latency/throughput exceed the optimized GPU-only path.
- NPU execution must preserve the exact model's numerical/semantic contract or be exposed as an explicitly different precision/performance profile.
- Keep an optional D3D12 integration route for Windows interoperability, but do not require a full second neural compiler frontend or separate manually maintained network.

## Validation and maintenance

- Model compatibility CI ingests each upstream CorridorKey checkpoint/config and emits: imported graph inventory, unsupported ops, compiled graph coverage, precision and device requirements, output-diff report, and full end-to-end smoke results where hardware is available.
- Source model release updates pass through import → legalization → compilation → differential tests → benchmark. Supported operators and schedulers remain reusable.
- Test existing graph fixtures as regression tests when changing compiler, IREE, MLIR, Vulkan features or AMD drivers.
- Build native components on `windows-latest` and current Ubuntu LTS. GPU correctness and performance must be validated on real AMD hardware.
- Initial architectural portability target: Vulkan 1.4-capable AMD RDNA GPUs, subject to actual extension/precision support. Radeon 680M-class integrated GPUs are a performance floor to evaluate, not a promise of identical fidelity and throughput at every capture resolution.
- Maintain CorridorKey attribution and the source license, including derivative-release and commercial-integration provisions, along with all third-party notices.
- Keep the inference engine independent of any compositor, pose tracker, 3D character editor, or video-capture implementation.

## Immediate development sequence

1. Implement a no-GPU metadata inspection utility for the runtime checkpoint to report active graphs, shapes, precision and external assets.
2. Build the CPU-side PyTorch-to-MLIR export harness for **real** foundation/Stage 1/Stage 2 graphs and generate an unsupported-op report.
3. Preserve intermediate reference tensors at graph boundaries and small subgraphs, including temporal and HDR cases.
4. Execute complete imported graph partitions on Vulkan, systematically legalizing unsupported operations in reusable compiler passes.
5. Integrate the original temporal scheduler semantics and prove full-clip native GPU correctness before optimization.

## Continuous integration: source-faithful Vulkan importer proof

The `.github/workflows/vulkan-import-proof.yml` workflow imports actual source modules from `corridorkey.blocks` using isolated CPU-side PyTorch, compiles the graphs to `vulkan-spirv` with IREE, and preserves MLIR, compiled VMFB and structured reference-output artifacts. Windows checks RDNA1-targeted compilation; Ubuntu 24.04 executes through Mesa llvmpipe and numerically compares device outputs to the unchanged source reference. This validates the compilation and GPU execution *semantics* for exercised graphs; it does not constitute an AMD hardware performance test.

The proof currently exercises eighteen configurations:
- RMSNorm with channel-first, channel-last and small NCHW output shapes
- SwiGLU
- Full-spatial and full-spatiotemporal attention
- Three shifted Swin-2D variants (diagonal, row, column), including padded boundaries and multi-batch routing
- Two shifted Swin-3D variants, including padded, multi-frame and multi-batch region routing
- Full-spatial, CSwin, and shifted-Swin blocks at the reduced Stage 2 token layout
- Native patchification and depatchification, each with non-overlapping and overlapping convolution kernels

The source-derived 2D shifted-region cache uses CPU indexing and grouped scatter unsuitable for static graph capture. The importer specializes shape-dependent cached groups, then exposes them as a constant boolean mask on the existing Swin attention module. It numerically verifies the legalized module before exporting it. This is a proof of correctness for modest attention windows; dense quadratic masks are **not** the production strategy for long temporal windows. The 3D proof instead retains the original grouped gather/scatter implementation and specializes only the geometry cache; both compilation and software-Vulkan numerical execution succeed on the tested shapes.

Other observed importer requirements:
- IREE's fused Vulkan `online_attention` lowering failed for the source SDPA paths. The current importer decomposes supported inference SDPA calls into scaled matrix multiplication, softmax and matrix multiplication. Optional boolean/additive masks are supported; dropout, causal paths, GQA, and untested attention patterns must receive explicit coverage before full model integration.
- The original `LayerNorm2d` computes normalization in NHWC then produces an NCHW tensor. The larger original output graph miscomputed values under llvmpipe. The importer can retain the original normalization with an NHWC output boundary and reconstruct documented NCHW ordering at the API boundary without relaxing comparison tolerances.
- The software Vulkan device requires aligned readback storage. The proof uses runtime-owned staging memory and validates source-buffer size before copying.

`scripts/ci/inspect_checkpoint.py` independently reads runtime `.safetensors` headers without PyTorch or a GPU, inventories model configuration, tensor descriptors, and explicit foundation flags, and checks that every checkpoint configuration field is understood by the current `CorridorKeyV3` constructor. Its dependency-free synthetic-fixture self-test runs in both CI jobs. When a real checkpoint is available, the same utility emits a versioned architecture manifest using `python scripts/ci/inspect_checkpoint.py weights/corridorkey_v2.safetensors --output build/checkpoint.json`. It records source runtime overrides separately because RVM activation and recurrent-site selection cannot be inferred from checkpoint flags alone.

**Next mandatory integration gate:** import checkpoint-backed Stage 1 and Stage 2 with their active foundations and versioned scheduling contracts, then execute short real footage with alpha, foreground, hint-carry and HDR differential tests. This must also cover large masked temporal attention without dense quadratic materialization and test actual AMD driver execution. No synthetic-weight module proof should be represented as complete model or production performance compatibility.

## Integrated Stage 2 Vulkan correctness

`scripts/ci/prove_stage2_vulkan.py` exports the complete source `Stage2` class in a deterministic reduced-width eight-block configuration. The proof retains original patchification, all pre/post transformer blocks (full-spatial, CSwin and shifted Swin), learned late pixel skip, and depatchification. The otherwise-zero residual projections are initialized to nontrivial deterministic values to ensure attention errors cannot be hidden by zero outputs.

The source output, the decomposed exported CPU graph, and the executed software-Vulkan graph are compared independently. Windows verifies RDNA1-targeted SPIR-V compilation. The existing tolerances remain unchanged. The full Stage 2 graph now exports, compiles, executes, and **passes its numerical comparison** on Ubuntu software Vulkan. The component suite passes 20/20 on Windows (compilation) and Linux (execution).

The path to that result revealed two IREE code generation constraints:

- CSwin's five-dimensional head gating/concatenation produced incorrect software-Vulkan values. Source-exact gate application before flattening into NHWC was separately verified and repairs the CSwin regression.
- The reference shifted Swin-2D region assembly uses grouped indexed writes. The original full graph diverged first after pre-block 1 (up to 0.026 in the reduced test), eventually leaving a small final-image mismatch. Attempts to replace the scatter with packed indexed gathers either produced incorrect Vulkan results or failed compilation. The passing import uses the source's **static region membership as a registered boolean mask buffer**, leaving Q/K/V, projections, head gating, and all learned weights unchanged. Padding-only rows have defined zero outputs, and CPU differential checks enforce equivalence.

The mask-based lowering is a **correctness import path**, not the final performance path. For a window containing N tokens, its mask requires O(N²) entries. Long-window/temporal production execution needs a source-equivalent fused or explicitly grouped Vulkan kernel with bounded working storage and no per-token scalar gathers. The original grouped Swin-3D path remains separately exercised by the component proof; do not extrapolate the 2D masked solution to Stage 1's temporal graphs.

When an integrated comparison fails, `scripts/ci/diagnose_stage2_vulkan.py` instruments the exported computation at true inter-block inputs, attention and output boundaries, and saves reference/Vulkan tensors. This differential path remains available to catch new shape- or backend-dependent failures.

## Trained checkpoint Stage 2 gate

`scripts/ci/prove_checkpoint_stage2.py` reads the runtime safetensors header, resolves the original `CorridorKeyV3` constructor defaults directly from source, instantiates exactly its `Stage2` configuration, and loads the checkpoint's `stage2.*` tensors with strict key/shape checks. It does **not** construct or download the frozen foundation networks. Its synthetic checkpoint round-trip and missing-key rejection are exercised on both CI platforms.

With a real runtime checkpoint available, validate trained weights and source CPU inference:

```bash
python scripts/ci/prove_checkpoint_stage2.py weights/corridorkey_v2.safetensors --height 64 --width 64
```

To additionally export the trained graph and compile it for the RDNA1 Vulkan target:

```bash
python scripts/ci/prove_checkpoint_stage2.py weights/corridorkey_v2.safetensors --height 64 --width 64 --compile --target rdna1
```

Use `--execute-vulkan` on a configured Vulkan device to compare native device output to trained-weight source inference. The tool supports the optional two-output Stage 2 refinement contract and preserves both outputs in reports. CPU PyTorch, safetensors and IREE are **conversion/verification dependencies**, not part of the intended deployed Vulkan runtime.

The real checkpoint has **not yet been run through this gate**. Trained-weight compilation, numerics across multiple shapes, AMD GPU execution and throughput remain to be verified. Completion of Stage 2 does not imply complete CorridorKey inference: Stage 1 recurrent/temporal behavior, active C-RADIO/MoGe/RVM paths, hint carry, foreground/alpha reconstruction, HDR and scheduling behavior require independent end-to-end validation.

