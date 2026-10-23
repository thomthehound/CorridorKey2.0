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

## Integrated Stage 2 export and differential gate

`scripts/ci/prove_stage2_vulkan.py` instantiates the complete source `Stage2` class with an eight-block reduced-width configuration and executes its original RGB patchification, full-spatial/CSwin/Swin stack, learned late RGB skip, and four-channel depatchification. The deterministic probe activates zero-initialized residual and output weights so that passing results cannot be attributed to an effectively zero network.

Only model-independent geometry caches are specialized at import time; all learned modules and their weights remain source-authored. The test compares three values independently: original PyTorch Stage 2, the exported/decomposed graph run on CPU, and Vulkan SPIR-V execution through llvmpipe. RDNA-targeted compilation is checked separately on Windows.

The integrated graph **successfully exports, compiles and executes**, while its CPU-exported form agrees with the source reference. The first Vulkan graph differed at nine of 4096 output values (maximum absolute error approximately 0.00355). Flattening CSwin's stripe-head layout before concatenation preserved the original arithmetic and corrected the separate CSwin source-block regression; the integrated graph now has two values outside its fixed output tolerance (maximum absolute error approximately 0.00325). Static RoPE tables, a channel-last result ABI, and changing the IREE optimization level did not eliminate these two differences. The source-element numerical gate therefore still fails.

The importer additionally checks original patch/depatchify and three spatial transformer families in Vulkan against source results. `scripts/ci/diagnose_stage2_vulkan.py` instruments boundaries in the **existing exported graph** to compare intermediate tensors against software Vulkan and locate the first meaningful divergence. When the integrated gate fails on Linux, CI executes this diagnostic and preserves its tensor samples and report, without suppressing the failure or relaxing the acceptance tolerance.

These source-topology tests are a necessary intermediate milestone, not a claim of production checkpoint compatibility. A complete production proof still requires actual 1280-channel Stage 2 weights, the active foundation models, Stage 1's real recurrent and temporal semantics, and hardware AMD performance measurements.

## Grouped Swin-2D Vulkan correctness gate

The integrated reduced-width Stage 2 test first diverges in pre-block 1, a shifted Swin-2D block. Exported-graph probes show the preceding block's output matches source within approximately `1.2e-6`, but the tensor entering pre-block 2 differs by up to `0.026`. The final image discrepancy (approximately `0.00325` maximum absolute error) is a downstream effect. These observations are from a synthetic-weight 2×2-token Stage 2 profile, not the production checkpoint.

The prior stand-alone Swin-2D proof lowered cached regions to a dense boolean attention mask. That numerically passes for small windows but does **not** exercise CorridorKey's integrated grouped Q/K/V gather and result scatter. Two additional stand-alone regression profiles now exercise the source-derived grouped path, including the degenerate 2×2 token grid and a larger multi-batch, partially padded arrangement. The small case initially failed Vulkan output comparison for 100/192 values, while the larger case passed.

`scripts/ci/iree_vulkan_legalizations.py` adds a source-equivalent, **linear-working-memory** packed-gather lowering: execute the original grouped SDPA calls, pack only computed window/token results, and gather them by a geometry-derived source-order route table. No quadratic dense attention masks or GPU-side indexed scatter are required. This approach passed CPU differential checks, but the small Vulkan software target exposed an IREE SPIR-V legalization issue involving a `vector<24xf32>` constant in the headwise attention gate. A source-equivalent token-major headwise-gate lowering is now under CI evaluation alongside the packed gather. This work remains provisional until both the small grouped regression and the integrated Stage 2 numerical test pass on software Vulkan; neither the RDNA-target compile-only check nor CPU equivalence can establish Vulkan numerical correctness.

Future full-model validation must preserve sparse/grouped attention efficiency and test model weights, multiple spatial/temporal profiles and real AMD GPUs rather than accepting green synthetic source-block tests as proof of complete inference.
