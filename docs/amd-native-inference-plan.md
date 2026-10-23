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

## Continuous integration: Vulkan compiler proof

The `.github/workflows/vulkan-import-proof.yml` workflow tests original source modules in `corridorkey.blocks` with CPU-only PyTorch as a conversion/reference dependency. It imports the real RMS normalization, SwiGLU, full-spatial attention, and full-temporal attention implementations through IREE Turbine, compiles each to `vulkan-spirv`, and retains exported MLIR, VMFB and diagnostics.

Ubuntu 24.04 also runs the compiled binaries with Mesa lavapipe and compares their outputs with original PyTorch evaluation using nontrivial deterministic initialization. Windows builds the same Vulkan graphs without requiring Vulkan hardware. A failed graph, compiler, or numerical test fails the workflow. The test never interprets CPU execution of neural operators as Vulkan success.

These tests establish **operator/subgraph compiler feasibility only**. Randomly initialized instances and a software Vulkan device do not prove full checkpoint loading, complete Stage 1/Stage 2 export, active foundation model compatibility, AMD driver execution, final matte/foreground numerical equivalence, or production performance. Those remain required milestones and must receive their own concrete checkpoint-backed and GPU-backed CI gates rather than being inferred from this proof.
