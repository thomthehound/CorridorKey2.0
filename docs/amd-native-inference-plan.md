# AMD-native inference plan

## Objective

Run the existing CorridorKey 2.0 inference model and its original weights natively on AMD GPUs on Windows and Linux, without CUDA or HIP/ROCm as required runtime dependencies. Preserve the model's foreground reconstruction, alpha quality, HDR color semantics, and temporal behavior. Build a reusable inference component that can subsequently serve compositors, batch tools, and performance-capture pipelines.

Initial baseline: Vulkan 1.4 with GPU feature-based capability selection; a Windows D3D12 execution path is a complementary target. C++23 for the native host. Do not assume that a backend advertised as supported can compile and execute this specific model correctly.

## Source-of-truth contracts

- Load `weights/corridorkey_v2.safetensors` model configuration and weights without changing the learned architecture. Foundation model enablement must follow checkpoint metadata, not class defaults.
- The foundation models are C-RADIOv4-SO400M, MoGe-2 ViT-B, and RVM small, with their distinct licenses. Model configuration determines which components contribute features.
- Preserve input preprocessing: display-coded versus scene-linear sources, signed-asinh encoding, foundation RGB proxies, padding, channel conventions, and output inverse transforms.
- Preserve Stage 1 temporal attention, overlapping Hann windows, hint carry, continuous RVM recurrent state, and Stage 2 output refinement. Stage 2 processes frames independently after the temporal baton is assembled.
- Preserve native-resolution linear foreground and alpha outputs; expose structured frames and timestamps without requiring an encoded preview-video round trip.
- Existing `corridorkey/*.py` is the reference implementation. The native runtime must not silently substitute approximated model behavior or switch to CPU for unsupported operators.

## Milestone 0: executable reference and export feasibility

1. Inventory the exact checkpoint config, active model components, model input/output signatures, precision, tensor shapes, attention patterns, and dynamic control flow.
2. Construct deterministic short reference clips and per-subgraph tensor fixtures for: no hint, first-frame hint, full hint, carried hints, moving edges, blur, occlusion, and HDR/linear inputs.
3. Record Stage 1, Stage 2, foundation, and final alpha/foreground reference outputs in representative window modes (`1/1`, `8/4`, `24/12`). Reference generation may use a CUDA machine; minimal subgraphs can use a CPU reference where feasible.
4. Evaluate PyTorch export and ONNX/IREE import **separately** for RVM, MoGe, checkpoint-enabled C-RADIO, Stage 1, and Stage 2. Identify unsupported operations, data-dependent Python control flow, shape constraints, and required decompositions. Attention masks/region routing and recurrent state deserve special attention.
5. Probe IREE Vulkan execution and Windows ML / ONNX Runtime execution-provider coverage on a real AMD device. Require an execution-provider partition report; CPU fallback must be visible and fail the native GPU acceptance test.

**Gate:** A documented and reproducible conversion route for every active graph; otherwise solve the specific exporter/operator gap before building a second implementation of the model.

## Milestone 1: correct end-to-end AMD execution

- Compile and execute the complete set of required graph partitions on AMD without CUDA. IREE/Vulkan is the preferred portable candidate; Windows ML/DirectML/MIGraphX can supply an independent compatibility and performance reference.
- Keep model configuration, checkpoint loading, hint semantics, and temporal state orchestration common across execution backends.
- Run a short clip from source pixels to native alpha and foreground using the real checkpoints. Preserve intermediate tensors for layerwise comparisons.
- Record exact precision and execution-provider placements. Do not describe a mixed CPU/GPU run as native GPU inference.
- Support Windows and Linux. A backend may remain platform-specific, but a complete validated Vulkan path should not depend on a proprietary driver stack beyond ordinary Vulkan GPU drivers.

**Gate:** The complete clip passes reference comparisons, with discrepancies understood rather than hidden by loose aggregate tolerances.

## Milestone 2: native runtime and memory architecture

- Implement a C++23 runtime with explicit video/image inputs, frame timestamps, model/profile selection, recurrent-state ownership, and native alpha/foreground outputs. Retain Python as a reference/export tool, not a required inference host.
- Preallocate and reuse persistent tensor buffers and compiled graph variants across shots and frames. Cache by graph configuration, device capabilities, and resolution/window profile as needed.
- Treat foundation extraction, Stage 1 Hann accumulation, hint carry, Stage 2, and image output as one scheduled pipeline, not a set of full-frame host/device transfers.
- Support both GPU-resident streaming and bounded-VRAM execution with an explicit, measured offload budget. Preserve the exact low-VRAM semantics, including chronological RVM inference and overlapping-window blending.
- Establish native Vulkan resource/synchronization ownership; provide D3D12 interop where supported without imposing a D3D12-only core.

**Gate:** Frame processing no longer requires the original Python/CUDA path, intermediate state remains consistent, and measured transfer/allocation overhead is reported.

## Milestone 3: performance engineering

Measure each component before implementing targeted optimizations:

- Attention: fused/scaled-dot-product attention, RoPE, shifted-window masks, CSwin partitions, recurrent execution, and temporal window scheduling.
- Convolution/linear/MLP: compiled operator fusion, tensor layout, SIMD subgroup operations, cooperative matrix support where device capabilities allow.
- Precision: reference BF16 behavior; test FP16, mixed precision, and FP32 at sensitive boundaries against reference. Do not assume FP16 is numerically interchangeable with BF16, especially for HDR.
- Minimize intermediate conversion, redundant inference over overlapping windows, unnecessary device synchronization, CPU feature-map staging, and repeated compilation.
- Use asynchronous execution and overlap decoding, preprocessing, inference, and output when profiling shows a benefit. Make cache/memory tradeoffs explicit.
- Quantization, NPU partitioning, and model distillation are **separate, optional** experiments after full-fidelity GPU correctness and baseline performance are established.

**Gate:** Publish per-stage latency, end-to-end throughput, startup/compile time, peak VRAM, host-to-device bytes, and output-quality deltas for representative resolutions and window lengths, including comparison with a Windows GPU provider where available. No unevidenced speedup claims.

## Test and validation matrix

- Build/test native code on `windows-latest` and current Ubuntu LTS; run hardware GPU tests on real AMD hardware, not a GitHub-hosted CPU runner.
- Exercise feature-based compatibility, with an AMD RDNA1 architectural target and Radeon 680M-class integrated GPU as a practical lower-performance reference where viable. Record actual extensions/features rather than inferring from model names.
- Invariants: shapes and crop alignment; color transforms and negative/HDR values; alpha edge quality; temporal smoothness without erased motions; hint identity and carry; continuous recurrent state; deterministic output within defined numerical tolerance.
- Use layerwise numerical comparisons, masked MAE/RMSE and worst-case errors, boundary-focused alpha metrics, temporal flicker/identity tests, and real-world visual fixtures. Numerical thresholds follow measured reference variation and production relevance, not arbitrary global thresholds.
- Benchmark `720p` and `1080p` first; expand to `1440p` and `4K` with bounded-VRAM modes. Use fixed source clips, identical window settings, and warmed-up/steady-state and cold-start measurements.

## Distribution and scope

- Preserve the Corridor Key Licence and third-party notices; do not relicense the fork as Apache/MIT. CorridorKey derivatives are subject to release/attribution, redistribution, and commercial-integration restrictions in `LICENSE`.
- Maintain upstream-compatible CLI and output semantics while adding an explicit backend choice and structured inference API.
- Do not couple the keying core to a particular compositor, tracking framework, 3D engine, or pose-estimation pipeline. Keep high-precision mattes, foreground RGB, and frame timing accessible to downstream tools.
- Future NPU+GPU work should be separately benchmarked against GPU-only end-to-end execution, accounting for compilation support, memory copies, synchronization, and precision differences.

## Immediate next change set

1. Add a dependency-light diagnostic that reads checkpoint metadata and reports graph/precision/capability requirements without initializing CUDA.
2. Introduce deterministic reference-fixture capture and per-component export probes.
3. Produce a conversion/coverage report and choose the first completely viable AMD backend from measured evidence.
4. Implement end-to-end execution before optimizing any isolated operator beyond what is necessary for correct full-model support.
