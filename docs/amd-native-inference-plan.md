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

The proof currently exercises twenty-two configurations:
- RMSNorm with channel-first, channel-last and small NCHW output shapes
- SwiGLU
- Full-spatial and full-spatiotemporal attention
- Three shifted Swin-2D variants (diagonal, row, column), including padded boundaries and multi-batch routing
- Two shifted Swin-3D variants, including padded, multi-frame and multi-batch region routing
- Full-spatial, CSwin, and shifted-Swin blocks at the reduced Stage 2 token layout
- Native patchification and depatchification, each with non-overlapping and overlapping convolution kernels
- Two source-derived grouped Swin-2D region-mask configurations, including a small degenerate window and a batched padded profile
- Two Stage 1-shaped RMSNorm configurations covering high-rank token tensors and sixteen-dimensional head normalization

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

To obtain only the official CorridorKey checkpoint without downloading
the foundation networks, run `python scripts/download_weights.py
--stage2_checkpoint_only` from the repository root. This intentionally
does not fetch MoGe, C-RADIO, RVM, or foundation statistics because the
standalone Stage 2 checkpoint proof does not instantiate them.

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


## Stage 1 integrated temporal graph gate

`scripts/ci/prove_stage1_vulkan.py` now instantiates the actual `Stage1` class and feeds two frames of RGB, subject hint/quality, and the documented pre-normalized MoGe feature interface into the full stage-one forward path. This avoids instantiating foundation networks during the import proof without substituting another Stage 1 architecture. The exercised path includes RGB/hint patchification, the /16 encoder, /64 temporal body, /16 intermediate refinement, /16 decoder and the source baton/auxiliary output heads. The probe deliberately uses reduced channel widths, disabled checkpointing and no recurrent sites; these are explicit profile settings, not claims of compatibility with trained production topology.

The export step compares the source `Stage1` output with the fully decomposed exported CPU graph using the same sample inputs and unchanged numerical tolerances. Its first two-frame export passed the CPU gate but a direct high-rank RMSNorm graph triggered an IREE `arith.addf` type error under Vulkan compilation. Flattening the leading dimensions before invoking the **original RMSNorm module** (then reshaping the output) preserves last-axis normalization, epsilon, and learned weights. An independent CPU equivalence check enforces the source semantics before export. That legalized graph now **compiles for RDNA1 on Windows and numerically executes on Linux software Vulkan**, with both baton and auxiliary-head outputs checked. The standalone 22-case Vulkan proof is also green on both platforms.

A stronger regression fixture additionally activates the source's initially zeroed mid-stage detour projections, /64-to-/16 upsample contribution, and learned late skip. These are fixture weights, not a production model modification; they prevent unused residual pathways from masking compiler defects. That enhanced test has passed source-to-export equivalence, Windows RDNA1 compilation, and software-Vulkan numerical execution on Linux (Actions run 40034698628). It is a stronger correctness proof, not a measurement of actual AMD hardware performance.

Still required separately: actual checkpoint Stage 1 configuration and tensor loading, RVM and MoGe/C-RADIO feature integration, recurrent memory-state inputs/outputs, time-stride and streaming/Hann-window scheduling, high-resolution padded shapes, and practical AMD execution/performance. Validation on the reduced source model cannot replace any of those gates.

## Local AMD importer environment (Windows and Linux)

Use an existing Conda environment for checkpoint validation, or create an
isolated one if desired. The installer resolves compatible packages without
requiring Python 3.12 or exact local package versions. The reproducible CI
profile uses separately pinned versions.

```powershell
python scripts/setup_amd_importer.py --profile checkpoint
python scripts/download_weights.py --stage2_checkpoint_only
python scripts/ci/prove_checkpoint_stage2.py weights/corridorkey_v2.safetensors --height 64 --width 64
```

For source-to-Vulkan export and trained checkpoint compilation, install the
additional compiler dependencies in the same environment:

```powershell
python scripts/setup_amd_importer.py --profile vulkan
python scripts/ci/prove_checkpoint_stage2.py weights/corridorkey_v2.safetensors --height 64 --width 64 --compile --target rdna1
```

On an AMD system with a configured Vulkan ICD, add `--execute-vulkan` to
perform numerical device/reference comparisons. The checkpoint profile does
not import or require IREE; neither profile installs a CUDA or ROCm build of
PyTorch. The setup script supports `--dry-run` to inspect the commands.
These are host-side model import/reference dependencies, not requirements for
the eventual native inference runtime. If a dependency combination fails,
record the versions and the failing operation before constraining local
environments.

The Stage2 exporter proves the dense shifted-window mask's real-query
attention membership against the source's grouped-window representation.
Because dense and grouped SDPA use different matrix/reduction shapes,
a source-to-legalized FP32 comparison uses 1e-5 absolute/relative tolerance;
the independent source-to-export and Vulkan checks remain separately enforced.

## Trained Stage2 Vulkan differential diagnosis

When a trained Stage2 graph imports and compiles but device output diverges,
run the structural-boundary diagnostic with the same checkpoint and image
shape:

```powershell
python scripts/ci/diagnose_stage2_vulkan.py --checkpoint weights/corridorkey_v2.safetensors --height 64 --width 64 --target rdna1 --output build/checkpoint-stage2-diagnostic
```

The diagnostic reuses the source Stage2 exporter and probes the checkpoint's
actual patchification, every pre-/post-skip transformer block input, the late
patch branch where present, and depatchification. It first checks that the
instrumented graph and its re-export still reproduce the original CPU output.
Each captured CPU and Vulkan tensor is written to the output directory, with
per-boundary error counts, maximum and percentile differences, and the earliest
divergent boundary in `report.json`. A different output signature can change
compiler optimization; if the original failure disappears under instrumentation,
the diagnostic report is not evidence that the original executable is fixed.

The `--probe-only` option checks probe capture and CPU re-export without an
IREE/Vulkan compile. The CI fixture uses that mode to exercise the probe
selection logic without downloading production weights.

### Isolating the first trained full-spatial block

If `input_pre_0` agrees and `input_pre_1` diverges, use the focused
checkpoint diagnostic:

```powershell
python scripts/ci/diagnose_stage2_vulkan.py --checkpoint weights/corridorkey_v2.safetensors --height 64 --width 64 --target rdna1 --first-block --output build/checkpoint-stage2-block0
```

This probes the original block 0's RMSNorm, linear QKV, normalized Q/K,
gated full-spatial attention, output projection, learned spatial
depthwise value/gate convolutions (when enabled), and SwiGLU MLP.
The first block in the ten-block production architecture does not
enable the separate QKV depthwise convolution. Probes are original
model tensor boundaries exported from the same graph; no independent
network or replacement attention implementation is used. Inspect
`report.json` and the saved CPU/Vulkan intermediate tensors to
identify the first divergent operation. CPU probe mapping is exercised
on the small source-derived fixture in both CI environments.


## Isolated production-width RMSNorm validation

When the first transformer block's input agrees but `pre0_norm1`
diverges, isolate that original normalization from the rest of Stage2.
The following test loads the trained block 0 normalization parameters,
captures the exact source input from the trained model, and compiles
both native-rank and flattened-token views of the same source RMSNorm:

```powershell
python scripts/ci/prove_checkpoint_rmsnorm.py --checkpoint weights/corridorkey_v2.safetensors --height 64 --width 64 --target rdna1 --execute-vulkan --output build/checkpoint-rmsnorm
```

The source output is checked against the flattened-token source output
at zero tolerance before either is exported; exported CPU inference
is independently compared to the source. Both Vulkan executions must
meet the existing `3e-3` absolute/relative gate. The command records
per-variant differences and the actual input/CPU/Vulkan tensors, so a
failure is evidence of a particular backend operation rather than a
reason to loosen the threshold. The source-derived 1280-wide fixture
is compiled for RDNA1 on Windows and executed on software Vulkan on
Linux CI without downloading trained weights.

## Production-width RMSNorm backend reduction investigation

The actual trained model's initial `pre_blocks.0.norm1` produces the first
incorrect Vulkan values at 1280 channels. In a source-preserving standalone
test, the GPU input, elementwise square, and trained affine weight match CPU
exactly. The first mismatch is the mean-square reduction across 1280 channels.
Both native-rank, flattened-rank, and two-step 256-channel source-equivalent
reductions produce the same wrong output on the tested AMD device.

The standalone proof supports an opt-in compiler-lowering comparison, retaining
all source weights and the same final numerical acceptance criteria:

```powershell
python scripts/ci/prove_checkpoint_rmsnorm.py --checkpoint weights/corridorkey_v2.safetensors --height 64 --width 64 --target rdna1 --execute-vulkan --explore-reductions --output build/checkpoint-rmsnorm
```

This additionally tests a constant-vector matrix contraction of the squared
channels, a power-of-two zero-padded reduction divided by the original channel
count, and the original source normalization compiled at `O0`. Each case
requires source/export CPU equivalence before Vulkan compilation and retains
its own .mlir, .vmfb, GPU array, and numerical report. These are comparative
compiler diagnostics, not production fallback selections. The report stores
the 16 mean-square CPU and Vulkan scalar values and their ratio; a consistent
ratio can expose a faulty padded divisor. No alternate method becomes the
Stage2 export lowering until the trained GPU gate is passed and the performance
trade-off is measured.

On the current IREE llvmpipe compiler, the optional seven-output term tracing
graph fails codegen when lowering a vector constant, independently of the
standalone RMSNorm operation. CI exercises the uninstrumented native, flattened,
and chunked source-equivalent normalization proofs with `--no-trace-terms` on
Linux, while keeping full term tracing enabled by default for hardware-local
reproduction where it compiles. Neither behavior relaxes the numerical gates.

## 1280-channel reduction width sweep

With the official checkpoint's first-block input captured as
`build/checkpoint-rmsnorm/input.npy`, independently compile exact
sum-of-squares and mean-of-squares for multiple channel widths:

```powershell
python scripts/ci/prove_vulkan_reductions.py --source-input build/checkpoint-rmsnorm/input.npy --target rdna1 --execute-vulkan --output build/reduction-sweep
```

A standalone source-derived suite (using deterministic inputs) runs two
representative widths in CI, compiling for RDNA1 on Windows and executing
on Linux software Vulkan. The local suite reports each of the 16 GPU/CPU
reductions, error counts, and min/median/max scaling, and names the
Vulkan devices visible to IREE. The default IREE device is used for
execution; on systems with multiple Vulkan devices, verify device selection.

This test separates a mean-only division bug from a broken sum-of-squares
reduction, establishes the first failing channel width, and keeps the
compiler target explicit. Choose the target architecture that matches
the actual GPU (e.g. rdna1 for RX 5000, rdna2 for RX 6000, rdna3 for
RX 7000); do not mistake successful RDNA1 compilation for a correctness
test on a newer architecture. For generated SPIR-V and lowered MLIR use
`--dump-executables`; all compile and device artifacts are written under
`build/reduction-sweep`. This is an isolated compiler regression test,
not a hardcoded source-model normalization correction.

## AMD Vulkan reduction correctness — reproduction details

A local Radeon 8060S running AMD's proprietary Vulkan LLPC driver
26.8.1 reproduced an exactly **2×** result for both `sum(square(x))`
and `mean(square(x))` on all 16 spatial tokens at widths 256, 512, 768,
1024, 1280, and 2048. Widths 32, 64, and 128 numerically agree.
Using the `rdna3` IREE target instead of `rdna1` did not correct
256- or 1280-channel reductions, and the trained Stage2 output still
failed its source comparison. This disproves a simple architecture-family
target mismatch as the solution.

The standalone reduction sweep now also supports
`split128_sum_square` and `split128_mean_square`. Those expressions
reduce each 128-channel slice independently and sum its partial results,
preserving the original normalization domain and divisor. In contrast
to dividing a known failing result by two, these tests evaluate a
mathematically equivalent reduction graph and expose whether IREE
merges the source's independent small reductions into the failing
wide-reduction lowering. No such alternative has been adopted for
production inference. Compiler and runtime package versions are
recorded alongside the device and per-width numerical errors.

Suggested focused run, using the actual checkpoint's cached RMSNorm input:

```powershell
python scripts/ci/prove_vulkan_reductions.py --source-input build/checkpoint-rmsnorm/input.npy --widths 128,192,224,255,256,257,1280 --operations sum_square,mean_square,split128_sum_square,split128_mean_square --target rdna3 --execute-vulkan --output build/reduction-sweep-segmented
```

For follow-up compiler/driver investigation, collect the generated SPIR-V
with `--dump-executables` and compare the reduction subgroup sizes,
workgroup sizes, and scheduling across the 128/256 boundary.
