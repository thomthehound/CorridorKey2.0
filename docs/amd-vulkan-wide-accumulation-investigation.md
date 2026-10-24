# AMD Vulkan wide-accumulation correctness investigation

**Status:** Reproduced on physical AMD Vulkan; underlying compiler/driver/runtime fault not yet attributed. A replacement AMD Vulkan driver reproduced the same Stage1 failure and segmented success on October 24, 2026 (replacement version not yet recorded). Segmented source-equivalent lowerings are correctness candidates, not root-cause fixes.

**Scope:** CorridorKey 2.0 Stage2 RMSNorm reductions and Stage1 trained downsample convolution/GEMM, imported with Torch Export / IREE Turbine and compiled through IREE's Vulkan SPIR-V backend. The exported computation retains checkpoint parameters and source arithmetic. DXC is not part of this direct SPIR-V compilation path.

## Environment and reproduction controls

The previously recorded physical-GPU environment for the Stage2 investigation was Radeon 8060S, AMD proprietary Vulkan LLPC 26.8.1, IREE compiler/runtime 3.13.0. The exact driver, IREE, PyTorch, and Turbine versions used for each subsequent Stage1 reproduction must be recorded independently rather than inferred from this baseline. Both `rdna1` and `rdna3` IREE targets exhibited the Stage2 issue on the same physical device.

A driver A/B test should keep the same repository revision, original checkpoint, Python/importer/compiler/runtime versions, inputs, target, optimization flags, and GPU. Record both Vulkan device/driver identities with `vulkaninfo --summary` (or the equivalent device report), alongside the IREE package versions. Run both comparisons without changing graph legalization between driver installations. Preserve each run's `report.json`, CPU arrays, Vulkan arrays, and available compiler artifacts in distinct output directories.

The first reported replacement-driver run did not change the tested Stage1 values: both `gemm4` and `gemm4runtime` still fit an accumulation scale of `2`, with 511/512 CPU mismatches, while `seg128` still matches CPU at all 512 elements and a maximum absolute error of `7.87e-6`. This is evidence against that particular driver update fixing the fault, not evidence that all AMD driver versions behave identically.

A driver-dependent result localizes the problem toward driver-sensitive code generation or execution, but does not by itself prove a driver bug: IREE may generate target-dependent code whose behavior varies with the device capabilities reported by the driver. A driver-independent result also does not exonerate IREE.

## Observed failure A — trained Stage2 RMSNorm

- Original trained Stage2 source and CPU-export outputs agree. The first divergent trained GPU boundary is `pre_blocks.0.norm1`, reducing 1,280 feature channels.
- Input, elementwise squared values, and trained affine parameters match CPU. The unsegmented sum and mean of squared values are approximately **2×** the CPU results on the observed AMD Vulkan device.
- Isolated widths 32, 64, and 128 pass. Widths 256, 512, 768, 1024, 1280, and 2048 reproduce doubling; width 192 behaves input-dependently. The failure is not established to have a simple universal width threshold.
- Segmenting the same full-domain sum into contiguous blocks of at most 128 channels, then adding partial totals and dividing by the original width exactly once, matches the CPU result. At width 1280, the recorded mean-square maximum error was approximately `4.8e-7`.
- Trained Stage2 with opt-in `--rms-reduction segmented128` passed the tested 64×64 physical-AMD checkpoint proof. This is not a performance result or complete-model acceptance.

Reproduction and scope: `scripts/ci/prove_vulkan_reductions.py`, `scripts/ci/prove_checkpoint_rmsnorm.py`, `scripts/ci/diagnose_stage2_vulkan.py`, and `scripts/ci/prove_checkpoint_stage2.py`. The broader original Stage2 evidence and exact historical commands remain in `docs/amd-native-inference-plan.md` and [PR #1](https://github.com/thomthehound/CorridorKey2.0/pull/1).

## Observed failure B — trained Stage1 downsample / GEMM

The original checkpoint has 512 Stage1 tensors. The Stage1 differential passes the stem and encoder probes (maximum absolute encoder-output error `1.049e-5`, no mismatches with the diagnostic tolerance) and first diverges at the `down` boundary.

- Downsample: `Conv2d`, input `(1, 512, 4, 4)`, weights `(512, 512, 8, 8)`, stride 4, padding 2, output `(1, 512, 1, 1)`. The output comprises 512 dot products of **8192 contributing products** at this input geometry.
- Native convolution and IREE's compiler-side im2col option give identical wrong Vulkan values. Explicit spatial gather + GEMM gives the same values. A cropped 4×4 convolution also fails, with a different numerical pattern.
- An equivalent 4×4 center-weight GEMM, with either captured original checkpoint parameters or weights and bias supplied explicitly as runtime tensors, gives the same GPU defect. This argues against embedded-weight materialization as the primary cause.
- The fitted accumulator-only factor for both unsegmented GEMM variants is `2.0000000021`. The error is consistent with `GPU ≈ 2 * (CPU - bias) + bias`: maximum residual `1.72e-5`; **0/512** mismatches under the diagnostic tolerance for that hypothesis.
- A source-equivalent segmented implementation using independent 128-element GEMMs followed by summation and one bias addition gives fitted accumulation factor `1.0000000122`, maximum error `7.87e-6`, and **0/512** GPU/CPU mismatches under the unchanged acceptance tolerance.
- An explicit `torch.nn.functional.unfold` variant agrees with CPU but fails IREE compilation at a `torch.constant.int` legalization step; it does not provide an independent GPU correctness result.

These numerical results come from physical-device runs of `scripts/ci/diagnose_stage1_vulkan.py` and `scripts/ci/diagnose_stage1_down_vulkan.py`, with the actual checkpoint and captured encoder activation. The independent generalized lowering `scripts/ci/stage1_down_lowering.py` and its use in complete Stage1 export are **not yet validated on the physical GPU**. No connected Stage1→Stage2 Vulkan correctness claim follows from the isolated passing GEMM.

Reproduce the already-validated comparison on an unchanged checkout:

```powershell
python scripts/ci/diagnose_stage1_down_vulkan.py `
  --checkpoint weights/corridorkey_v2.safetensors `
  --height 64 --width 64 --frames 1 `
  --target rdna3 `
  --modes gemm4,gemm4runtime,seg128 `
  --output build/down-driver-comparison
```

Use a distinct `--output` directory for each driver. The script stores GPU and CPU `.npy` values and a `report.json` with fitted accumulation scaling and bias-aware residual statistics. A nonzero exit is expected when intentionally retaining the failing unsegmented control variants.

## Integration after fixing the first Stage1 boundary

In the subsequent **full trained Stage1** run with
`--down-reduction segmented128`, the downsample probe matched GPU/CPU
(maximum absolute error `7.51e-6`, 0/512 mismatches), and the following
`fuse.norm` also matched (`4.77e-6`, 0/2816 mismatches). The next
divergence moved to `fuse` (maximum error `3.1314`, 1531/1536
mismatches), and the complete baton remained incorrect. This confirms the
corrected downsample in the original trained graph and independently exposes
a second wide arithmetic boundary.

The original `FusionLinSwiglu` source contains four learned `nn.Linear`
operations: `linear_shortcut`, `w_gate`, `w_up`, and `w_down`. An
opt-in `--fuse-reduction segmented128` now changes only their export-time
dot-product grouping; the original modules and exact parameters are restored
after capture. Diagnostic probes expose each of those source projections
rather than treating `fuse` as an opaque block. Source-parity and
cross-platform import checks are mandatory. **No physical-device success for
this new fusion lowering has yet been demonstrated.** The original native
linear operators remain available for continued backend investigation.

## Trained Stage1 temporal attention QKV evidence

With segmented `down` and `fuse` projections, physical AMD Vulkan
matches all measured Stage1 operations through `body_64.0.norm1`. The
next first-divergent operation is the source `body_64.0.attn.qkv.qkv`
learned linear projection, mapping 1536 input features to 4608 outputs:
4583/4608 values mismatch, with maximum absolute error `1.7739406`.
The Q/K normalization, attention gate, and output projections are already
downstream of the corrupted QKV input; their mismatches do not establish
independent defects.

The export path now offers the optional
`--linear-reduction segmented128` to apply the same source-equivalent
partial-sum lowering to original wide `nn.Linear` operations across
Stage1, without modifying learned parameters or requiring a separate
fix for each projection. The existing `--fuse-reduction segmented128`
remains an independent control. The original source modules are restored
after graph export. **The broad lowering has not yet been proven correct
for the complete trained graph on the physical AMD device.** Compare its
device results and performance with the failing native baseline before
claiming a fix.

## Root-cause hypotheses (unresolved)

1. **IREE lowering/tiling of wide reductions or matmuls.** The repeated 2× accumulation with both convolution and independent GEMM is consistent with duplicate contribution, lane/subgroup reuse, or a malformed reduction tree. These are hypotheses, not observations of generated machine code.
2. **AMD Vulkan driver SPIR-V compilation.** A driver/compiler fault could mishandle otherwise valid wide accumulations; the forthcoming same-graph cross-driver comparison is necessary.
3. **Runtime descriptor/layout or execution issue.** Runtime-provided weights exhibit the same effect, reducing the likelihood of a constant-parameter-only binding failure, but readback and resource layout should still be examined with intermediate device buffers and a minimal kernel.
4. **Graph-specific compiler optimization.** The different `crop4` failure and the fact that `seg128` succeeds suggest comparing generated dispatch shape, partial reduction code, memory layouts, and compiler pipelines rather than assuming all variants use the same machine instructions.

No evidence presently attributes the defect to DXC. IREE's Vulkan backend emits SPIR-V for the selected Vulkan implementation; attribution must follow executable inspection and cross-driver reproduction.

## Investigation plan and acceptance gates

- **Driver comparison:** Repeat the same isolated Stage1 `gemm4,gemm4runtime,seg128` tests and Stage2 width sweep on old and new drivers with unchanged compiled-tool versions and identical input arrays. Capture driver, Vulkan device, IREE version, target, run status, and per-variant errors.
- **Cross-implementation reproduction:** Where available, execute the same kernel/VMFB on AMD's proprietary Vulkan driver, Mesa RADV, and software Vulkan (lavapipe). Record precisely which backends pass/fail; software Vulkan is not a substitute for physical AMD hardware validation.
- **Minimal reproducer:** Reduce the 8,192-element dot and 1,280-element squared reduction to deterministic small standalone models with known analytic answers and explicit runtime operands. Sweep widths, shape/stride, output channels, and scalar/subgroup/tiling choices.
- **Compiler evidence:** Save imported Torch-MLIR, optimized IREE IR, generated SPIR-V, dispatch metadata, and (where tooling permits) driver ISA. Compare passing 128-element segments with failing unsegmented dispatches for duplicate accumulation, incorrect lane masks, and buffer indexing.
- **Upstream fix:** Locate the first incorrect compiler/runtime stage or driver transformation; report an upstream bug only with reproducible input, version/driver matrix, IR/SPIR-V and actual/expected values. Prefer a reusable compiler/backend correction over a CorridorKey-specific numerical patch.
- **Regression/performance:** Add a targeted GPU correctness regression covering widths on both sides of the failure and production checkpoint profiles. Compare corrected native reductions against segmented source-equivalent lowerings for latency, throughput, allocation, and memory bandwidth. Keep source-faithful segmented lowerings opt-in until representative trained-graph correctness and performance are demonstrated.

**Success criterion:** Source-equivalent full-width operations compile and execute correctly on supported AMD Vulkan drivers, with a verified root cause and regression tests. The segmented implementations remain reproducible correctness controls and must not be silently replaced by a multiplicative output correction.
