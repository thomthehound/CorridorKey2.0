# IREE upstream handoff: subgroup correctness, Windows Python, and Vulkan readback

This document is the context-stable handoff for the CorridorKey2.0 AMD-native
Vulkan port. The local work is on
[`amd/checkpoint-environment`](https://github.com/thomthehound/CorridorKey2.0/tree/amd/checkpoint-environment),
tracked by [draft PR #1](https://github.com/thomthehound/CorridorKey2.0/pull/1)
and [issue #2](https://github.com/thomthehound/CorridorKey2.0/issues/2).

**Upstream IREE pull requests have not been opened.** There are two confirmed,
independent upstream candidates, and one newly identified runtime candidate.
Do not conflate them.

## 1. Confirmed IREE SPIR-V subgroup-size correctness defect

**Status: patched locally and GPU-validated. Ready to prepare upstream PR.**

Version: IREE compiler/runtime 3.13.0; target `rdna3`; Radeon 8060S,
AMD proprietary Vulkan driver 26.10.1 (LLPC). Device defaults to 64-lane
subgroups, permits 32–64, supports subgroup-size control.

In
`compiler/src/iree/compiler/Codegen/SPIRV/KernelConfig.cpp`,
`setReductionConfig` selects `subgroupSize =
target.getPreferredSubgroupSize()` and configures the reduction around
it. Both the static and dynamic `SubgroupReduce` branches previously
called `getSPIRVTranslationInfo(..., workgroupSize)` without supplying
the selected `subgroupSize`; the helper default is 0. Without a
nonzero translation subgroup size, SPIR-V entry-point ABI and Vulkan
`PipelineDef.subgroup_size` do not require the wave32 that codegen assumed.
The runtime already honors nonzero `subgroup_size` by using
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`.

Patch: [`patches/iree/3.13.0-require-subgroup-size.patch`](../patches/iree/3.13.0-require-subgroup-size.patch).
It adds the selected `subgroupSize` to **both** translation calls and
updates six upstream FileCheck expectations for 16/32/64 configurations.
It does not change the source model, checkpoint or reduction math.

**Before correction:** 128-channel sum-square passes; 192-channel
one-hot reduction double-counts 128 channels; 256 and 512 double-count
all channels.

**After correction (physical Radeon):** native `sum_square` and
`mean_square` each pass at widths **128, 192, 256, 512, 1280**,
with **zero numerical mismatches** and every one-hot channel counted
exactly once. Emitted SPIR-V MLIR includes
`spirv.entry_point_abi<subgroup_size = 32>`.

**Trained checkpoint:** native Stage1 (512 trained tensors) independently
compiles and matches the original source outputs; native Stage2 (226
trained tensors) independently compiles and matches source outputs
when driven by the trained Stage1 baton. Neither stage uses segmented
reductions in this test. The *combined runtime chain* remains a separate
acceptance condition (see issue 3 below).

Submit a narrow upstream fix with original/patched reduction output,
the 192-channel one-hot map, the IREE compiler test fixture, and physical
GPU properties. Do not claim that other subgroup-sensitive GEMM
lowering paths are fixed without direct evidence. A mixed-capability
`[32,64]` compiler test should be added if possible.

Reference:
[`docs/iree-3.13-wave32-fix.md`](iree-3.13-wave32-fix.md).

## 2. Confirmed cross-platform IREE CMake/Python invocation defect

**Status: reproduced on Windows; temporary CorridorKey build shim exists;
proper upstream fix remains to be implemented.**

IREE's CMake discovers `Python3_EXECUTABLE` correctly, but multiple
Bazel-generated CMake `iree_genrule` commands call a literal
`python3`. On Windows with an activated Conda environment containing
`python.exe` but no working `python3` command, Ninja fails after
thousands of LLVM/MLIR objects have compiled:

```text
FAILED: runtime/src/iree/vm/bytecode/isa/op_table.h
python3 .../build_tools/scripts/generate_vm_isa.py ...
Python was not found; run without arguments to install from the Microsoft Store...
ninja: build stopped: subcommand failed.
```

The same issue affects `encoding_table.c/.h` and the plugin ABI
generator. It exists in both IREE 3.13.0 and upstream `main` as
inspected 2026-10-24. The generated CMake files originate from
`BUILD.bazel`; do not hand-edit these generated files as the permanent
fix. An appropriate upstream implementation likely belongs in
`build_tools/cmake/iree_genrule.cmake`: route Python invocations through
the configured `Python3_EXECUTABLE`, preserving Bazel behavior, escaping
and other arbitrary genrule commands. Test on `windows-latest` with
`python3` absent from `PATH` while `Python3_EXECUTABLE` is configured.
IREE does have MSVC Windows CI, but its Python provisioning provides
`python3`, masking this portability failure.

The local builder
[`scripts/ci/build_and_prove_iree_wave32.ps1`](../scripts/ci/build_and_prove_iree_wave32.ps1)
supplies an invocation-local `python3.cmd` forwarding to the activated
Conda `python.exe`. It does not change Windows settings or rebuild
completed Ninja objects. This workaround is **not** the upstream fix.

## 3. IREE Python Vulkan host-import alignment failure

**Status: reproduced and source-localized; runtime upstream candidate;
combined native trained chain not yet proven end-to-end.**

With the patched compiler, Stage1 and Stage2 **both pass independently**,
but the combined IREE-only host aborts when calling
`DeviceArray.to_host()` for the Stage1 baton:

```text
INVALID_ARGUMENT; host allocation import pointer does not satisfy Vulkan
minImportedHostPointerAlignment 4096
```

In IREE 3.13.0's
`runtime/bindings/python/iree/runtime/array_interop.py`,
`DeviceArray._copy_to_host` queries import compatibility, but on its
`IMPORTABLE` branch it allocates a plain NumPy `np.empty` array and
calls `allocator.import_host_allocation`. NumPy does not guarantee the
device's required pointer or allocation-size alignment. Vulkan correctly
rejects it in
`runtime/src/iree/hal/drivers/vulkan/allocator.c`.
This is **not** a compiler/codegen failure and not evidence of
incorrect model arithmetic.

Upstream remedy to investigate: respect Vulkan host import alignment
for both pointer and size, or select the existing HAL-owned staging-copy
path whenever import requirements cannot be satisfied. Make
`_copy_to_host` robust on Vulkan devices reporting host-memory-import
compatibility and a nontrivial
`minImportedHostPointerAlignment`. Add device and mocked allocator
tests. An upstream runtime patch is **not yet implemented or filed**.

CorridorKey's execution-only local readback fix is isolated in
[`scripts/ci/iree_staged_readback.py`](../scripts/ci/iree_staged_readback.py)
and uses the same public HAL staging-copy operations as IREE's own
fallback instead of importing unaligned NumPy memory. IREE 3.13.0
exposes its buffer view on `DeviceArray` only as a private attribute;
this access is deliberately confined to that helper. A regression
test in `scripts/ci/test_iree_staged_readback.py` checks staging copy,
synchronization and returned-array ownership.

### Resume **without recompiling** the trained model

The previous independent native Stage1 and Stage2 runs generated VMFBs
and source-reference arrays in
`build/trained-patched-iree/native/stage1` and
`build/trained-patched-iree/native/stage2`. Preserve these files.

After pulling the latest CorridorKey branch, from the active Conda
environment:

```powershell
git pull --ff-only
.\scripts\ci\run_trained_with_patched_iree.ps1 -Mode Native -ReplayExisting
```

`-ReplayExisting` invokes only the IREE runtime on those **existing
compiled VMFBs** and previously written reference data. It does not
compile, reload checkpoints, rebuild LLVM/MLIR, or import PyTorch.
Results are written to `execution-replay/report.json` in the same
`native` proof folder, with subsequent replays numbered to preserve
prior reports. It compares both Stage1's baton/auxiliary outputs and
Stage2's output against the original independent reference tensors.

A passing replay will establish the **combined trained Stage1→Stage2
native Vulkan chain**, still using an explicit CPU NHWC→NCHW bridge.
It will **not** establish a GPU-resident interstage bridge, the
foundation-network execution path, a full video-matting application, or
end-to-end production performance. These are separate milestones.

### Preservation and handoff checklist

- Do not delete `F:\dev\iree-wave32-v3.13.0-build`; the successfully
  compiled compiler is reused for GPU tests and future A/B experiments.
- Keep the original failing reduction report, the passing 10-case
  wave32 report, the native Stage1/Stage2 reports and executable dumps,
  and the failed combined-host execution report.
- Keep upstream proposals **separate**: SPIR-V reduction subgroup
  contract, Windows CMake Python selection, and the new Python runtime
  Vulkan host-alignment candidate.
- A green CorridorKey Windows/Linux CI run proves project checks and
  unit tests, **not** physical GPU performance or correctness.
- No new upstream IREE issue or PR is represented as filed merely
  because it is listed here.
