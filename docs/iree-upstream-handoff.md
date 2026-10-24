# IREE upstream handoff: subgroup correctness, Windows Python, and Vulkan readback

This document is the context-stable handoff for the CorridorKey2.0 AMD-native
Vulkan port. The local work is on
[`amd/checkpoint-environment`](https://github.com/thomthehound/CorridorKey2.0/tree/amd/checkpoint-environment),
tracked by [draft PR #1](https://github.com/thomthehound/CorridorKey2.0/pull/1)
and [issue #2](https://github.com/thomthehound/CorridorKey2.0/issues/2).

**Upstream IREE pull requests have not been opened.** There are **three distinct upstream candidates**: two previously confirmed
issues (SPIR-V subgroup correctness and Windows Python invocation), plus an
independently reproduced Vulkan/Python runtime host-import alignment defect.
Keep their patches and tests separate. The combined trained Stage1 → Stage2
runtime replay now **passes** with the local HAL staging-readback workaround.

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
reductions in this test. **The combined IREE-only Stage1 → Stage2 runtime
replay now passes** with the local staging-readback workaround; see §3.
The frozen foundation networks remain outside this proof.

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

**Status: reproduced and source-localized; runtime upstream candidate.
Combined native trained Stage1 → Stage2 now validated with the local
HAL staging-readback workaround.**

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
fallback instead of importing unaligned NumPy memory. The follow-up
replay initially failed with `RuntimeError: bad cast` because the helper
passed NumPy's scalar class `np.float32` to `MappedMemory.asarray`
instead of a `np.dtype('float32')` descriptor. That helper call was
corrected and its mock test now enforces the binding's dtype contract.
IREE 3.13.0
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

**Passing hardware replay reported 2026-10-24:**

```text
[native Vulkan partitions] trained Stage1 -> Stage2 executed;
median wall-clock=79.229 ms (includes interstage readback/reupload)
[trained wave32] Compiled Native replay passed.
```

The report is stored at
`build/trained-patched-iree/native/execution-replay-2/report.json`.
This verifies **combined trained Stage1→Stage2 native Vulkan computation**
with original source-reference comparisons, via the explicit CPU
NHWC→NCHW bridge. The 79.229 ms figure includes runtime dispatch,
readback, CPU transformation and GPU reupload; it is **not** a pure
GPU-kernel timing or an optimized full-application benchmark.
It does **not** establish a GPU-resident interstage bridge, execution of
the frozen foundation networks, a complete video-matting application,
or production-level throughput. Those remain separate milestones.

To replay without compiling again, use the command above. The initial
failure folders remain available; subsequent replays receive numbered
output directories.

## Required upstream regression tests and CI portability contract

These are mandatory acceptance criteria for preparing the separate IREE
upstream changes in the next work session, not merely ideas for later.

### Windows must build without Unix/Python-command assumptions

- Add a **Windows MSVC + Ninja** regression using a valid explicitly
  configured `Python3_EXECUTABLE` pointing to `python.exe` and **no
  usable `python3` executable on PATH**. A stub named `python3` that
  fails with a clear error is a useful guard against accidentally falling
  back to it, provided the selected interpreter is still accessible.
- Run the real code-generation steps (VM ISA `op_table.h`, encoding
  `encoding_table.c/.h`, and the plugin ABI generator), check outputs,
  and verify reconfiguration and incremental rebuild without cleaning
  previously compiled objects. A configure-only smoke test is insufficient.
- Preserve both CMake and Bazel generation workflows: fix the shared
  `iree_genrule` CMake handling, not files generated from `BUILD.bazel`;
  avoid incorrectly rewriting arbitrary shell commands or breaking paths
  containing spaces. Confirm `Python_EXECUTABLE` and `Python3_EXECUTABLE`
  remain consistent when both are needed.
- Test at least `windows-latest` and supported Linux CI for the changed
  build rule. The upstream specialized Windows runner happens to install
  `python3`; **passing only on that preconfigured runner does not
  exercise the original failure condition**.

### Vulkan is a first-class AMD path, not a HIP/ROCm fallback

- Compiler tests must cover a Vulkan target advertising **both
  `subgroup_size_choices = [32,64]`**, with explicit assertions that
  the preferred subgroup width reaches `translation_info`, SPIR-V
  `entry_point_abi`, and serialized Vulkan `PipelineDef.subgroup_size`.
  Exercise **both static and dynamic reductions** and workgroup shapes
  that expose the earlier wave32/wave64 mismatch (notably 192 channels).
  Also cover a fixed-width target, rather than hard-coding wave32.
- Include runtime tests of native sum/mean-square reductions, especially
  widths 128, 192, 256, 512 and 1280, and a **per-channel one-hot
  multiplicity test** to catch double-counting that aggregate numerical
  checks can miss. Use Vulkan explicitly. Do not silently substitute
  CPU execution, HIP, ROCm, or segmented source rewrites.
- A Windows and Linux Vulkan software ICD can provide general smoke
  coverage when their subgroup capabilities support the case; the
  original dynamic subgroup-width contract must additionally be tested
  on real physical hardware supporting both wave32 and wave64.
  Report unsupported capabilities and skips explicitly; don't represent
  a skipped GPU test as a hardware pass.
- Do **not** make HIP or ROCm a prerequisite for configuring, compiling,
  or executing IREE's Vulkan correctness tests. Where backend-specific
  paths differ, test them independently. Prefer source-accurate cross-backend
  interfaces to vendor-specific assumptions.

### Host-memory import and readback must honor Vulkan's constraints

- Reproduce `DeviceArray.to_host()` on a Vulkan device with nontrivial
  `minImportedHostPointerAlignment` (4096 on the observed Radeon).
  Test both pointer and allocation-size alignment; a bit indicating
  `IMPORTABLE` does not mean a random NumPy allocation satisfies
  `vkImportMemoryHostPointerInfoEXT` requirements.
- Add a mocked allocator test in IREE's Python runtime suite to force
  the importability branch **and reject unaligned pointers**, verifying
  either a properly aligned import or safe fallback to HAL-owned staging.
  Also exercise the existing staging path, synchronization, the NumPy
  dtype-descriptor contract, and returned-array lifetime after unmapping.
- Benchmark the appropriate fast aligned-import path if supported;
  correctness should not require disabling host import globally or
  degrading HIP/CUDA readbacks. The regression must explicitly cover
  Vulkan without any HIP/ROCm dependency.

### Upstream submission boundaries

Prepare **separate** IREE PRs with focused regression tests and independent
reproducers: (1) SPIR-V reduction subgroup contract, (2) Windows CMake
Python executable selection, (3) Python Vulkan host-import alignment.
Don't assert the third runtime bug is a compiler problem, and don't
turn the first into a blanket wave32 setting. Upstream PRs/issues are
**not yet opened**. Attach the passing Radeon hardware proof and the
portable Windows repro. Keep repository handoff status current when
upstream work proceeds.

## Upstream duplication and maintainer preflight (2026-10-24)

A search of IREE's current `main` code, open/closed issues, PRs and
contribution guide found **related work, but none of these three
specific correctness/build failures is currently fixed in `main`**.
Repeat the issue/PR search when preparing each submission.

1. **Subgroup bug — prior unmerged implementation:**
   [IREE PR #25021](https://github.com/iree-org/iree/pull/25021),
   closed unmerged October 11, 2026, contains the **same
   `SubgroupReduce` missing required-size fix**, plus a mixed
   `[32,64]` regression. The author withdrew it following a
   contribution-policy compliance discussion, not a demonstrated
   correctness rebuttal. It also tackles
   [#18516](https://github.com/iree-org/iree/issues/18516) and
   [#20318](https://github.com/iree-org/iree/issues/20318), separate
   subgroup-reduction expansion cleanup tasks. **Our upstream fix
   should remain narrowly scoped to correctness**, reference #25021
   and credit prior discovery appropriately, and avoid broad unrelated
   reduction-pipeline changes.

   **Required refinement of our local v3.13 patch:** #25021 records
   `requestedSubgroupSize` **only for a target advertising multiple
   supported subgroup widths**. Our current local patch sets it
   unconditionally. Requesting a size unnecessarily can fail on
   fixed-width Vulkan devices without subgroup-size-control support.
   Rework the upstream patch with a fixed-width negative test and a
   mixed-width positive test. Do not modify the already-proven local
   compiler build until a separate regression pass validates the
   portable upstream change.

   Related but different:
   [issue #24688](https://github.com/iree-org/iree/issues/24688)
   concerns dangerous Vulkan default-target assumptions;
   [draft PR #24726](https://github.com/iree-org/iree/pull/24726)
   addresses the default-target policy, not these two translation
   call sites.

2. **Windows Python invocation:** no matching existing patch located
   for the generated genrule commands' literal `python3`.
   [#18813](https://github.com/iree-org/iree/issues/18813) discusses
   Windows CI frequency/infrastructure rather than our selected-Python
   failure. Earlier [PR #18370](https://github.com/iree-org/iree/pull/18370)
   and [PR #11696](https://github.com/iree-org/iree/pull/11696) deal
   with interpreter discovery/pinning but do not remove the hard-coded
   command. Current `main` still embeds literal `python3` in the
   generated VM ISA and plugin ABI rules.

3. **Vulkan Python readback alignment:** merged
   [PR #23792](https://github.com/iree-org/iree/pull/23792)
   introduced the host-import-based `DeviceArray._copy_to_host`
   path. Its review **explicitly discussed Vulkan support** and added
   allocator compatibility queries, but missed the fact that
   `IMPORTABLE` does not guarantee a NumPy allocation satisfies
   pointer/length alignment. Historical
   [issue #7242](https://github.com/iree-org/iree/issues/7242)
   already discussed Vulkan host import alignment and safe fallback.
   [PR #24737](https://github.com/iree-org/iree/pull/24737) changes
   Vulkan extension-function availability, **not** Python's pointer
   alignment. The erroneous `np.empty` import branch is present in
   current `main`. Reference #23792 and #7242 when filing the
   focused runtime correction, avoiding a duplicated general
   feature request.

**Contributor/review protocol:** read the current
[IREE contribution guide](https://iree.dev/developers/general/contributing/)
and [CODEOWNERS](https://github.com/iree-org/iree/blob/main/.github/CODEOWNERS)
before opening PRs. The project requires a DCO `Signed-off-by`
trailer on commits from non-organization members, focused changes,
in-tree regression tests, addressed review comments and typically
squash merges. New contributors may need a maintainer to authorize
CI runs; `ci-extra: windows_x64_msvc` can opt into Windows coverage.
Their contributor rules also include specialized restrictions for
`good first issue` work: do not associate a narrow independent
correctness repair with an unrelated newcomer cleanup ticket without
clarifying the scope first. Refer to official current policies for
all contribution-origin declarations and acknowledgments.

**Evidence custody:** CorridorKey's repo contains the versioned fix,
scripts, test fixture, handoff and CI records. The raw successful
Radeon reduction JSON and combined native replay report are
user-generated artifacts from the physical machine, **not
public upstream attachments**; copy/sanitize the relevant results
and hardware properties before creating PRs. The passing combined
report path is
`build/trained-patched-iree/native/execution-replay-2/report.json`.
CorridorKey CI run
[190](https://github.com/thomthehound/CorridorKey2.0/actions/runs/40126750060)
completed successfully on both Windows and Ubuntu. These CI jobs do
not replace the user's physical Radeon proof.

### Preservation and handoff checklist

- Do not delete `F:\dev\iree-wave32-v3.13.0-build`; the successfully
  compiled compiler is reused for GPU tests and future A/B experiments.
- Keep the original failing reduction report, the passing 10-case
  wave32 report, native Stage1/Stage2 reports and executable dumps,
  the failed combined-host execution reports, and the **passing
  `execution-replay-2/report.json`** result (79.229 ms median).
- Keep upstream proposals **separate**: SPIR-V reduction subgroup
  contract, Windows CMake Python selection, and the new Python runtime
  Vulkan host-alignment candidate.
- A green CorridorKey Windows/Linux CI run proves project checks and
  unit tests, **not** physical GPU performance or correctness.
- No new upstream IREE issue or PR is represented as filed merely
  because it is listed here.
