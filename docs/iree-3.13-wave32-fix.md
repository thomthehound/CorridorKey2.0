# IREE 3.13.0 Vulkan subgroup-reduction fix

## Root cause in the compiler configuration

The native FP32 reduction on an AMD Radeon 8060S (proprietary LLPC 26.10.1)
was proven to count input contributions twice when reductions exceed a
single safe region. With one-hot inputs, 128 channels contribute once;
the first 128 of 192 contribute twice and the last 64 once; all 256 and
512 channels contribute twice. The Vulkan implementation reports default
`subgroupSize=64` but supports an explicit requested size of 32 for
compute shaders.

IREE v3.13.0
[`compiler/src/iree/compiler/Codegen/SPIRV/KernelConfig.cpp`](https://github.com/iree-org/iree/blob/v3.13.0/compiler/src/iree/compiler/Codegen/SPIRV/KernelConfig.cpp)
uses `target.getPreferredSubgroupSize()` to select 32 for the
`SubgroupReduce` pipeline. The algorithm explicitly lays out two-stage
warp reductions assuming that width. **Both** its dynamic and static
reduction paths call `getSPIRVTranslationInfo(..., workgroupSize)`
without passing the computed `subgroupSize`, leaving the translation
info's required size at zero. IREE's SPIR-V target serializes the
selected size from the entry-point ABI into the Vulkan executable's
`PipelineDef.subgroup_size`, and the v3.13 Vulkan HAL passes a
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` during pipeline
creation *only if the size is nonzero*.

This is a **GPU-validated compiler-to-Vulkan subgroup-contract omission**.
The patched IREE 3.13.0 compiler passed native sum/mean-square reductions
at widths 128, 192, 256, 512 and 1280 on the Radeon 8060S, including
per-channel one-hot checks with zero double-counting. Native trained
Stage1 (512 tensors) and Stage2 (226 tensors) individually matched the
source and the combined IREE-only Stage1→Stage2 replay passed on
2026-10-24 (79.229 ms median, including CPU staging/reupload).

Other translation pipelines must still be tested on their own merits.
`setMatmulOpConfig` derives a preferred subgroup width but may select
lowerings that do not require an explicit subgroup contract; an omitted
subgroup argument is **not by itself evidence of a separate GEMM bug**.
The trained native Stage1→Stage2 proof passed, so there is no currently
reproduced native GEMM correctness failure to upstream from this work.

The IREE 3.13 Vulkan device planner also sets
`VkPhysicalDeviceVulkan13Features.subgroupSizeControl` based on the
physical-device feature and marks `IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL`
when available. Your 8060S reports this feature. Thus the normal
runtime device-creation path already has the required capability;
no separate Vulkan loader shim should be needed for this experiment.

The IREE upstream [issue #24688](https://github.com/iree-org/iree/issues/24688)
already warns that a target subgroup-size configuration differing from the
device's actual subgroup size can cause incorrect results. That issue concerns
default Vulkan target selection. Crucially, an **earlier upstream PR,
[#25021](https://github.com/iree-org/iree/pull/25021)**, already identified
and proposed the same missing `SubgroupReduce` subgroup-size propagation
in October 2026, along with a mixed-width regression. That proposal was
closed without merging. The omission remained in upstream `main` when
checked on 2026-10-24. Its previous discovery must be acknowledged in any
subsequent upstream submission.

## Minimal upstream-compatible patch

[`patches/iree/3.13.0-require-subgroup-size.patch`](../patches/iree/3.13.0-require-subgroup-size.patch)
changes both `SubgroupReduce` call sites from:

```cpp
getSPIRVTranslationInfo(ctx, SPIRVPipeline::SubgroupReduce, workgroupSize)
```

to:

```cpp
getSPIRVTranslationInfo(ctx, SPIRVPipeline::SubgroupReduce,
                        workgroupSize, subgroupSize)
```

The patch also updates **six existing upstream FileCheck assertions** to
verify that the required subgroup size (16, 32, or 64, depending on the
test target) appears in translation information. It does not alter
the original PyTorch graph, checkpoint tensors, or arithmetic.

## Resuming an interrupted Windows build

If Ninja reports exit status **9009** while generating
`runtime/src/iree/vm/bytecode/isa/op_table.h` or
`encoding_table.c`, and Windows says `python3` is unavailable,
the compiler object files are **not** corrupted. Some upstream IREE 3.13
custom commands call literal `python3`, even when CMake was given
`Python3_EXECUTABLE`. Windows may resolve that name to its Microsoft
Store app execution alias rather than the activated Conda interpreter.

The build wrapper now creates a small `python3.cmd` forwarding to the
current Conda `python.exe`, prepends it to `PATH` for its own process,
and verifies the command from `cmd.exe` before resuming Ninja. It
restores the original `PATH` afterwards. No Python install,
Windows Settings change, cache clean, or new IREE checkout is needed.

```powershell
git pull --ff-only
.\scripts\ci\build_and_prove_iree_wave32.ps1
```

Existing successful compilation outputs in the unchanged CMake/Ninja
build tree remain reusable; the script builds the targeted
`IREECompilerPythonModules` dependency graph rather than the full
default `all` target. Normal CMake reconfiguration does not by itself
require wiping compiled objects. For an isolated check of `python3`
resolution without a compiler build, run:

```powershell
.\scripts\ci\build_and_prove_iree_wave32.ps1 -CheckPythonOnly
```

## One-command Windows build and AMD proof

From the CorridorKey checkout with the existing Conda environment activated:

```powershell
git fetch origin amd/checkpoint-environment
git merge --ff-only FETCH_HEAD

.\scripts\ci\build_and_prove_iree_wave32.ps1
```

The script automatically clones **IREE v3.13.0** into a sibling checkout,
applies the version-specific patch (or detects that it was already applied),
initializes the installed Visual Studio C++ environment when necessary,
fetches IREE submodules, builds the native compiler and Python bindings
with CMake/Ninja, and temporarily prepends the new compiler bindings to
`PYTHONPATH`. It then runs the existing **native, unsegmented** reduction
and one-hot tests against the same installed IREE runtime and Turbine.
**It does not pip-install or uninstall anything in the Conda environment.**

Git, CMake (3.26 or newer), Ninja, MSVC C++ Build Tools and the existing
Conda Python environment must be installed. You can launch it from an
ordinary PowerShell session; the script locates MSVC through `vswhere` if
`cl.exe` is not already available. This is a native compiler build,
so adequate disk space and memory are required.

The default proof output is `build/subgroup-wave32-native/report.json`.
Repeated runs preserve previous reports and select an unused numbered
directory. The IREE source/build trees are reused incrementally.

Useful options:

```powershell
# Use more build workers
.\scripts\ci\build_and_prove_iree_wave32.ps1 -Jobs 8

# Reuse an already-built patched compiler without rebuilding
.\scripts\ci\build_and_prove_iree_wave32.ps1 -SkipBuild

# Check only compiler build/import, without executing Vulkan
.\scripts\ci\build_and_prove_iree_wave32.ps1 -SkipProof
```

Before Vulkan execution, the script now runs IREE's original six-case
`config_default_reduction.mlir` fixture through the **just-built `iree-opt`**,
checking that its translation metadata requests the correct 16-, 32-, and
64-lane widths. This does not require the full LLVM testing infrastructure;
the compiler's resulting MLIR is retained as
`iree-wave32-v3.13.0-build/corridorkey-subgroup-config.mlir`.

The script refuses to use a checkout at any revision other than `v3.13.0`,
refuses to consider the compiler patched without both corrected call
sites, verifies that Python resolves the *built* compiler rather than
the installed wheel, and requires explicit subgroup size 32 in generated
IR before executing each GPU reduction. A failed proof leaves its
`report.json` and executable dumps in the selected output directory.

## Verify against the original native reduction

The one-command script above runs this verification automatically.
To perform the native proof directly with an already selected patched
compiler, preserve the earlier failing run and use a new output directory:

```powershell
python scripts/ci/prove_vulkan_reductions.py `
  --widths 128,192,256,512,1280 `
  --operations sum_square,mean_square `
  --target rdna3 --execute-vulkan --one-hot `
  --dump-executables --record-subgroups `
  --require-subgroup-size 32 `
  --output build/subgroup-wave32-native
```

A proper result must meet **both** criteria:

1. The dumped compiler IR contains an explicit requested subgroup size
   of **32** for each of these reductions. The `--require-subgroup-size`
   guard rejects missing or conflicting compiler metadata. The
   `subgroup_size_choices=[32,64]` capability is not sufficient.
2. Every native reduction matches CPU, with every one-hot contribution
   counted **exactly once**. Zero mismatches at the existing tolerance
   are required.

Inspect `build/subgroup-wave32-native/report.json` and save the full
IR/SPIR-V executable dumps. Once native reductions pass, rerun the
original trained Stage2 and Stage1 without segmented reductions,
individually first. Other operator classes must not be assumed fixed.

For an upstream report, include the unpatched/patched compiler revision,
`vulkaninfo` output, both generated executables and the 192-channel
one-hot map. The original IREE compiler test is
`compiler/src/iree/compiler/Codegen/SPIRV/test/config_default_reduction.mlir`.

### Limitations and portability

Do **not** hard-code 32 for all Vulkan devices. The local GPU-validated
patch passes the target's selected subgroup width; this is correct for
the measured RDNA3 device but overconstrains *fixed-width* targets that
may lack subgroup-size-control support. The **separate upstream candidate**
requests an explicit subgroup size only when
`target.getMinSubgroupSize() != target.getMaxSubgroupSize()` and adds
mixed-width static and dynamic regressions, retaining implicit width on
fixed-width devices. The candidate is structurally checked against IREE
3.13.0 but **has not been rebuilt, FileCheck-tested or hardware-proven**.
Do not replace the locally validated patch in the existing IREE build.

The segmented-source paths remain independent correctness controls;
however, both trained stages have now passed with native reductions and
the combined trained replay has also passed.

## Using the patched compiler for trained Stage1 and Stage2

Once `build_and_prove_iree_wave32.ps1` has produced the patched compiler,
the trained proof **does not rebuild IREE, LLVM, or MLIR**:

```powershell
.\scripts\ci\run_trained_with_patched_iree.ps1
```

The default `Auto` mode compiles and proves the original checkpoint's
Stage1→Stage2 computation with **all four reduction choices set to
`native`**: RMS, downsample, fusion and the remaining wide linears.
Unlike the original trained proof's `segmented128` RMS default, this
makes it possible to determine whether the patched IREE compiler fixes
native RMS reductions too. If native verification fails, the command
preserves its consolidated diagnostics and runs a *second, separate* proof
using the source-equivalent segmented128 controls. It reports success only
if an entire Stage1→Stage2 proof passes.

For one mode only:

```powershell
.\scripts\ci\run_trained_with_patched_iree.ps1 -Mode Native
.\scripts\ci\run_trained_with_patched_iree.ps1 -Mode Segmented
```

Output is stored beneath `build/trained-patched-iree/native` and
`build/trained-patched-iree/segmented`; subsequent runs preserve prior
output under numbered directories. Each run emits one self-contained
`corridorkey-vulkan-diagnostics.json` report. Both modes use the built
IREE compiler through an invocation-scoped `PYTHONPATH` and preserve
the Conda packages; the script verifies the loaded compiler path.
The trained proof validates the two neural stages only, not the
foundation networks or a complete video-matting application.

**Do not treat a passing segmented proof as evidence that native GEMM
or RMS correctness has been repaired.** The reduced source-equivalent
path and fully native compilation remain separate acceptance cases.

### Current trained-pipeline acceptance (2026-10-24)

The original checkpoint's two trained stages now pass independently on
native Vulkan and pass in an IREE-only combined replay. The successful
execution report is
`build/trained-patched-iree/native/execution-replay-2/report.json`.
The combined run uses HAL-owned readback and a CPU NHWC→NCHW conversion,
then reuploads to Vulkan; the measured **79.229 ms median** includes this
staging and is not a GPU-only latency metric. Frozen foundation networks,
full video inference and a GPU-resident bridge remain outside this proof.

The runtime readback fix is local to CorridorKey and bypasses another
independent upstream defect, where `DeviceArray.to_host()` tries to
import an unaligned NumPy pointer despite Vulkan's 4096-byte host-pointer
alignment requirement on this device. Its upstream investigation and
Windows CMake `python3` problem are covered separately in
[`docs/iree-upstream-handoff.md`](iree-upstream-handoff.md).

## Upstream patch submission criteria

The first upstream patch should remain surgical: in the static and
dynamic `SPIRVPipeline::SubgroupReduce` paths, pass the exact
`subgroupSize` used to specialize the lowering into
`getSPIRVTranslationInfo`. Do not silently force wave32 on every GPU,
nor change the reduction arithmetic. The existing patch updates six
`FileCheck` assertions for subgroup selection at 16, 32 and 64.
A separate **upstream-only draft patch** at
[`patches/iree/3.13.0-upstream-portable-subgroup-size.patch`](../patches/iree/3.13.0-upstream-portable-subgroup-size.patch)
contains the conditional-size refinement and new
`subgroup_size_choices = [32, 64]` test cases for both static and dynamic
reductions. Existing fixed-width cases retain their implicit-width
assertions. It is an **unvalidated submission candidate** and is not
used by `build_and_prove_iree_wave32.ps1`. Its structural applicability
has been checked against the pristine IREE 3.13.0 source. Run upstream
FileCheck/lit tests and validate it on fixed-width and variable-width
Vulkan targets before submission. Attribute the prior overlapping
proposal in upstream PR #25021.

The v3.13.0 native reduction failures and the patched Radeon numerical
success have already been recorded. Before filing an upstream PR, package
the original and passing reduction reports, compiler metadata and SPIR-V
dumps, physical GPU properties, and the passing trained-stage reports.
These are user-generated local artifacts, not yet publicly attached.
Do not claim a separate GEMM bug without an independent failing case.
Never assume that the preferred architecture subgroup is universally
required for layout-neutral pipelines.

Once a source patch passes on the user's physical GPU, producing a
versioned `iree-base-compiler` wheel on Windows and Linux is a useful
way to avoid repetitive local LLVM builds while an upstream release
is pending. Treat this as a *packaging job*, not a reason to modify
the original source graph or to pin the project permanently to v3.13.0.

