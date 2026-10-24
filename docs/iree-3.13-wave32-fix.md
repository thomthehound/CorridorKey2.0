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

This is an identified **compiler-to-Vulkan subgroup-contract omission**.
The 2× result and missing subgroup metadata are consistent with it;
device execution of the patched compiler is still needed to demonstrate
that this is the complete numerical root cause. Matrix projections
using other lowering pipelines may have separate subgroup-contract
mistakes, so do not claim all trained inference corrected yet. In
particular, `setMatmulOpConfig` within the same IREE file derives a
preferred subgroup width and passes it to matmul tiling helpers, but the
`MatmulPromoteVectorize` translation call explicitly supplies
`std::nullopt`, while the `BaseVectorize` translation call omits the
subgroup argument. This is a **separate audit target** for Stage1's
trained matrix-projection failures. It is not modified by the reduction
patch: keep those questions independent until a controlled native GEMM
A/B establishes whether the same correction is required.

The IREE 3.13 Vulkan device planner also sets
`VkPhysicalDeviceVulkan13Features.subgroupSizeControl` based on the
physical-device feature and marks `IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL`
when available. Your 8060S reports this feature. Thus the normal
runtime device-creation path already has the required capability;
no separate Vulkan loader shim should be needed for this experiment.

The IREE upstream [issue #24688](https://github.com/iree-org/iree/issues/24688)
already warns that a target subgroup-size configuration differing from the
device's actual subgroup size can cause incorrect results. That issue concerns
default Vulkan target selection; the **specific missing subgroup-size
propagation at the two `SubgroupReduce` call sites** is independently
identified here and should be reported to IREE with the patch and physical
reproducer after verification. The omissions are still present in the
current upstream `main` source as inspected on 2026-10-24.

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

Do **not** hard-code 32 for all Vulkan devices. The source fix propagates
each target's selected subgroup width rather than imposing wave32 on
hardware that may only support a different size. IREE's Vulkan runtime
validates the requested width against
`VkPhysicalDeviceSubgroupSizeControlProperties`, and unsupported
pipeline creation must fail rather than silently execute a different
configuration. The original segmented-source paths remain independent
correctness controls until both Stage1 and Stage2 pass with the patched
native compiler.
