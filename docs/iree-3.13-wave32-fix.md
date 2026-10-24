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
mistakes, so do not claim all trained inference corrected yet.

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

## Build the patched IREE 3.13 compiler (Windows, PowerShell)

Use a **separate checkout and build directory**. Do not update Turbine or
the 3.13 runtime during this A/B comparison. From the CorridorKey repo
directory in the existing Conda environment:

```powershell
conda activate py313

git clone --branch v3.13.0 --depth 1 https://github.com/iree-org/iree.git ..\iree-wave32
git -C ..\iree-wave32 apply --check patches\iree\3.13.0-require-subgroup-size.patch
git -C ..\iree-wave32 apply patches\iree\3.13.0-require-subgroup-size.patch
git -C ..\iree-wave32 submodule update --init --recursive

cmake -S ..\iree-wave32 -B ..\iree-wave32-build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DIREE_BUILD_PYTHON_BINDINGS=ON `
  -DIREE_BUILD_TESTS=ON `
  -DPython3_EXECUTABLE="$((Get-Command python).Source)"
cmake --build ..\iree-wave32-build --parallel

# Install the compiler bindings from this new build into the active Conda
# environment. The Vulkan runtime still comes from iree-base-runtime 3.13.
$env:CMAKE_INSTALL_MODE = "ABS_SYMLINK"
python -m pip install --no-build-isolation -e ..\iree-wave32-build\compiler
Remove-Item Env:\CMAKE_INSTALL_MODE

python -c "import iree.compiler; print(iree.compiler.__file__)"
python -c "import iree.runtime; print(iree.runtime.__file__)"
```

Requires MSVC Build Tools / Developer PowerShell, Ninja, CMake, and the
build prerequisites documented in
[IREE's build guide](https://iree.dev/building-from-source/getting-started/).
This is a full native compiler rebuild, not a lightweight Python
monkeypatch. The editable installation step and dependency resolution
may vary by the checked-out version. Ensure the Python compiler package
actually resolves to the patched build; if not, use the generated
`..\iree-wave32-build\.env.ps1` to set the binding paths as documented
by IREE. Do not interpret successful compilation alone as a GPU fix.

On Linux use the equivalent CMake/Ninja build with a Conda or virtual
environment and source `../iree-wave32-build/.env` if required.

## Verify against the original native reduction

Preserve the previous failing result in `build/subgroup-investigation`.
Run a **new** result directory with the patched compiler:

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
