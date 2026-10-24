#Requires -Version 5.1
<#
.SYNOPSIS
  Run the original trained CorridorKey stages with the locally patched IREE compiler.
.DESCRIPTION
  Reuses the built IREE compiler from build_and_prove_iree_wave32.ps1.
  Never installs packages, modifies checkpoints or rebuilds LLVM/MLIR.
  Native (the default) uses original reductions throughout. Segmented
  is an explicit diagnostic control. Auto may attempt that control after
  a failed native proof but must not be treated as native acceptance.
.EXAMPLE
  .\scripts\ci\run_trained_with_patched_iree.ps1
.EXAMPLE
  .\scripts\ci\run_trained_with_patched_iree.ps1 -Mode Native
#>
[CmdletBinding()]
param(
    [string]$BuildDir = "",
    [string]$Checkpoint = "",
    [string]$OutputRoot = "",
    [ValidateSet("Native", "Segmented", "Auto")]
    [string]$Mode = "Native",
    [string]$Target = "rdna3",
    [switch]$ReplayExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $BuildDir) {
    $BuildDir = Join-Path (Split-Path -Parent $repoRoot) "iree-wave32-v3.13.0-build"
}
if (-not $Checkpoint) {
    $Checkpoint = Join-Path $repoRoot "weights\corridorkey_v2.safetensors"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $repoRoot "build\trained-patched-iree"
}
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$Checkpoint = [IO.Path]::GetFullPath($Checkpoint)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if ($ReplayExisting) {
    if ($Mode -eq "Auto") {
        throw "Replay requires -Mode Native or -Mode Segmented, identifying the existing compiled proof."
    }
    $sourceRoot = Join-Path $OutputRoot ($Mode.ToLowerInvariant())
    $stage1Dir = Join-Path $sourceRoot "stage1"
    $stage2Dir = Join-Path $sourceRoot "stage2"
    $stage1Vmfb = Join-Path $stage1Dir "stage1.vmfb"
    $stage2Vmfb = Join-Path $stage2Dir "stage2.vmfb"
    foreach ($required in @(
        $stage1Vmfb, $stage2Vmfb,
        (Join-Path $stage1Dir "input_0.npy"),
        (Join-Path $stage1Dir "source_0.npy"),
        (Join-Path $stage1Dir "source_1.npy"),
        (Join-Path $stage2Dir "source_0.npy")
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Required compiled artifact or reference not found: $required"
        }
    }
    $candidate = Join-Path $sourceRoot "execution-replay"
    $out = $candidate
    $index = 2
    while (Test-Path -LiteralPath $out) {
        $out = "$candidate-$index"
        $index++
    }
    Write-Host "[trained wave32] Replaying existing $Mode VMFBs without recompilation."
    & python (Join-Path $repoRoot "scripts\ci\execute_compiled_stages.py") `
        --stage1-vmfb $stage1Vmfb --stage2-vmfb $stage2Vmfb `
        --inputs $stage1Dir --stage1-reference $stage1Dir `
        --stage2-reference $stage2Dir --runs 3 --warmup 1 --output $out
    if ($LASTEXITCODE -ne 0) {
        throw "Compiled $Mode replay failed; report: $(Join-Path $out 'report.json')"
    }
    Write-Host "[trained wave32] Compiled $Mode replay passed."
    Write-Host "[trained wave32] Report: $(Join-Path $out 'report.json')"
    return
}

$compilerBindings = Join-Path $BuildDir "compiler\bindings\python"
$pipeline = Join-Path $repoRoot "scripts\ci\prove_trained_pipeline.py"

if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Checkpoint not found: $Checkpoint"
}
if (-not (Test-Path -LiteralPath (Join-Path $compilerBindings "iree\compiler"))) {
    throw "Patched compiler build not found: $compilerBindings. Run build_and_prove_iree_wave32.ps1 first."
}

$originalPythonPath = $env:PYTHONPATH
try {
    if ($originalPythonPath) {
        $env:PYTHONPATH = $compilerBindings + [IO.Path]::PathSeparator + $originalPythonPath
    } else {
        $env:PYTHONPATH = $compilerBindings
    }
    $env:CORRIDORKEY_EXPECTED_COMPILER = $compilerBindings
    try {
        $validation = @'
import os
import pathlib
import importlib.metadata as md
import iree.compiler as compiler
from iree.compiler import ir
expected = pathlib.Path(os.environ["CORRIDORKEY_EXPECTED_COMPILER"]).resolve()
loaded = [pathlib.Path(p).resolve() for p in compiler.__path__]
if not any(p.is_relative_to(expected) for p in loaded):
    raise RuntimeError(f"Patched compiler not loaded: {loaded}")
runtime = md.version("iree-base-runtime")
if runtime != "3.13.0":
    raise RuntimeError(f"IREE runtime version mismatch: {runtime}")
print("[trained wave32] Built compiler:", loaded, flush=True)
print("[trained wave32] Runtime version:", runtime, flush=True)
'@
        & python -c $validation
        if ($LASTEXITCODE -ne 0) {
            throw "Patched compiler import or version verification failed"
        }
    } finally {
        Remove-Item Env:\CORRIDORKEY_EXPECTED_COMPILER -ErrorAction SilentlyContinue
    }

    $modes = if ($Mode -eq "Auto") { @("Native", "Segmented") } else { @($Mode) }
    foreach ($pass in $modes) {
        $candidate = Join-Path $OutputRoot ($pass.ToLowerInvariant())
        $out = $candidate
        $index = 2
        while ((Test-Path -LiteralPath $out) -and
               @(Get-ChildItem -LiteralPath $out -Force).Count -gt 0) {
            $out = "$candidate-$index"
            $index++
        }
        $options = @(
            $pipeline,
            $Checkpoint,
            "--height", "64", "--width", "64",
            "--target", $Target,
            "--rms-reduction", "native",
            "--down-reduction", "native",
            "--fuse-reduction", "native",
            "--linear-reduction", "native",
            "--output", $out
        )
        # Auto is a fast correctness smoke test first. A failed native
        # compile should not immediately recompile the same graph with
        # instrumentation before the fallback is attempted.
        if ($Mode -eq "Auto" -and $pass -eq "Native") {
            $options += "--no-stage1-diagnostic"
        }
        if ($pass -eq "Segmented") {
            $options = @(
                $pipeline,
                $Checkpoint,
                "--height", "64", "--width", "64",
                "--target", $Target,
                "--rms-reduction", "segmented128",
                "--down-reduction", "segmented128",
                "--fuse-reduction", "segmented128",
                "--linear-reduction", "segmented128",
                "--output", $out
            )
        }
        Write-Host "[trained wave32] ${pass}: $out"
        & python @options
        $result = $LASTEXITCODE
        Write-Host "[trained wave32] $pass exit status: $result"
        Write-Host "[trained wave32] Share report: $(Join-Path $out 'corridorkey-vulkan-diagnostics.json')"
        if ($result -eq 0) {
            Write-Host "[trained wave32] Trained Stage1 -> Stage2 passed using $pass math"
            return
        }
        if ($pass -eq "Native" -and $Mode -eq "Auto") {
            Write-Host "[trained wave32] Native proof failed; trying source-faithful segmented fallback."
        } else {
            throw "$pass trained-pipeline proof failed; diagnostics preserved in $out"
        }
    }
} finally {
    $env:PYTHONPATH = $originalPythonPath
}
