#Requires -Version 5.1
<#
.SYNOPSIS
  Run the original trained CorridorKey stages with the locally patched IREE compiler.
.DESCRIPTION
  Reuses the built IREE compiler from build_and_prove_iree_wave32.ps1.
  Never installs packages, modifies checkpoints or rebuilds LLVM/MLIR.
  Native mode has no segmented lowering. Auto also tries source-faithful
  segmented fallbacks if native verification fails.
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
    [string]$Mode = "Auto",
    [string]$Target = "rdna3"
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
        $candidate = Join-Path $OutputRoot $pass.ToLowerInvariant()
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
        Write-Host "[trained wave32] $pass: $out"
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
