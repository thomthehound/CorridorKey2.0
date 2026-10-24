#Requires -Version 5.1
<#
.SYNOPSIS
  Build patched IREE v3.13.0 and prove native Vulkan wave32 reductions.
.DESCRIPTION
  Keeps the IREE source/build separate from CorridorKey, applies the
  version-specific upstream patch idempotently, builds IREE with MSVC/Ninja,
  and uses the freshly built compiler through PYTHONPATH only for this run.
  Does not pip-install, uninstall, or replace the active Conda environment.
  Saves a distinct proof report and complete Vulkan executable dumps.
.EXAMPLE
  .\scripts\ci\build_and_prove_iree_wave32.ps1
.EXAMPLE
  .\scripts\ci\build_and_prove_iree_wave32.ps1 -Jobs 8 -SkipBuild
#>
[CmdletBinding()]
param(
    [string]$SourceDir = "",
    [string]$BuildDir = "",
    [string]$OutputDir = "",
    [ValidateRange(1, 64)]
    [int]$Jobs = 4,
    [ValidateSet("rdna1", "rdna2", "rdna3", "rdna4")]
    [string]$Target = "rdna3",
    [switch]$SkipBuild,
    [switch]$SkipProof
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )
    Write-Host ("[wave32] " + $Executable + " " + ($Arguments -join " "))
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable exited with status $LASTEXITCODE"
    }
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing '$Name' on PATH. Use an MSVC Developer PowerShell with Git, CMake, Ninja and your activated Conda Python."
    }
}

function Initialize-Msvc {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $installer)) {
        throw "MSVC tools are not in PATH and vswhere.exe was not found. Install Visual Studio C++ Build Tools."
    }
    $installation = (& $installer -latest -products "*" `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or -not $installation) {
        throw "A Visual Studio installation with C++ build tools is required."
    }
    $vcvars = Join-Path $installation "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) {
        throw "MSVC environment initializer not found: $vcvars"
    }
    Write-Host "[wave32] Loading MSVC x64 environment from $vcvars"
    $lines = & cmd.exe /d /c "call `"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw "Could not initialize MSVC environment" }
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)
$parent = Split-Path -Parent $repoRoot
if (-not $SourceDir) { $SourceDir = Join-Path $parent "iree-wave32-v3.13.0" }
if (-not $BuildDir) { $BuildDir = Join-Path $parent "iree-wave32-v3.13.0-build" }
if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot "build\subgroup-wave32-native" }
$SourceDir = [IO.Path]::GetFullPath($SourceDir)
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$patch = Join-Path $repoRoot "patches\iree\3.13.0-require-subgroup-size.patch"
$proof = Join-Path $repoRoot "scripts\ci\prove_vulkan_reductions.py"
$originalPythonPath = $env:PYTHONPATH

if (-not (Test-Path -LiteralPath $patch -PathType Leaf)) {
    throw "CorridorKey's IREE patch is missing: $patch"
}
if ($SourceDir -eq $repoRoot -or $BuildDir -eq $repoRoot -or $SourceDir -eq $BuildDir) {
    throw "The IREE source, build and CorridorKey directories must be distinct."
}

try {
    foreach ($name in @("git", "cmake", "ninja", "python")) {
        Require-Command $name
    }
    if (-not $SkipBuild) {
        # Initialize Visual Studio tools automatically in an ordinary Conda shell.
        Initialize-Msvc
    }

    # The proof is an A/B test against the same installed runtime and Turbine.
    # Importing this code does not alter packages in the Conda environment.
    Invoke-Checked "python" @("-c", (
        "import importlib.metadata as m; " +
        "a=m.version('iree-base-runtime'); " +
        "b=m.version('iree-turbine'); " +
        "assert a=='3.13.0', f'Expected IREE runtime 3.13.0, got {a}'; " +
        "print('IREE runtime:',a,'Turbine:',b)"
    ))

    if (-not (Test-Path -LiteralPath $SourceDir)) {
        if ($SkipBuild) {
            throw "IREE source not found at $SourceDir; remove -SkipBuild."
        }
        Invoke-Checked "git" @(
            "clone", "--depth", "1", "--branch", "v3.13.0",
            "https://github.com/iree-org/iree.git", $SourceDir
        )
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SourceDir ".git"))) {
        throw "The IREE source directory is not a Git checkout: $SourceDir"
    }
    $head = (& git -C $SourceDir rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Failed to read IREE HEAD" }
    $tag = (& git -C $SourceDir rev-list -n 1 "v3.13.0").Trim()
    if ($LASTEXITCODE -ne 0 -or $head -ne $tag) {
        throw "IREE checkout is not at v3.13.0. Use a fresh -SourceDir; no reset was performed."
    }

    $alreadyPatched = $false
    try {
        & git -C $SourceDir apply --reverse --check $patch 2>$null
        $alreadyPatched = ($LASTEXITCODE -eq 0)
    } catch {
        # An unapplied patch is an expected state on the first invocation.
        $alreadyPatched = $false
    }
    if ($alreadyPatched) {
        Write-Host "[wave32] The exact IREE patch is already applied."
    } else {
        if ($SkipBuild) {
            throw "IREE patch is not applied; remove -SkipBuild."
        }
        Invoke-Checked "git" @("-C", $SourceDir, "apply", "--check", $patch)
        Invoke-Checked "git" @("-C", $SourceDir, "apply", $patch)
    }

    $expectedSource = Join-Path $SourceDir "compiler\src\iree\compiler\Codegen\SPIRV\KernelConfig.cpp"
    $sourceText = [IO.File]::ReadAllText($expectedSource)
    if (($sourceText.Split([string[]]@("workgroupSize, subgroupSize)))) {
        # Validated below using exact syntax instead of counts on whitespace.
    }
    $subgroupCalls = [regex]::Matches(
        $sourceText,
        "getSPIRVTranslationInfo\(ctx,\s*SPIRVPipeline::SubgroupReduce,\s*workgroupSize,\s*subgroupSize\)"
    )
    if ($subgroupCalls.Count -ne 2) {
        throw "Expected two patched SubgroupReduce calls, found $($subgroupCalls.Count)."
    }

    if (-not $SkipBuild) {
        Invoke-Checked "git" @(
            "-C", $SourceDir, "submodule", "update", "--init", "--recursive",
            "--depth", "1"
        )
        $python = (Get-Command python -ErrorAction Stop).Source
        Invoke-Checked "cmake" @(
            "-S", $SourceDir, "-B", $BuildDir, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DIREE_BUILD_COMPILER=ON",
            "-DIREE_BUILD_PYTHON_BINDINGS=ON",
            "-DIREE_BUILD_TESTS=OFF",
            "-DPython3_EXECUTABLE=$python"
        )
        Invoke-Checked "cmake" @("--build", $BuildDir, "--parallel", "$Jobs")
    }

    $compilerBindings = Join-Path $BuildDir "compiler\bindings\python"
    if (-not (Test-Path -LiteralPath (Join-Path $compilerBindings "iree\compiler"))) {
        throw "Built compiler Python bindings are missing: $compilerBindings"
    }
    if ($originalPythonPath) {
        $env:PYTHONPATH = $compilerBindings + [IO.Path]::PathSeparator + $originalPythonPath
    } else {
        $env:PYTHONPATH = $compilerBindings
    }

    # Abort if Python silently imports the original installed compiler.
    $env:CORRIDORKEY_EXPECTED_COMPILER = $compilerBindings
    try {
        Invoke-Checked "python" @("-c", (
            "import iree.compiler as c, iree.runtime as r, pathlib, os; " +
            "expected=pathlib.Path(os.environ['CORRIDORKEY_EXPECTED_COMPILER']).resolve(); " +
            "paths=[pathlib.Path(p).resolve() for p in c.__path__]; " +
            "assert any(p.is_relative_to(expected) for p in paths), " +
            "f'Patched compiler not imported: {paths}'; " +
            "from iree.compiler import ir; " +
            "print('Patched compiler:',paths); " +
            "print('Runtime:',list(r.__path__))"
        ))
    } finally {
        Remove-Item Env:\CORRIDORKEY_EXPECTED_COMPILER -ErrorAction SilentlyContinue
    }

    if ($SkipProof) {
        Write-Host "[wave32] Build/import verification passed; proof skipped."
        return
    }

    if ((Test-Path -LiteralPath $OutputDir) -and
        @(Get-ChildItem -LiteralPath $OutputDir -Force).Count -gt 0) {
        throw "Output directory is not empty: $OutputDir. Use -OutputDir to preserve prior evidence."
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    Write-Host "[wave32] Running original native reductions with a patched IREE compiler."
    Write-Host "[wave32] Device results, one-hot multiplicity and executable dumps: $OutputDir"
    Invoke-Checked "python" @(
        $proof,
        "--widths", "128,192,256,512,1280",
        "--operations", "sum_square,mean_square",
        "--target", $Target,
        "--execute-vulkan", "--one-hot", "--record-subgroups",
        "--dump-executables", "--require-subgroup-size", "32",
        "--output", $OutputDir
    )

    $reportPath = Join-Path $OutputDir "report.json"
    $summary = @'
import json
import pathlib
import sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in report["cases"]:
    hot = case.get("one_hot", {})
    print(f"[wave32] {case['name']}: mismatched={case.get('mismatched')}; "
          f"one-hot once={hot.get('roughly_once')}/{case['width']}")
    if case.get("numerical_match") is not True:
        raise SystemExit(f"Uncorrected native Vulkan result: {case['name']}")
    if hot.get("roughly_once") != case["width"]:
        raise SystemExit(f"Bad one-hot multiplicity: {case['name']}")
print("[wave32] Native reduction correctness proof passed")
'@
    # A script file avoids shell quoting differences for multi-line Python.
    $summaryFile = Join-Path $OutputDir "check_native_reductions.py"
    [IO.File]::WriteAllText($summaryFile, $summary)
    Invoke-Checked "python" @($summaryFile, $reportPath)
    Write-Host "[wave32] Report: $reportPath"
} finally {
    $env:PYTHONPATH = $originalPythonPath
}
) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
    Require-Command "cl.exe"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$parent = Split-Path -Parent $repoRoot
if (-not $SourceDir) { $SourceDir = Join-Path $parent "iree-wave32-v3.13.0" }
if (-not $BuildDir) { $BuildDir = Join-Path $parent "iree-wave32-v3.13.0-build" }
if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot "build\subgroup-wave32-native" }
$SourceDir = [IO.Path]::GetFullPath($SourceDir)
$BuildDir = [IO.Path]::GetFullPath($BuildDir)
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$patch = Join-Path $repoRoot "patches\iree\3.13.0-require-subgroup-size.patch"
$proof = Join-Path $repoRoot "scripts\ci\prove_vulkan_reductions.py"
$originalPythonPath = $env:PYTHONPATH

if (-not (Test-Path -LiteralPath $patch -PathType Leaf)) {
    throw "CorridorKey's IREE patch is missing: $patch"
}
if ($SourceDir -eq $repoRoot -or $BuildDir -eq $repoRoot -or $SourceDir -eq $BuildDir) {
    throw "The IREE source, build and CorridorKey directories must be distinct."
}

try {
    foreach ($name in @("git", "cmake", "ninja", "python")) {
        Require-Command $name
    }
    if (-not $SkipBuild) {
        # cl.exe is available from an MSVC Developer PowerShell; do not try
        # building with a random non-developer command prompt.
        Require-Command "cl.exe"
    }

    # The proof is an A/B test against the same installed runtime and Turbine.
    # Importing this code does not alter packages in the Conda environment.
    Invoke-Checked "python" @("-c", (
        "import importlib.metadata as m; " +
        "a=m.version('iree-base-runtime'); " +
        "b=m.version('iree-turbine'); " +
        "assert a=='3.13.0', f'Expected IREE runtime 3.13.0, got {a}'; " +
        "print('IREE runtime:',a,'Turbine:',b)"
    ))

    if (-not (Test-Path -LiteralPath $SourceDir)) {
        if ($SkipBuild) {
            throw "IREE source not found at $SourceDir; remove -SkipBuild."
        }
        Invoke-Checked "git" @(
            "clone", "--depth", "1", "--branch", "v3.13.0",
            "https://github.com/iree-org/iree.git", $SourceDir
        )
    }
    if (-not (Test-Path -LiteralPath (Join-Path $SourceDir ".git"))) {
        throw "The IREE source directory is not a Git checkout: $SourceDir"
    }
    $head = (& git -C $SourceDir rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Failed to read IREE HEAD" }
    $tag = (& git -C $SourceDir rev-list -n 1 "v3.13.0").Trim()
    if ($LASTEXITCODE -ne 0 -or $head -ne $tag) {
        throw "IREE checkout is not at v3.13.0. Use a fresh -SourceDir; no reset was performed."
    }

    & git -C $SourceDir apply --reverse --check $patch 2>$null
    $alreadyPatched = $LASTEXITCODE -eq 0
    if ($alreadyPatched) {
        Write-Host "[wave32] The exact IREE patch is already applied."
    } else {
        if ($SkipBuild) {
            throw "IREE patch is not applied; remove -SkipBuild."
        }
        Invoke-Checked "git" @("-C", $SourceDir, "apply", "--check", $patch)
        Invoke-Checked "git" @("-C", $SourceDir, "apply", $patch)
    }

    $expectedSource = Join-Path $SourceDir "compiler\src\iree\compiler\Codegen\SPIRV\KernelConfig.cpp"
    $sourceText = [IO.File]::ReadAllText($expectedSource)
    if (($sourceText.Split([string[]]@("workgroupSize, subgroupSize)))) {
        # Validated below using exact syntax instead of counts on whitespace.
    }
    $subgroupCalls = [regex]::Matches(
        $sourceText,
        "getSPIRVTranslationInfo\(ctx,\s*SPIRVPipeline::SubgroupReduce,\s*workgroupSize,\s*subgroupSize\)"
    )
    if ($subgroupCalls.Count -ne 2) {
        throw "Expected two patched SubgroupReduce calls, found $($subgroupCalls.Count)."
    }

    if (-not $SkipBuild) {
        Invoke-Checked "git" @(
            "-C", $SourceDir, "submodule", "update", "--init", "--recursive",
            "--depth", "1"
        )
        $python = (Get-Command python -ErrorAction Stop).Source
        Invoke-Checked "cmake" @(
            "-S", $SourceDir, "-B", $BuildDir, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DIREE_BUILD_COMPILER=ON",
            "-DIREE_BUILD_PYTHON_BINDINGS=ON",
            "-DIREE_BUILD_TESTS=OFF",
            "-DPython3_EXECUTABLE=$python"
        )
        Invoke-Checked "cmake" @("--build", $BuildDir, "--parallel", "$Jobs")
    }

    $compilerBindings = Join-Path $BuildDir "compiler\bindings\python"
    if (-not (Test-Path -LiteralPath (Join-Path $compilerBindings "iree\compiler"))) {
        throw "Built compiler Python bindings are missing: $compilerBindings"
    }
    if ($originalPythonPath) {
        $env:PYTHONPATH = $compilerBindings + [IO.Path]::PathSeparator + $originalPythonPath
    } else {
        $env:PYTHONPATH = $compilerBindings
    }

    # Abort if Python silently imports the original installed compiler.
    $env:CORRIDORKEY_EXPECTED_COMPILER = $compilerBindings
    try {
        Invoke-Checked "python" @("-c", (
            "import iree.compiler as c, iree.runtime as r, pathlib, os; " +
            "expected=pathlib.Path(os.environ['CORRIDORKEY_EXPECTED_COMPILER']).resolve(); " +
            "paths=[pathlib.Path(p).resolve() for p in c.__path__]; " +
            "assert any(p.is_relative_to(expected) for p in paths), " +
            "f'Patched compiler not imported: {paths}'; " +
            "from iree.compiler import ir; " +
            "print('Patched compiler:',paths); " +
            "print('Runtime:',list(r.__path__))"
        ))
    } finally {
        Remove-Item Env:\CORRIDORKEY_EXPECTED_COMPILER -ErrorAction SilentlyContinue
    }

    if ($SkipProof) {
        Write-Host "[wave32] Build/import verification passed; proof skipped."
        return
    }

    if ((Test-Path -LiteralPath $OutputDir) -and
        @(Get-ChildItem -LiteralPath $OutputDir -Force).Count -gt 0) {
        throw "Output directory is not empty: $OutputDir. Use -OutputDir to preserve prior evidence."
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    Write-Host "[wave32] Running original native reductions with a patched IREE compiler."
    Write-Host "[wave32] Device results, one-hot multiplicity and executable dumps: $OutputDir"
    Invoke-Checked "python" @(
        $proof,
        "--widths", "128,192,256,512,1280",
        "--operations", "sum_square,mean_square",
        "--target", $Target,
        "--execute-vulkan", "--one-hot", "--record-subgroups",
        "--dump-executables", "--require-subgroup-size", "32",
        "--output", $OutputDir
    )

    $reportPath = Join-Path $OutputDir "report.json"
    $summary = @'
import json
import pathlib
import sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in report["cases"]:
    hot = case.get("one_hot", {})
    print(f"[wave32] {case['name']}: mismatched={case.get('mismatched')}; "
          f"one-hot once={hot.get('roughly_once')}/{case['width']}")
    if case.get("numerical_match") is not True:
        raise SystemExit(f"Uncorrected native Vulkan result: {case['name']}")
    if hot.get("roughly_once") != case["width"]:
        raise SystemExit(f"Bad one-hot multiplicity: {case['name']}")
print("[wave32] Native reduction correctness proof passed")
'@
    # A script file avoids shell quoting differences for multi-line Python.
    $summaryFile = Join-Path $OutputDir "check_native_reductions.py"
    [IO.File]::WriteAllText($summaryFile, $summary)
    Invoke-Checked "python" @($summaryFile, $reportPath)
    Write-Host "[wave32] Report: $reportPath"
} finally {
    $env:PYTHONPATH = $originalPythonPath
}
