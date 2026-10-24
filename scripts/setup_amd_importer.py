#!/usr/bin/env python3
"""Install host-side dependencies for AMD-native CorridorKey model validation.

This installs CPU PyTorch for checkpoint/reference execution and optionally
the IREE Vulkan importer/compiler. It does not install CUDA or ROCm.
The --ci option selects reproducible compiler dependencies for CI.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
COMMON = (
    "numpy>=1.26",
    "safetensors>=0.6",
    "huggingface-hub>=0.36",
)


def invoke(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("checkpoint", "vulkan"), default="checkpoint",
        help="checkpoint: source CPU proof and model download; "
             "vulkan: also install the IREE compiler/runtime",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print installation commands without running them",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="Use the pinned CI toolchain rather than compatible installed "
             "versions; local use does not require the CI Python version",
    )
    args = parser.parse_args()

    if args.ci and sys.version_info[:2] != (3, 12):
        parser.error("The reproducible CI toolchain uses Python 3.12")

    torch_spec = "torch==2.9.1" if args.ci else "torch>=2.9"
    common = (
        ("numpy>=1.26,<2", "safetensors>=0.6,<1", "huggingface-hub>=0.36")
        if args.ci else COMMON
    )
    compiler = "iree-turbine==3.9.0" if args.ci else "iree-turbine>=3.9"

    pip = [sys.executable, "-m", "pip", "install"]
    invoke(pip + [torch_spec, "--index-url", TORCH_CPU_INDEX],
           dry_run=args.dry_run)
    invoke(pip + list(common), dry_run=args.dry_run)
    if args.profile == "vulkan":
        invoke(pip + [compiler], dry_run=args.dry_run)

    if not args.dry_run:
        invoke([sys.executable, "-m", "pip", "check"], dry_run=False)
        modules = "import torch, numpy, safetensors, huggingface_hub"
        if args.profile == "vulkan":
            modules += "; import iree.turbine.aot, iree.runtime"
        invoke([sys.executable, "-c", modules], dry_run=False)
        print(f"{args.profile} dependencies verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
