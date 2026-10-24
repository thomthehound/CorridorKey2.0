#!/usr/bin/env python3
"""Verify IREE v3.13.0's original subgroup-reduction configuration fixture.

This exercises the real iree-opt codegen pass, without relying on a separate
LLVM FileCheck build. It is independent of a GPU and uses no model tensors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys

SUBGROUP_TRANSLATION = re.compile(
    r"#iree_codegen\.translation_info<"
    r"\s*pipeline\s*=\s*#iree_gpu\.spirv_pipeline<SubgroupReduce>"
    r"\s+workgroup_size\s*=\s*\[\s*(\d+)\s*,\s*1\s*,\s*1\s*\]"
    r"\s+subgroup_size\s*=\s*(\d+)"
)
# Expected sizes for six translated subgroup-reduction configurations in
# upstream compiler/src/iree/compiler/Codegen/SPIRV/test/
# config_default_reduction.mlir, evaluated for their original targets.
EXPECTED_CASES = Counter({
    (128, 16): 1,
    (64, 64): 2,
    (64, 32): 1,
    (512, 32): 1,
    (16, 16): 1,
})


def verify_output(text: str) -> Counter:
    found = Counter(
        (int(workgroup), int(subgroup))
        for workgroup, subgroup in SUBGROUP_TRANSLATION.findall(text)
    )
    for config, expected_count in EXPECTED_CASES.items():
        if found[config] < expected_count:
            raise AssertionError(
                f"Expected at least {expected_count} subgroup-reduction "
                f"translations with workgroup/subgroup sizes {config}, "
                f"got {found}; inspect the compiler's emitted MLIR"
            )
    return found


def self_test() -> None:
    def attribute(workgroup: int, subgroup: int) -> str:
        return (
            "#iree_codegen.translation_info<"
            "pipeline = #iree_gpu.spirv_pipeline<SubgroupReduce> "
            f"workgroup_size = [{workgroup}, 1, 1] "
            f"subgroup_size = {subgroup}>"
        )
    output = "\n".join(
        attribute(workgroup, subgroup)
        for (workgroup, subgroup), count in EXPECTED_CASES.items()
        for _ in range(count)
    )
    assert verify_output(output) == EXPECTED_CASES
    try:
        verify_output(output.replace("subgroup_size = 32", "subgroup_size = 0"))
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing wave32 export was not caught")
    try:
        verify_output(output.replace("subgroup_size = 16", "subgroup_size = 32"))
    except AssertionError:
        pass
    else:
        raise AssertionError("Mismatched subgroup target was not caught")
    try:
        verify_output(output.replace("workgroup_size = [128", "workgroup_size = [256"))
    except AssertionError:
        pass
    else:
        raise AssertionError("Mismatched workgroup shape was not caught")
    print("[IREE codegen] fixture verifier self-test passed", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-opt", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.iree_opt or not args.source or not args.output:
        parser.error("--iree-opt, --source, and --output are required")
    if not args.iree_opt.is_file():
        parser.error(f"Built iree-opt missing: {args.iree_opt}")
    if not args.source.is_file():
        parser.error(f"Upstream reduction test source missing: {args.source}")
    result = subprocess.run(
        [
            str(args.iree_opt), "--split-input-file",
            "--pass-pipeline=builtin.module("
            "iree-spirv-select-lowering-strategy-pass)",
            str(args.source),
        ],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"iree-opt exited with status {result.returncode}")
    found = verify_output(result.stdout)
    print(
        f"[IREE codegen] Source fixture subgroup widths: {dict(found)}; "
        f"MLIR: {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
