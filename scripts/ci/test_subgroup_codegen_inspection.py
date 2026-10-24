#!/usr/bin/env python3
"""Pure Python regression for inspecting IREE's Vulkan subgroup IR artifacts."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

from prove_vulkan_reductions import (
    collect_executable_subgroup_metadata, inspect_existing_dumps,
)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        files = root / "sum_square_256_executables"
        files.mkdir()
        (files / "configured.mlir").write_text(
            'hal.executable.variant target(<"vulkan-spirv", {'
            'subgroup_size_choices = [32, 64]}>)\n'
            'iree_codegen.dispatch_config @reduction {\n'
            '  workgroup_size = [64, 1, 1],\n'
            '  subgroup_size = 32\n'
            '}\n', encoding="utf-8"
        )
        (files / "spirv.mlir").write_text(
            'spirv.func @reduction() attributes {\n'
            '  spirv.entry_point_abi = '
            '#spirv.entry_point_abi<workgroup_size = [64, 1, 1], '
            'subgroup_size = 32>\n'
            '}\n', encoding="utf-8"
        )
        (files / "reduction.spv").write_bytes(
            struct.pack("<II", 0x07230203, 0x00010300)
        )
        evidence = collect_executable_subgroup_metadata(files)
        assert evidence["target_subgroup_size_choices"] == [[32, 64]]
        assert evidence["selected_subgroup_sizes_in_ir"] == [32]
        assert len(evidence["dispatch_configurations"]) == 1
        assert "workgroup_size = [64, 1, 1]" in (
            evidence["dispatch_configurations"][0]["config"]
        )
        assert len(evidence["spirv_entry_point_abis"]) == 1
        assert evidence["spirv_binaries"][0]["spirv_version"] == "1.3"

        (root / "report.json").write_text(json.dumps({
            "device_names": ["synthetic"],
            "cases": [{
                "name": "sum_square_256", "width": 256,
                "numerical_match": False,
                "one_hot": {"min": 2.0, "max": 2.0,
                            "roughly_once": 0, "roughly_twice": 256},
            }],
        }), encoding="utf-8")
        assert inspect_existing_dumps(root) == 0
        report = json.loads(
            (root / "subgroup-inspection.json").read_text(encoding="utf-8")
        )
        assert report["cases"][0]["codegen_subgroup_metadata"] == evidence
        assert report["cases"][0]["one_hot_summary"]["roughly_twice"] == 256

        # Absent pipeline requests must never be confused with the target's
        # supported subgroup_size_choices.
        (files / "configured.mlir").write_text(
            'subgroup_size_choices = [32, 64]\n'
            'iree_codegen.dispatch_config @reduction {\n'
            '  workgroup_size = [64, 1, 1]\n'
            '}\n', encoding="utf-8"
        )
        (files / "spirv.mlir").unlink()
        absent = collect_executable_subgroup_metadata(files)
        assert absent["target_subgroup_size_choices"] == [[32, 64]]
        assert absent["selected_subgroup_sizes_in_ir"] == []
        assert absent["spirv_entry_point_abis"] == []
    print("[subgroup inspection] all tests passed", flush=True)


if __name__ == "__main__":
    main()
