#!/usr/bin/env python3
"""Tests for the trained pipeline's single shareable diagnostic artifact."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from collect_pipeline_diagnostics import (
    BUNDLE_NAME, load_reports, make_bundle, write_bundle,
)


def dump(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def run():
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        dump(output / "report.json", {
            "gpu_target": "rdna3", "down_reduction": "segmented128",
            "fuse_reduction": "segmented128", "stage1": False,
            "error": "Stage1 failed",
        })
        dump(output / "stage1/report.json", {
            "vulkan_execute": False,
            "error": "Stage1 output differs",
        })
        dump(output / "stage1_diagnostic/report.json", {
            "first_divergence": "probe_body_64.0.attn",
            "probes": [
                {"name": "probe_fuse", "mismatched": 0, "total": 1536},
                {
                    "name": "probe_body_64.0.attn",
                    "mismatched": 1511, "total": 1536,
                    "max_absolute_error": 0.67766345,
                },
            ],
        })
        bundle_path = write_bundle(output)
        assert bundle_path.name == BUNDLE_NAME
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert payload["summary"]["status"] == "failed"
        assert payload["summary"]["first_divergence"] == "probe_body_64.0.attn"
        assert payload["summary"]["available_reports"] == [
            "pipeline", "stage1", "stage1_diagnostic"
        ]
        assert payload["summary"]["stage1_probes"][1]["mismatched"] == 1511
        assert payload["reports"]["stage1"]["error"] == "Stage1 output differs"
        assert "stage2" not in payload["reports"]
        assert payload == make_bundle(load_reports(output))

        # An incomplete run may have no stage reports.
        (output / "stage1/report.json").unlink()
        (output / "stage1_diagnostic/report.json").unlink()
        (output / "report.json").unlink()
        assert make_bundle(load_reports(output))["summary"]["status"] == "incomplete"

        # A fully executed pipeline should report success without requiring
        # the differential instrumentation to have run.
        dump(output / "report.json", {
            "stages_executed_without_pytorch": True, "stage1": True,
            "stage2": True,
        })
        dump(output / "stage1/report.json", {"vulkan_execute": True})
        dump(output / "stage2/report.json", {"vulkan_execute": True})
        dump(output / "execution/report.json", {
            "success": True, "torch_runtime": False,
        })
        success = make_bundle(load_reports(output))
        assert success["summary"]["status"] == "passed"
        assert success["summary"]["first_divergence"] is None
        assert success["summary"]["completed_stages"] == [
            "stage1", "stage2", "execution",
        ]
    print("[trained pipeline] consolidated diagnostic tests passed", flush=True)


if __name__ == "__main__":
    run()
