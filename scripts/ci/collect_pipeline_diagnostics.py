#!/usr/bin/env python3
"""Collect all trained Vulkan pipeline reports into one shareable JSON.

The individual reports and compiler artifacts remain in their original
subdirectories. This collector copies the complete JSON reports into one
self-contained file and supplies a compact first-divergence summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORTS = (
    ("pipeline", Path("report.json")),
    ("stage1", Path("stage1/report.json")),
    ("stage1_diagnostic", Path("stage1_diagnostic/report.json")),
    ("stage2", Path("stage2/report.json")),
    ("execution", Path("execution/report.json")),
)
BUNDLE_NAME = "corridorkey-vulkan-diagnostics.json"


def load_reports(output: Path) -> dict:
    """Read available reports; missing stages are normal after early failure."""
    reports = {}
    for name, relative in REPORTS:
        path = output / relative
        if not path.is_file():
            continue
        try:
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Preserve a failed report's read error instead of presenting it
            # as a stage that was simply never executed.
            reports[name] = {
                "error": f"Could not read {relative.as_posix()}: {exc}"
            }
    return reports


def make_bundle(reports: dict) -> dict:
    """Prepare a self-contained, unabridged report with a useful summary."""
    pipeline = reports.get("pipeline", {})
    diagnostic = reports.get("stage1_diagnostic", {})
    first_divergence = (
        diagnostic.get("first_divergence")
        or pipeline.get("stage1_first_divergence")
    )
    probes = diagnostic.get("probes", [])
    summary = {
        "status": "failed" if any(
            report.get("error") for report in reports.values()
        ) or pipeline.get("stage1") is False
        and pipeline.get("error") else "passed" if (
            pipeline.get("stages_executed_without_pytorch") is True
            and not pipeline.get("error")
        ) else "incomplete",
        "first_divergence": first_divergence,
        "stage1_probes": [
            {
                key: entry[key]
                for key in (
                    "name", "shape", "max_absolute_error",
                    "mean_absolute_error", "mismatched", "total",
                ) if key in entry
            }
            for entry in probes
        ],
        "completed_stages": [
            name for name in ("stage1", "stage2", "execution")
            if name in reports and not reports[name].get("error")
        ],
        "available_reports": list(reports),
        "target": pipeline.get("gpu_target") or diagnostic.get("target"),
        "down_reduction": pipeline.get("down_reduction"),
        "fuse_reduction": pipeline.get("fuse_reduction"),
        "pipeline_error": pipeline.get("error"),
    }
    return {
        "schema_version": 1,
        "description": "CorridorKey trained Vulkan pipeline diagnostic bundle",
        "summary": summary,
        "reports": reports,
    }


def write_bundle(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    bundle = make_bundle(load_reports(output))
    path = output / BUNDLE_NAME
    path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Existing trained pipeline output directory",
    )
    args = parser.parse_args()
    path = write_bundle(args.output)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    print(f"[trained pipeline] Share this one file: {path}", flush=True)
    print(
        "[trained pipeline] Status: "
        f"{bundle['summary']['status']}; "
        f"first divergence: {bundle['summary']['first_divergence']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
