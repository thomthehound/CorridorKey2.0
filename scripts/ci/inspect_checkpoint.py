#!/usr/bin/env python3
"""Inspect CorridorKey runtime checkpoint metadata without loading neural tensors.

Reads only the safetensors header, so checkpoint releases can be inventoried
on machines without a GPU, PyTorch, CUDA, ROCm, or ONNX. This does not establish
that model graphs can be compiled or that weights numerically match inference.
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import tempfile
from pathlib import Path

HEADER_LIMIT = 128 * 1024 * 1024


def model_constructor_parameters(source: Path) -> set[str]:
    root = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in root.body:
        if isinstance(node, ast.ClassDef) and node.name == "CorridorKeyV3":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                    return {arg.arg for arg in child.args.posonlyargs + child.args.args
                            if arg.arg != "self"}
    raise ValueError(f"Could not locate CorridorKeyV3 constructor in {source}")


def inspect_checkpoint(path: Path, model_source: Path) -> dict:
    with path.open("rb") as stream:
        header_size_raw = stream.read(8)
        if len(header_size_raw) != 8:
            raise ValueError("Incomplete safetensors header length")
        (header_size,) = struct.unpack("<Q", header_size_raw)
        if not 0 < header_size <= HEADER_LIMIT:
            raise ValueError(f"Invalid safetensors header size {header_size}")
        header_raw = stream.read(header_size)
        if len(header_raw) != header_size:
            raise ValueError("Truncated safetensors header")
    header = json.loads(header_raw)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    meta = header.pop("__metadata__", None)
    if not isinstance(meta, dict):
        raise ValueError("Checkpoint is missing metadata")
    if meta.get("format") != "corridorkey_v2_runtime_checkpoint":
        raise ValueError(f"Unsupported checkpoint format: {meta.get('format')!r}")
    cfg = json.loads(meta.get("model_config", "null"))
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint model_config must be a JSON object")
    known = model_constructor_parameters(model_source)
    unrecognized = sorted(set(cfg).difference(known))
    if unrecognized:
        raise ValueError(f"Unknown model constructor fields: {unrecognized}")
    reserved = {"use_moge", "use_cradio", "rvm_ckpt", "rvm_variant",
                "stage1_recurrent_64_sites"}
    extra = {}
    for name, entry in header.items():
        if not isinstance(entry, dict) or not {"dtype", "shape", "data_offsets"} <= entry.keys():
            raise ValueError(f"Malformed tensor descriptor {name}")
        offsets = entry["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets)
                or offsets[0] < 0 or offsets[1] < offsets[0]):
            raise ValueError(f"Invalid tensor offsets for {name}")
        if (not isinstance(entry["shape"], list)
                or any(type(v) is not int or v < 0 for v in entry["shape"])):
            raise ValueError(f"Invalid shape for tensor {name}")
        extra[name] = entry
    data_bytes = path.stat().st_size - 8 - header_size
    if data_bytes < 0:
        raise ValueError("Truncated checkpoint file")
    max_offset = max((entry["data_offsets"][1] for entry in extra.values()), default=0)
    if max_offset > data_bytes:
        raise ValueError(
            f"Tensor data extends beyond checkpoint ({max_offset} > {data_bytes})"
        )
    dtype_counts = {}
    for t in extra.values():
        dtype_counts[t["dtype"]] = dtype_counts.get(t["dtype"], 0) + 1
    return {
        "checkpoint": str(path),
        "checkpoint_format": meta["format"],
        "checkpoint_metadata": json.loads(meta.get("metadata", "{}")),
        "architecture_config": cfg,
        "source_constructor_fields": len(known),
        "checkpoint_explicit_foundations": {
            "moge": cfg.get("use_moge", True),
            "cradio": cfg.get("use_cradio", False),
            "rvm": "selected from runtime-provided checkpoint asset",
        },
        "runtime_behavior": {
            "foundation model paths injected by load_model": True,
            "stage1_recurrent_64_sites overridden to empty": True,
        },
        "tensor_count": len(extra),
        "tensor_dtypes": dtype_counts,
        "tensor_bytes": max_offset,
    }


def self_test(source: Path) -> None:
    config = {"d_16": 512, "use_moge": True, "use_cradio": False}
    meta = {"format": "corridorkey_v2_runtime_checkpoint",
            "model_config": json.dumps(config), "metadata": "{}"}
    hdr = {"__metadata__": meta,
           "stage1.test": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    with tempfile.TemporaryDirectory(prefix="corridorkey-checkpoint-") as d:
        p = Path(d) / "fixture.safetensors"
        raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
        result = inspect_checkpoint(p, source)
        assert result["tensor_count"] == 1
        assert result["architecture_config"] == config
        assert result["tensor_dtypes"] == {"F32": 1}
        bad = dict(hdr)
        bad["__metadata__"] = dict(meta, model_config='{"made_up_parameter": 1}')
        raw = json.dumps(bad).encode("utf-8")
        p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
        try:
            inspect_checkpoint(p, source)
        except ValueError as exc:
            assert "Unknown model constructor" in str(exc)
        else:
            raise AssertionError("Unknown architecture parameter was accepted")
    print("Checkpoint metadata self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] /
                        "corridorkey/model.py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.source)
        return
    if args.checkpoint is None:
        parser.error("Specify a checkpoint path or --self-test")
    report = inspect_checkpoint(args.checkpoint, args.source)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
