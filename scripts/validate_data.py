#!/usr/bin/env python3
"""Verify the byte identity of the published baseline's external input files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "data_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(data_dir: Path, freq_index: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    checked = []
    errors = []
    for item in manifest["files"]:
        path = freq_index if item["role"] == "frequency_index" else data_dir / item["name"]
        if not path.is_file():
            errors.append(f"missing input: {path}")
            continue
        size = path.stat().st_size
        if size != item["bytes"]:
            errors.append(f"size mismatch for {path.name}: expected {item['bytes']}, got {size}")
            continue
        digest = sha256(path)
        if digest != item["sha256"]:
            errors.append(f"SHA-256 mismatch for {path.name}")
            continue
        checked.append({"name": item["name"], "bytes": size, "sha256": digest})
    return {
        "ok": not errors,
        "manifest_sha256": sha256(MANIFEST),
        "files_verified": checked,
        "errors": errors,
        "scope": "Byte identity against the baseline input inventory; does not certify corpus rights or tokenizer provenance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--freq-index", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.data_dir.resolve(), args.freq_index.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
