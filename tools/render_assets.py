#!/usr/bin/env python3
"""Render the pinned, single-page paper figures without changing their content."""
import hashlib
import json
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "docs/figs/manifest.json"
manifest = json.loads(manifest_path.read_text())
for figure in manifest["figures"]:
    source = ROOT / figure["pdf"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != figure["source_sha256"]:
        raise SystemExit(f"Hash mismatch: {source.name}")
for figure in manifest["figures"]:
    with pymupdf.open(ROOT / figure["pdf"]) as document:
        if len(document) != 1:
            raise SystemExit(f"Expected one page: {figure['pdf']}")
        page = document[0]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2100 / page.rect.width, 2100 / page.rect.width), alpha=False)
        target = ROOT / figure["png"]
        pix.save(target)
        figure.update(width=pix.width, height=pix.height, png_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Rendered {len(manifest['figures'])} pinned paper assets")
