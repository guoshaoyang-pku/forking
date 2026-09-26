#!/usr/bin/env python3
"""Read-only integrity and site-root link check; does not verify remote publication."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids, self.links, self.images = set(), [], []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for name in ("href", "src"):
            if name in attrs:
                self.links.append(attrs[name])
        if tag == "img":
            self.images.append(attrs)


def main():
    errors, checked = [], 0
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            errors.append("File identity mismatch: " + row["path"])
    pages = {p.resolve(): Page(p) for p in DOCS.glob("*.html")}
    for path, page in pages.items():
        for target in page.links:
            url = urlsplit(target)
            if url.scheme or url.netloc:
                if url.netloc == "github.com" and url.path.startswith("/guoshaoyang-pku/forking/blob/"):
                    parts = url.path.split("/", 5)
                    if len(parts) != 6 or parts[4] != "release":
                        errors.append(f"Wrong source branch: {path.name}: {target}")
                    elif not (ROOT / unquote(parts[5])).exists():
                        errors.append(f"Missing source file: {path.name}: {target}")
                continue
            resolved = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not resolved.is_relative_to(DOCS):
                errors.append(f"Escapes deployed docs root: {path.name}: {target}")
            elif not resolved.exists():
                errors.append(f"Missing site target: {path.name}: {target}")
            elif url.fragment and resolved in pages and unquote(url.fragment) not in pages[resolved].ids:
                errors.append(f"Missing anchor: {path.name}: {target}")
            checked += 1
        for img in page.images:
            if not img.get("alt"):
                errors.append("Missing image alternative: " + path.name)
    markdown_links = 0
    for path in list(ROOT.glob("*.md")) + list(DOCS.glob("*.md")) + list((ROOT / "reproduce").glob("*.md")):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            if not (path.parent / unquote(url.path)).exists():
                errors.append(f"Missing Markdown target: {path.name}: {target}")
            markdown_links += 1
    figures = json.loads((DOCS / "figs/manifest.json").read_text())["figures"]
    for f in figures:
        for key, hashkey in (("pdf", "source_sha256"), ("png", "png_sha256")):
            if hashlib.sha256((ROOT / f[key]).read_bytes()).hexdigest() != f[hashkey]:
                errors.append("Figure hash mismatch: " + f[key])
        size = struct.unpack(">II", (ROOT / f["png"]).read_bytes()[16:24])
        if size != (f["width"], f["height"]):
            errors.append("Figure dimension mismatch: " + f["png"])
    python_count = 0
    for row in manifest["files"]:
        if row["path"].endswith(".py"):
            try:
                ast.parse((ROOT / row["path"]).read_text())
                python_count += 1
            except SyntaxError:
                errors.append("Python syntax error: " + row["path"])
    report = dict(status="FAIL" if errors else "PASS", pinned_files=len(manifest["files"]),
                  html_pages=len(pages), local_site_links=checked, markdown_links=markdown_links,
                  figure_pairs=len(figures), python_files=python_count, errors=errors,
                  scope="Local integrity and site-root links; no remote publication or experiment certification.")
    print(json.dumps(report, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
