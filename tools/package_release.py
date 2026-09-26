#!/usr/bin/env python3
"""Pin the curated file inventory and optionally create a new ZIP without caches."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"code", "configs", "reproduce", "evidence", "tests", "docs", "tools", "scripts", ".github"}
ROOT_FILES = {"README.md", "AGENTS.md", "CODE_PROVENANCE.json", ".gitignore", ".gitattributes", "pytest.ini",
              "requirements-test.txt", "requirements-site.txt", "requirements-figures.txt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="New archive path outside the candidate directory")
    args = parser.parse_args()
    if args.zip and (args.zip.exists() or args.zip.resolve().is_relative_to(ROOT)):
        parser.error("Choose a new ZIP path outside the candidate directory")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in {"__pycache__", ".pytest_cache", ".DS_Store"} for part in rel.parts):
            continue
        if path.suffix == ".pyc":
            continue
        if (len(rel.parts) == 1 and rel.name in ROOT_FILES) or (len(rel.parts) > 1 and rel.parts[0] in DIRECTORIES):
            files.append(path)
    inventory = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    manifest = {"format_version": 1, "prepared": "2026-09-26", "title": "Forking: Sudden Overfitting Under Replay",
                "scope": "Curated release branch. The manifest excludes itself, local caches, and generated training outputs.",
                "public_source_revision": "3262ddd4fc0af1e9702725bc9aaabed68b824abc",
                "paper_figure_revision": "83f17dbc441e61d6a73b7cc0952801f6fc4f3999",
                "file_count": len(inventory), "total_bytes": sum(i["bytes"] for i in inventory), "files": inventory}
    target = ROOT / "RELEASE_MANIFEST.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files + [target]:
                info = zipfile.ZipInfo("forking/" + path.relative_to(ROOT).as_posix(), date_time=(2026, 9, 26, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())
        print(json.dumps({"archive": str(args.zip), "sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest()}))
    print(json.dumps({"files": len(files) + 1, "bytes": manifest["total_bytes"] + target.stat().st_size}))


if __name__ == "__main__":
    main()
