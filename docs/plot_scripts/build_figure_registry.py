#!/usr/bin/env python3
"""Build a stable, full asset registry for docs/figs."""
import json, re, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figs"
OUT = ROOT / "docs" / "figure-registry.json"
EXTS = {".png", ".svg", ".html", ".jpg", ".jpeg", ".webp"}

def main():
    old = {}
    if OUT.exists():
        try: old = {x["path"]: x for x in json.loads(OUT.read_text()).get("assets", [])}
        except Exception: pass
    used = {x.get("number") for x in old.values()}
    next_id = 1
    assets = []
    for p in sorted(x for x in FIG.rglob("*") if x.is_file() and x.suffix.lower() in EXTS):
        path = p.relative_to(ROOT).as_posix()
        prev = old.get(path, {})
        number = prev.get("number")
        if not number:
            while f"图0-{next_id}" in used: next_id += 1
            number = f"图0-{next_id}"; used.add(number); next_id += 1
        assets.append({"number": number, "path": path, "asset_type": p.suffix[1:].lower(), "bytes": p.stat().st_size, "scripts": [], "data_paths": ["待按对应生成脚本核实"], "clusters": ["待按对应 run 核实"], "status": "asset-in-git"})
    payload = {"schema": "figure-asset-registry-v1", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(), "asset_root": "docs/figs/", "numbering_rule": "正式图沿用图章节-序号；未编号资产使用稳定图0-N；已有 number 永不重排。", "asset_count": len(assets), "assets": assets}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}: {len(assets)} assets")

if __name__ == "__main__": main()
