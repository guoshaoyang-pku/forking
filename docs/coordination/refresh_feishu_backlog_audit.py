#!/usr/bin/env python3
"""Refresh a local audit of Feishu TODO markers without editing cloud documents."""
import argparse, json, re, subprocess, hashlib
from pathlib import Path

URLS = {
    "todo": "https://gcnsa0liqppq.feishu.cn/wiki/G4KKwHMWriuHDOkARHCcHWBenwc",
    "paper": "https://gcnsa0liqppq.feishu.cn/wiki/Lyr3wfH8FiIWT6kdQmrc6dGUnEf",
}
PATTERN = re.compile(r"\[ \]|待补|待核|未启动|进行中|预留|TODO|NO-GO", re.I)

def fetch(url):
    p = subprocess.run(["lark-cli", "docs", "+fetch", "--doc", url,
                       "--doc-format", "markdown", "--detail", "simple"],
                      check=True, capture_output=True, text=True)
    return json.loads(p.stdout)["data"]["document"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/coordination/feishu-marker-audit-latest.json"); args=ap.parse_args()
    result = {"generated_by": "docs/coordination/refresh_feishu_backlog_audit.py", "sources": {}}
    for name, url in URLS.items():
        doc = fetch(url); content = doc["content"]; lines = [{"line": i, "text": l} for i,l in enumerate(content.splitlines(),1) if PATTERN.search(l)]
        result["sources"][name] = {"url": url, "revision_id": doc.get("revision_id"), "bytes": len(content.encode()), "sha256": hashlib.sha256(content.encode()).hexdigest(), "marker_count": len(lines), "markers": lines}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(args.out)
if __name__ == "__main__": main()
