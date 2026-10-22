#!/usr/bin/env python3
"""Print the shortest prefix of manifest.train_order (comma list) whose chunks cover D tokens.
usage: v27_prefix.py <manifest.json> <D_tokens>"""
import json
import sys

m = json.load(open(sys.argv[1]))
need = int(sys.argv[2])
acc, out = 0, []
for s in m["train_order"]:
    out.append(s)
    acc += m["chunks"][str(s)] * 2048
    if acc >= need:
        break
if acc < need:
    sys.exit(f"clean pool has {acc} tokens < D={need}")
print(",".join(map(str, out)))
