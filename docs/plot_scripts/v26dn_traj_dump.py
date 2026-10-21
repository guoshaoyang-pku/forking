#!/usr/bin/env python3
"""Dump a binned val/train trajectory per done v26dn run, for the §129 audit.

Companion to `v26dn_harvest.py`: that one emits epoch-boundary readings, this one
emits the *shape* of the run, which is what tells us whether a run is
quasi-stationary (val monotonically improving) or went through a transient.

Output is TSV so the two machines' dumps can be merged locally, exactly like the
harvest TSVs.

Usage
-----
    python3 docs/plot_scripts/v26dn_traj_dump.py <runs_fixed_dir> [--bins 24]

Columns: run_id, n_rows, then `v{i}` and `t{i}` (bin means of val_loss and
train_loss) for i in 0..bins-1.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

BINS = 24


def binned(rows, key, bins):
    vals = [r[key] for r in rows if r.get(key) is not None]
    n = len(vals)
    if n < bins:
        return None, n
    size = n // bins
    return [st.mean(vals[i * size:(i + 1) * size]) for i in range(bins)], n


def main():
    argv = sys.argv[1:]
    if "--bins" in argv:
        i = argv.index("--bins")
        global BINS
        BINS = int(argv[i + 1])
        del argv[i:i + 2]
    root = Path(argv[0])
    out = []
    for d in sorted(root.glob("v26dn_*_fd_fixed")):
        log = d / "train_log.jsonl"
        if not log.exists():
            continue
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        v, n = binned(rows, "val_loss", BINS)
        if v is None:
            continue
        t, _ = binned(rows, "train_loss", BINS)
        out.append((d.name[: -len("_fixed")], n, v, t))
    hdr = ["run_id", "n_rows"] + [f"v{i}" for i in range(BINS)] + [f"t{i}" for i in range(BINS)]
    print("\t".join(hdr))
    for rid, n, v, t in out:
        cells = [rid, str(n)] + [f"{x:.4f}" for x in v] + [f"{x:.4f}" for x in t]
        print("\t".join(cells))


if __name__ == "__main__":
    main()
