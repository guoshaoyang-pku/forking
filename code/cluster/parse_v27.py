#!/usr/bin/env python3
"""Parse harvested v27 runs -> status table + E3-window paired deltas.

Reads data/harvest/v27_a100/<host>/<run_id>/{train_log.jsonl,summary.json}.
run_id grammar: v27_<model>_<arm>_r<rk>k_dn<dn>_tls<tls>_s<seed>_e<ep>[_sub<f>p<p>][_nl]_fixed
Emits rows: host run_id model arm rk dn tls seed ep steps last_train last_val
            e3_val (mean of val points in final 1/10 of training) done
Writes data/harvest/v27_a100/status.csv
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/.tmp/harvest/v27_a100"
PAT = re.compile(
    r"v27_(?P<model>\w+?)_(?P<arm>exact|nogram)_r(?P<rk>\d+)k_dn(?P<dn>\d+)"
    r"_tls(?P<tls>[\d.]+)_s(?P<seed>\d+)_e(?P<ep>\d+)"
    r"(?:_sub(?P<sub>[\d.]+)p(?P<p>\d+))?(?P<nl>_nl)?_fixed")


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    out_rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "v27_*_fixed"))):
        host = os.path.basename(os.path.dirname(d))
        run_id = os.path.basename(d)
        m = PAT.match(run_id)
        if not m:
            continue
        g = m.groupdict()
        jl = os.path.join(d, "train_log.jsonl")
        if not os.path.exists(jl):
            continue
        recs = read_jsonl(jl)
        recs = [r for r in recs if isinstance(r, dict) and "step" in r]
        if not recs:
            continue
        vals = [r["val_loss"] for r in recs if r.get("val_loss") is not None]
        steps = [r["step"] for r in recs]
        last_step = max(steps) if steps else 0
        # E3-window: val points with step in the final 10% of observed range
        val_steps = [r["step"] for r in recs if r.get("val_loss") is not None]
        cutoff = 0.9 * max(val_steps) if val_steps else 0
        e3_vals = [r["val_loss"] for r in recs
                   if r.get("val_loss") is not None and r["step"] >= cutoff]
        done = os.path.exists(os.path.join(d, "summary.json"))
        out_rows.append({
            "host": host, "run_id": run_id,
            "model": g["model"], "arm": g["arm"], "rk": int(g["rk"]),
            "dn": int(g["dn"]), "tls": g["tls"], "seed": int(g["seed"]),
            "ep": int(g["ep"]), "sub": g.get("sub") or "", "nl": bool(g.get("nl")),
            "steps": len(recs), "last_step": last_step, "n_val": len(vals),
            "last_val": vals[-1] if vals else "",
            "e3_val": round(sum(e3_vals) / len(e3_vals), 6) if e3_vals else "",
            "done": done,
        })
    fields = ["host", "run_id", "model", "arm", "rk", "dn", "tls", "seed", "ep",
              "sub", "nl", "steps", "last_step", "n_val", "last_val", "e3_val", "done"]
    out = os.path.join(ROOT, "status.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    # quick paired view (exact vs nogram, same model/dn/seed/ep/sub)
    key = lambda r: (r["model"], r["dn"], r["seed"], r["ep"], r["sub"])
    by_key = {}
    for r in out_rows:
        by_key.setdefault(key(r), {})[r["arm"]] = r
    print(f"{'model':6} {'dn':>4} {'s':>3} {'e':>2} {'sub':>8} | {'nog e3':>8} {'exact e3':>8} {'dVal mK':>8} | done")
    for k in sorted(by_key, key=lambda x: (x[0], x[1], x[2], x[4])):
        pair = by_key[k]
        nog, ex = pair.get("nogram"), pair.get("exact")
        if not (nog and ex and nog["e3_val"] != "" and ex["e3_val"] != ""):
            continue
        dv = (nog["e3_val"] - ex["e3_val"]) * 1000
        print(f"{k[0]:6} {k[1]:>4} {k[2]:>3} {k[3]:>2} {k[4] or '-':>8} | "
              f"{nog['e3_val']:>8.4f} {ex['e3_val']:>8.4f} {dv:>8.1f} | "
              f"{'Y' if nog['done'] and ex['done'] else 'run'}")
    print(f"\nwrote {out} ({len(out_rows)} runs)")


if __name__ == "__main__":
    main()
