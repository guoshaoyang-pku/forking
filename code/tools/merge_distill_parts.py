#!/usr/bin/env python3
"""Merge per-worker teacher-residual parts into one table (pro5k6 -> lab).

Each worker writes ``part_w<k>_<tag>.npz`` with identical assignment keys; only
the residual sums and counts differ.  Merging therefore means summing sums and
counts, then optionally converting to per-context means.

    python3 merge_distill_parts.py --in_dir /data/distill_out --tag w3p \
        --out /data/distill_out/teacher_table_o2to5.npz [--mean]

Output adds, for every order n and depth d:
    mean_L<d>_<n>   (R_n, hid) fp32   sum / max(cnt,1)   (with --mean)
plus a ``coverage`` block: number of rows with cnt >= 1 / 2 / 4 / 8.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/data/distill_out")
    ap.add_argument("--tag", default="w3p")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mean", action="store_true", help="also write per-row means")
    ap.add_argument("--allow_partial", action="store_true",
                    help="proceed when only some workers have parts")
    ap.add_argument("--min_count", type=int, default=0,
                    help="zero out rows with cnt below this (keeps shape)")
    ap.add_argument("--include_ckpt", action="store_true",
                    help="also include periodic .ckpt.npz parts (final parts win when "
                         "both exist for a worker: pass --prefer_final to drop ckpts)")
    ap.add_argument("--parts", default="",
                    help="explicit comma-separated file list; bypasses --tag globbing and "
                         "is how two runs with different tags are combined (e.g. run 1's "
                         "last checkpoints plus run 2's final parts)")
    args = ap.parse_args()

    if args.parts:
        parts = [p.strip() for p in args.parts.split(",") if p.strip()]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"missing parts: {missing}")
    else:
        parts = sorted(glob.glob(os.path.join(args.in_dir, f"part_w*_{args.tag}.npz")))
        ckpts = sorted(glob.glob(os.path.join(args.in_dir, f".part_w*_{args.tag}.ckpt.npz")))
        if args.include_ckpt:
            # one part per worker: prefer the final file, else the newest checkpoint
            have = {os.path.basename(p).split("_")[1]: p for p in parts}
            for c in ckpts:
                w = os.path.basename(c).split("_")[1]
                if w not in have:
                    have[w] = c
            parts = [have[k] for k in sorted(have)]
    if not parts:
        raise SystemExit(f"no parts matching part_w*_{args.tag}.npz in {args.in_dir}")
    print(f"merging {len(parts)} parts:")
    for p in parts:
        print(f"  {p} ({os.path.getsize(p)/1e9:.2f}GB)")

    first = np.load(parts[0])
    orders = json.loads(str(first["meta_json"].item()))["orders"]
    arrays = {}
    got = set()
    tokens = docs = 0
    frac = {}
    for p in parts:
        z = np.load(p)
        meta = json.loads(str(z["meta_json"].item()))
        tokens += meta.get("tokens", 0)
        docs += meta.get("docs", 0)
        nkeys = len(z.files)
        for i, k in enumerate(z.files):
            if k == "meta_json":
                continue
            a = z[k]
            frac[p] = i / nkeys
            if k in arrays:
                arrays[k] = arrays[k] + a
            else:
                arrays[k] = a.copy()
        got.add(meta.get("worker", -1))
        print(f"  + {os.path.basename(p)} tokens={meta.get('tokens',0)/1e9:.3f}B "
              f"docs={meta.get('docs',0)} shards={len(meta.get('shards', []))}")

    out = dict(arrays)
    coverage = {}
    for n in orders:
        cnt = arrays[f"cnt_{n}"]
        for thr in (1, 2, 4, 8):
            coverage[f"cov_{n}_ge{thr}"] = int((cnt >= thr).sum())
        if args.mean:
            den = np.maximum(cnt, 1).astype(np.float32)[:, None]
            for k in list(arrays):
                if k.startswith("sum_") and k.endswith(f"_{n}"):
                    name = k.replace("sum_", "mean_", 1)
                    out[name] = (arrays[k] / den).astype(np.float32)
        if args.min_count:
            keep = cnt >= args.min_count
            out[f"keep_{n}"] = keep
    print(f"workers present: {sorted(got)}")
    out["coverage"] = np.array(json.dumps(coverage))
    out["meta_json"] = np.array(json.dumps({
        "parts": [os.path.basename(p) for p in parts],
        "tokens": int(tokens), "docs": int(docs), "orders": orders,
        "min_count": args.min_count, "coverage": coverage,
    }))
    np.savez(args.out, **out)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f}GB)")
    print(f"total project tokens = {tokens/1e9:.3f}B over {docs} docs")
    for n in orders:
        print(f"  order {n}: rows={arrays[f'cnt_{n}'].size} "
              f"nonzero={coverage[f'cov_{n}_ge1']} "
              f"ge2={coverage[f'cov_{n}_ge2']} ge4={coverage[f'cov_{n}_ge4']} "
              f"ge8={coverage[f'cov_{n}_ge8']}")


if __name__ == "__main__":
    main()
