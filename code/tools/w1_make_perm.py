#!/usr/bin/env python3
"""W1.5 PERM control: shuffle a semantic assignment within same-order
log-count bins (fork-summary-v3 2026-10-19, queue item 1).

PERM destroys semantic grouping while preserving, per order:
  - the codebook shape (same R_n, same row multiplicities within each bin),
  - the traffic distribution (each bin keeps the same multiset of rows).
If PERM ~= C3 on LM validation, the C3 gain comes from the occupancy
structure, not semantics. If C3 > PERM, semantics is worth real nats.

CLI: python3 w1_make_perm.py <src_assign_npz> <out_path> [--seed S]
"""
import sys
import json
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0xC1EA00D5)
    args = ap.parse_args()

    z = np.load(args.src)
    orders = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("keys_"))
    out = {}
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    for n in orders:
        keys = z[f"keys_{n}"]
        rows = z[f"rows_{n}"]
        counts = z[f"counts_{n}"].astype(np.int64)
        bins = np.floor(np.log2(np.maximum(counts, 1))).astype(np.int64)
        perm_rows = rows.copy()
        rng = np.random.default_rng(args.seed ^ (0x9E3779B9 * n))
        for b in np.unique(bins):
            m = bins == b
            if rows.ndim == 2:
                perm = rng.permutation(m.sum())
                perm_rows[:, m] = rows[:, m][:, perm]
            else:
                perm = rng.permutation(m.sum())
                perm_rows[m] = rows[m][perm]
        out[f"keys_{n}"] = keys
        out[f"rows_{n}"] = perm_rows.astype(np.int32)
        out[f"R_{n}"] = z[f"R_{n}"]
        out[f"counts_{n}"] = z[f"counts_{n}"]
    out["counts_total"] = z["counts_total"] if "counts_total" in z.files else np.int64(0)
    meta = dict(meta, algo="perm", src=str(args.src), seed=int(args.seed),
                note="rows permuted within same-order log2-count bins")
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez(args.out, **out)
    print(f"[perm] wrote {args.out}: orders={orders}")


if __name__ == "__main__":
    main()
