#!/usr/bin/env python3
"""Shared pieces for W1 routing builders (teacher-free OE algorithm search).

Canonical artifacts (single source of truth for all W1 arms):
  - selected context sets: f>=256 per order from freq_index_o2to5_1b_f30.npz
    (bit-identical to the v25 f256 dict membership, verified against run logs)
  - R_n allocation: largest remainder, sum(R_n) = 131072, R_n proportional to
    the number of selected contexts of order n (plan v2 section 4)
  - assignment npz format consumed by train.py --ngram_assign_file:
      keys_<n> int64 (K_n,) sorted; rows_<n> int32 (K_n,) or (2,K_n);
      R_<n> int64 scalar; counts_<n> int64; counts_total int64; meta_json str
"""
import json
import numpy as np

ORDERS = (2, 3, 4, 5)
M_TOTAL = 131072
FMIN = 256
ROUTING_SEED = 0xC1EA00D5  # fixed for W1; routing-seed ablations come later


def load_selected(index_path: str):
    z = np.load(index_path)
    sel = {}
    for n in ORDERS:
        c = z[f"counts_{n}"].astype(np.int64)
        m = c >= FMIN
        sel[n] = (np.ascontiguousarray(z[f"keys_{n}"][m]), c[m])
    total = int(sum(z[f"counts_{n}"].astype(np.int64).sum() for n in ORDERS))
    return sel, total


def allocate_R(sel, m_total=M_TOTAL):
    """Largest remainder: R_n proportional to #selected contexts, sum = m_total."""
    sizes = {n: sel[n][0].size for n in ORDERS}
    tot = sum(sizes.values())
    raw = {n: m_total * sizes[n] / tot for n in ORDERS}
    R = {n: int(np.floor(raw[n])) for n in ORDERS}
    R = {n: max(1, min(R[n], sizes[n])) for n in ORDERS}
    rem = m_total - sum(R.values())
    frac_order = sorted(ORDERS, key=lambda n: (raw[n] - np.floor(raw[n])), reverse=True)
    i = 0
    while rem > 0:
        n = frac_order[i % len(frac_order)]
        if R[n] < sizes[n]:
            R[n] += 1
            rem -= 1
        i += 1
    return R


def splitmix64(z):
    z = np.asarray(z, dtype=np.uint64)
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def write_assignment(path, sel, R_map, rows_map, counts_total, meta):
    out = {}
    for n in ORDERS:
        keys, counts = sel[n]
        out[f"keys_{n}"] = keys
        out[f"rows_{n}"] = rows_map[n].astype(np.int32)
        out[f"R_{n}"] = np.int64(R_map[n])
        out[f"counts_{n}"] = counts
    out["counts_total"] = np.int64(counts_total)
    out["meta_json"] = json.dumps(meta, sort_keys=True)
    np.savez_compressed(path, **out)
    print(f"[w1] wrote {path}: {meta.get('algo')} R={R_map}")
