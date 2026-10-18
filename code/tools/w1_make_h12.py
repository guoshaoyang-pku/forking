#!/usr/bin/env python3
"""W1 random-hash controls H1 / H2 (plan v2 section 4).

H1: filtered single hash -- each selected context c of order n maps to
    splitmix64(key ^ seed_n) % R_n; unselected contexts are absent (zero).
    Random collisions among selected contexts are kept.
H2: filtered two-head random hash -- two independent hashes into two tables of
    R_n rows (each half width in the model), matched params/active-width.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w1_common import (ORDERS, ROUTING_SEED, load_selected, allocate_R,
                       splitmix64, write_assignment)

def build(index_path, algo, out_path, seed=ROUTING_SEED):
    sel, total = load_selected(index_path)
    R = allocate_R(sel)
    rows = {}
    for n in ORDERS:
        keys = sel[n][0].astype(np.uint64)
        h1 = splitmix64(keys ^ np.uint64(seed ^ (0x9E3779B9 * n)))
        if algo == "h1":
            rows[n] = (h1 % np.uint64(R[n])).astype(np.int32)
        else:
            h2 = splitmix64(keys ^ np.uint64(seed ^ (0x7F4A7C15 * n)) ^ np.uint64(0xD1B54A32D192ED03))
            rows[n] = np.stack([(h1 % np.uint64(R[n])).astype(np.int32),
                                (h2 % np.uint64(R[n])).astype(np.int32)])
    write_assignment(out_path, sel, R, rows, total,
                     {"algo": algo, "seed": seed, "fmin": 256, "m_total": sum(R.values())})

if __name__ == "__main__":
    index_path, algo, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    assert algo in ("h1", "h2")
    build(index_path, algo, out_path)
