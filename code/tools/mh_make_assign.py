#!/usr/bin/env python3
"""Multi-head hash assignment (Engram-style): per order keep the top-K contexts by
count, map each to H rows, one per head, with H independent splitmix64 hashes
into R rows each. In train.py each head is a separate d/H-wide table and the
readout is the concatenation, so total table params = R * d per order for any H
(param-matched to an H=1 table of R rows).

--shared: every head uses head 0's hash (the "same mapping, several tables"
control; with concat readout it is exactly one d-wide table of R rows).

usage: mh_make_assign.py <freq_index_npz> <K_per_order> <R_per_head> <H> <out> [--shared]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w1_common import splitmix64  # noqa: E402

ORDERS = (2, 3, 4, 5)
SEED = 0xC1EA00D5


def head_rows(keys_u64, n, h, R):
    salt = np.uint64((SEED ^ (0x9E3779B9 * n) ^ (0x7F4A7C15 * (h + 1))) & 0xFFFFFFFFFFFFFFFF)
    return (splitmix64(keys_u64 ^ salt) % np.uint64(R)).astype(np.int32)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    shared = "--shared" in sys.argv
    index_path, K, R, H, out_path = args[0], int(args[1]), int(args[2]), int(args[3]), args[4]
    z = np.load(index_path)
    out, info = {}, {}
    for n in ORDERS:
        keys = z[f"keys_{n}"].astype(np.int64)
        counts = z[f"counts_{n}"].astype(np.int64)
        k = min(K, keys.size)
        idx = np.argpartition(-counts, k - 1)[:k]
        order = np.argsort(keys[idx])
        sel, cnt = keys[idx][order], counts[idx][order]
        ku = sel.astype(np.uint64)
        heads = [head_rows(ku, n, 0 if shared else h, R) for h in range(H)]
        rows = heads[0] if H == 1 else np.stack(heads)
        out[f"keys_{n}"], out[f"rows_{n}"], out[f"R_{n}"], out[f"counts_{n}"] = sel, rows, np.int64(R), cnt
        load = np.bincount(heads[0], minlength=R)
        info[str(n)] = {"K": int(k), "R": R, "min_count": int(cnt.min()),
                        "keys_per_row": round(k / R, 3), "empty_rows_h0": int((load == 0).sum()),
                        "distinct_total": int(keys.size)}
    out["counts_total"] = np.int64(sum(int(out[f"counts_{n}"].sum()) for n in ORDERS))
    out["meta_json"] = np.array(json.dumps({"algo": "multihead_hash", "H": H, "shared": shared,
                                            "seed": SEED, "src": index_path, "per_order": info},
                                           sort_keys=True))
    np.savez_compressed(out_path, **out)
    print(f"[mh] wrote {out_path}: H={H} shared={shared} {info}")


if __name__ == "__main__":
    main()
