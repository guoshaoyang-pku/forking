#!/usr/bin/env python3
"""D/N=20 ladder exact assignment builder (user 2026-10-20 invariant scan).

Per-order top-R by count from a freq-index npz, identity rows (zero collision,
exact analog of w1_make_exact but count-truncated to R per order instead of
full f>=256 coverage). Ladder rule: R_per_order = 8 * backbone params.

usage: dn20_make_exact_r.py <freq_index_npz> <R_per_order> <out_path>
"""
import sys
import json
import numpy as np

ORDERS = (2, 3, 4, 5)


def main():
    index_path, R, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    z = np.load(index_path)
    out = {}
    info = {}
    total_rows = 0
    for n in ORDERS:
        keys = z[f"keys_{n}"].astype(np.int64)
        counts = z[f"counts_{n}"].astype(np.int64)
        order = np.argsort(keys)
        keys_s, counts_s = keys[order], counts[order]
        K = min(R, keys_s.size)
        idx = np.argpartition(-counts_s, K - 1)[:K]
        sel = np.sort(keys_s[idx])
        pos = np.searchsorted(keys_s, sel)
        out[f"keys_{n}"] = sel
        out[f"rows_{n}"] = np.arange(K, dtype=np.int32)
        out[f"R_{n}"] = np.int64(K)
        out[f"counts_{n}"] = counts_s[pos]
        total_rows += K
        info[str(n)] = {"R": int(K), "requested": R,
                        "min_count": int(counts_s[pos].min()),
                        "distinct_total": int(keys_s.size)}
    out["counts_total"] = np.int64(sum(int(out[f"counts_{n}"].sum()) for n in ORDERS))
    out["meta_json"] = np.array(json.dumps(
        {"algo": "exact_topR", "src": index_path, "per_order": info,
         "total_rows": int(total_rows)}, sort_keys=True))
    np.savez_compressed(out_path, **out)
    print(f"[dn20] wrote {out_path}: total_rows={total_rows} per_order={info}")


if __name__ == "__main__":
    main()
