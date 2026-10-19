#!/usr/bin/env python3
"""W1.5 exact (identity) assignment builder: every selected context gets its
own row (R_n = K_n per order, rows = arange). Built from any existing
full-coverage assignment npz (e.g. assign_h2_974k.npz) by replacing rows.

Zero-collision reference arm; under adam+clip (post-§99) this is deployable.

CLI: python3 w1_make_exact.py <src_assign_npz> <out_path>
"""
import sys
import json
import numpy as np


def main():
    src, out_path = sys.argv[1], sys.argv[2]
    z = np.load(src)
    orders = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("keys_"))
    out = {}
    total_rows = 0
    for n in orders:
        keys = z[f"keys_{n}"]
        K = keys.size
        out[f"keys_{n}"] = keys
        out[f"rows_{n}"] = np.arange(K, dtype=np.int32)
        out[f"R_{n}"] = np.int64(K)
        out[f"counts_{n}"] = z[f"counts_{n}"]
        total_rows += K
    out["counts_total"] = z["counts_total"] if "counts_total" in z.files else np.int64(0)
    meta = {"algo": "exact", "src": src, "note": "identity assignment, zero collision",
            "total_rows": int(total_rows)}
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez(out_path, **out)
    print(f"[exact] wrote {out_path}: total_rows={total_rows}")


if __name__ == "__main__":
    main()
