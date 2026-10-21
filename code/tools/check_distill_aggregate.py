#!/usr/bin/env python3
"""Sanity-check a distilled teacher-residual table (aggregate-level validation).

Reads only the merged/part npz, so it is independent of the production code path.

    python3 check_distill_aggregate.py /data/distill_out/teacher_table_o2to5.npz

Checks
  1. counts / coverage per order, and the mean count.
  2. per-depth mean-RMS versus the teacher's own hidden-state RMS measured with HF
     on a real document (d0=0.0105, d10=0.0516, d25=0.1079, d40=0.340,
     final normed=2.494), swept over count thresholds.  A correctly aligned
     accumulation keeps the mean RMS near the depth reference; a misaligned one
     (context bound to the wrong target position) averages unrelated states and
     decays towards zero as the count grows.
  3. finiteness of every stored mean.
"""
import argparse
import json

import numpy as np

REF_RMS = {"0": 0.0105, "10": 0.0516, "25": 0.1079, "40": 0.340, "final": 2.494}
THRESHOLDS = (1, 2, 8, 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.path)
    meta = json.loads(str(z["meta_json"].item())) if "meta_json" in z.files else {}
    orders = meta.get("orders", [2, 3, 4, 5])
    print(f"file={args.path}")
    print(f"meta: tokens={meta.get('tokens', 0)/1e9:.3f}B docs={meta.get('docs', 0)} "
          f"rows={meta.get('rows', 0)} depths={meta.get('depths')} "
          f"final_block={meta.get('final_block')}")
    if meta.get("hit_rate"):
        print(f"hit rates: {meta['hit_rate']}")

    # ---- 1. counts -------------------------------------------------------
    for n in orders:
        cnt = z[f"cnt_{n}"]
        print(f"[count n={n}] rows={cnt.size} nonzero={int((cnt > 0).sum())} "
              f"({(cnt > 0).mean()*100:.1f}%) sum={int(cnt.sum()):,} "
              f"ge2={int((cnt >= 2).sum())} ge8={int((cnt >= 8).sum())} "
              f"ge64={int((cnt >= 64).sum())} max={int(cnt.max())}")

    # ---- 2. magnitude vs HF reference ------------------------------------
    sums = {}
    for k in z.files:
        if not k.startswith("sum_L"):
            continue
        depth, n = k[len("sum_L"):].rsplit("_", 1)
        sums.setdefault(depth, {})[int(n)] = k
    for depth in sorted(sums, key=lambda d: (d == "final", d)):
        ref = REF_RMS.get(depth)
        line = [f"[rms L{depth}] ref={ref}"]
        for thr in THRESHOLDS:
            acc = []
            for n, k in sorted(sums[depth].items()):
                cnt = z[f"cnt_{n}"]
                sel = cnt >= thr
                if not sel.any():
                    continue
                mean = z[k][sel] / cnt[sel].astype(np.float32)[:, None]
                # pooled RMS over the sampled rows of this order
                acc.append((int(sel.sum()), float(np.sqrt((mean ** 2).mean()))))
            if acc:
                rows = sum(a[0] for a in acc)
                rms = float(np.sqrt(sum(a[0] * a[1] ** 2 for a in acc) / rows))
                line.append(f"cnt>={thr}: {rms:.4g} ({rows} rows)")
        print("  ".join(line))

    # ---- 3. finiteness ---------------------------------------------------
    bad = []
    for k in z.files:
        if not k.startswith("sum_L"):
            continue
        cnt = z[f"cnt_{int(k.rsplit('_', 1)[1])}"]
        sel = cnt > 0
        if sel.any():
            blk = z[k][sel]
            if not np.isfinite(blk).all():
                bad.append(k)
    print("finiteness: " + ("OK (all stored sums finite)" if not bad else f"FAIL {bad}"))
    print("checks done")


if __name__ == "__main__":
    main()
