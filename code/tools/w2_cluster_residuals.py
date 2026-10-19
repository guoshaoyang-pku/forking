#!/usr/bin/env python3
"""W2 residual clustering (user directive 2026-10-19, route (b)): traffic
-weighted two-level k-means on per-context mean residuals from
w2_dump_residuals.py, producing per-layer assignment npz (train.py
--ngram_layer_assign) + centroid init npz (--ngram_layer_init).

Features: X(c) = sum_n[c] / cnt_n[c]  (mean residual at the injection point,
NOT unit-normed -- weak-signal contexts keep their small norm, v2 section 3.4).
Weights:  w(c) = cnt_n[c]             (token-level traffic, C1 convention).

Scalability (v2 section 6): ~COARSE_K coarse groups via weighted k-means ->
per-group local weighted k-means with local K by deterministic largest
remainder of group weight; sum(local K) == R_n exactly (shortfall reported,
never silently padded). No N x R matrix is ever materialized.

Budget: --budget is the TOTAL rows for this layer, split across orders by
largest remainder proportional to #selected contexts (w1_common.allocate_R).

Load balance: measured and reported (L_j = sum of member traffic, entropy,
max/mean, >4x-mean share) -- soft constraint only (v2 section 7.3); single
ultra-high-frequency contexts are exempted, never force-split.

CLI:
  python3 w2_cluster_residuals.py <resid_L.npz> <out_assign.npz> <out_init.npz> \
      --budget 325000 [--seed 0xC1EA00D5]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w1_common import ORDERS, ROUTING_SEED, write_assignment  # noqa: E402

COARSE_K = 256
KM_ITERS_COARSE = 25
KM_ITERS_LOCAL = 30
CHUNK = 65536


def wkmeans(X, w, K, iters, seed, device):
    """Weighted Lloyd k-means, L2 geometry, weighted-mean centers.
    X (N,D) torch fp32, w (N,) fp32. Returns (assign (N,) long, C (K,D))."""
    N, D = X.shape
    K = min(K, N)
    g = torch.Generator(device="cpu").manual_seed(seed)
    # init: weight-proportional sampling without replacement
    idx = torch.multinomial(w.cpu(), K, replacement=False, generator=g).to(device)
    C = X[idx].clone()
    assign = torch.zeros(N, dtype=torch.long, device=device)
    for _ in range(iters):
        for s in range(0, N, CHUNK):
            d = torch.cdist(X[s:s + CHUNK], C)  # (b, K)
            assign[s:s + CHUNK] = d.argmin(dim=1)
        newC = torch.zeros_like(C)
        wsum = torch.zeros(K, dtype=torch.float32, device=device)
        wsum.index_add_(0, assign, w)
        newC.index_add_(0, assign, X * w.unsqueeze(1))
        live = wsum > 0
        newC[live] /= wsum[live].unsqueeze(1)
        newC[~live] = C[~live]
        if torch.equal(newC, C):
            break
        C = newC
    for s in range(0, N, CHUNK):
        d = torch.cdist(X[s:s + CHUNK], C)
        assign[s:s + CHUNK] = d.argmin(dim=1)
    return assign, C


def cluster_order(X, w, R, seed, device):
    """Two-level weighted k-means: coarse COARSE_K groups -> local k-means."""
    N = X.shape[0]
    R = min(R, N)
    coarseK = min(COARSE_K, R)
    cass, _ = wkmeans(X, w, coarseK, KM_ITERS_COARSE, seed, device)
    rows = torch.empty(N, dtype=torch.long, device=device)
    # local K by largest remainder of group WEIGHT
    gw = torch.zeros(coarseK, dtype=torch.float32, device=device)
    gw.index_add_(0, cass, w)
    raw = R * gw / gw.sum()
    localK = raw.floor().long()
    rem = int(R - localK.sum().item())
    order = torch.argsort(raw - localK.float(), descending=True).cpu()
    li = 0
    while rem > 0:
        gidx = int(order[li % coarseK])
        gsz = int((cass == gidx).sum().item())
        if localK[gidx] < gsz:
            localK[gidx] += 1
            rem -= 1
        li += 1
    base = 0
    shortfall = 0
    for gidx in range(coarseK):
        m = cass == gidx
        nz = int(m.sum().item())
        if nz == 0:
            continue
        lk = int(localK[gidx].item())
        if lk == 0:
            shortfall += 1
            # still must assign: merge into the single nearest used cluster
            lk = 1
        sub = m.nonzero(as_tuple=True)[0]
        if nz <= lk:
            rows[sub] = torch.arange(base, base + nz, device=device)
            base += nz
            continue
        lass, _ = wkmeans(X[sub], w[sub], lk, KM_ITERS_LOCAL,
                          seed ^ (0x9E3779B9 * (gidx + 1)), device)
        rows[sub] = lass + base
        base += lk
    return rows, base, shortfall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resid_npz")
    ap.add_argument("out_assign")
    ap.add_argument("out_init")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=ROUTING_SEED)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(args.resid_npz)
    sel, feats, weights = {}, {}, {}
    for n in ORDERS:
        k = z[f"keys_{n}"].astype(np.int64)
        cnt = z[f"cnt_{n}"].astype(np.int64)
        live = cnt > 0
        X = z[f"sum_{n}"].astype(np.float32)
        X = (X[live] / np.maximum(cnt[live], 1)[:, None]).astype(np.float32)
        sel[n] = (np.ascontiguousarray(k[live]), cnt[live])
        feats[n] = X
        weights[n] = cnt[live].astype(np.float64)

    # budget split across orders by largest remainder on #selected contexts
    sizes = {n: sel[n][0].size for n in ORDERS}
    tot = sum(sizes.values())
    raw = {n: args.budget * sizes[n] / tot for n in ORDERS}
    R = {n: max(1, min(int(np.floor(raw[n])), sizes[n])) for n in ORDERS}
    rem = args.budget - sum(R.values())
    fo = sorted(ORDERS, key=lambda n: raw[n] - np.floor(raw[n]), reverse=True)
    i = 0
    while rem > 0:
        n = fo[i % len(fo)]
        if R[n] < sizes[n]:
            R[n] += 1
            rem -= 1
        i += 1

    rows_map, init_map, lb_report = {}, {}, {}
    t0 = time.time()
    for n in ORDERS:
        X = torch.from_numpy(feats[n]).to(device)
        w = torch.from_numpy(weights[n].astype(np.float32)).to(device)
        rows, used, shortfall = cluster_order(X, w, R[n],
                                              args.seed ^ (0x9E3779B9 * n),
                                              device)
        rows_map[n] = rows.cpu().numpy().astype(np.int32)
        # centroids (weighted means) for init
        C = torch.zeros(used, X.shape[1], dtype=torch.float32, device=device)
        ws = torch.zeros(used, dtype=torch.float32, device=device)
        ws.index_add_(0, rows, w)
        C.index_add_(0, rows, X * w.unsqueeze(1))
        C /= ws.clamp_min(1e-12).unsqueeze(1)
        init_map[n] = C.cpu().numpy()
        # load-balance report (traffic per physical row)
        Lj = torch.zeros(used, dtype=torch.float64, device=device)
        Lj.index_add_(0, rows, w.double())
        Lj = Lj.cpu().numpy()
        total = Lj.sum()
        p = Lj / max(total, 1e-12)
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        mean_l = Lj.mean()
        lb_report[n] = dict(K=int(sizes[n]), R=int(used), budget=int(R[n]),
                            shortfall_groups=int(shortfall),
                            load_entropy=round(ent, 3),
                            load_max_over_mean=round(float(Lj.max() / mean_l), 2),
                            load_over4x_share=round(float(Lj[Lj > 4 * mean_l].sum() / total), 4),
                            rows_used=int(used))
        print(f"[w2] n={n}: K={sizes[n]:,} -> R={used:,} "
              f"ent={ent:.2f} max/mean={Lj.max() / mean_l:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    counts_total = int(sum(weights[n].sum() for n in ORDERS))
    R_actual = {n: int(lb_report[n]["rows_used"]) for n in ORDERS}
    meta = {"algo": "w2resid", "src": args.resid_npz, "seed": args.seed,
            "budget": args.budget, "coarse_k": COARSE_K,
            "load_balance": lb_report}
    write_assignment(args.out_assign, sel, R_actual, rows_map, counts_total, meta)
    out_init = {f"table_{n}": init_map[n].astype(np.float32) for n in ORDERS}
    np.savez(args.out_init, **out_init)
    print(f"[w2] wrote {args.out_assign} + {args.out_init}", flush=True)
    print(json.dumps(lb_report, indent=1), flush=True)


if __name__ == "__main__":
    main()
