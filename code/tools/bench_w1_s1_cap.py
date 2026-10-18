#!/usr/bin/env python3
"""Real-scale synthetic benchmark for the w1_make_s1 candidate cap.

Profile target (order n=3 of the 360-2 run): K=400k contexts, vocab 8192,
S=100k siblings, R=50k target rows. Measures per-split wall time over
~200 splits with the production heap loop, then projects the full n=3 cost
(R_3 = 54916 splits) for numpy and torch backends.

This is a benchmark script, not a unit test. Run:
    python3 code/tools/bench_w1_s1_cap.py [n_splits] [backend]
"""
import os
import sys
import time
import heapq
import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from w1_make_s1 import TrieGrouping

VOCAB = 8192
S = 100_000
K = 400_000
R_TARGET = 50_000
R_3_REAL = 54916          # actual n=3 row budget of the 360-2 allocation
CAP = 256


def make_synthetic(S, V, seed=0):
    """Sibling count matrix with a realistic heavy-tailed F distribution and
    sparse concentrated continuations (Zipf-ish), float64 smoothed counts.
    Fully vectorized: each sibling puts its count mass on NDIST tokens drawn
    from a shared Zipf-ish popularity (per-row offset for variety)."""
    rng = np.random.default_rng(seed)
    # heavy-tailed sibling totals
    F = rng.pareto(1.2, size=S) + 1.0
    F *= 1e4 / F.mean()
    NDIST = 64
    pop = 1.0 / np.arange(1, V + 1, dtype=np.float64)
    pop /= pop.sum()
    # (S, NDIST) token ids: popularity-biased base + per-row random offset
    base = rng.choice(V, size=(S, NDIST), replace=True, p=pop)
    off = rng.integers(0, V, size=(S, 1))
    toks = (base + off) % V
    # Dirichlet-ish weights via normalized exponentials
    w = rng.gamma(0.6, 1.0, size=(S, NDIST))
    w /= w.sum(axis=1, keepdims=True)
    mass = F[:, None] * w
    sib_cnt = np.full((S, V), 128.0 / V, dtype=np.float64)  # backoff mass
    rows = np.repeat(np.arange(S), NDIST)
    np.add.at(sib_cnt, (rows, toks.ravel()), mass.ravel())
    sib_F = sib_cnt.sum(axis=1)
    return sib_cnt, sib_F


def run_splits(sib_cnt, sib_F, cap, backend, n_splits):
    tg = TrieGrouping(sib_cnt, sib_F, candidate_cap=cap, exact_backend=backend)
    heap = []
    fresh = set()
    c = tg.candidate(0)
    if c is not None:
        heapq.heappush(heap, (-c[0], c[1], 0, 0))
        fresh.add(0)
    occupied = 1
    times = []
    t_all = time.time()
    while occupied < n_splits and heap:
        t_sp = time.time()
        neggain, s, lid, gen = heapq.heappop(heap)
        if lid >= len(tg.leaf_cnt) or s not in tg.leaf_sibset[lid]:
            fresh.discard(lid)
            continue
        if gen != tg.leaf_gen[lid]:
            if lid in fresh:
                continue
            c = tg.candidate(lid)
            if c is None:
                continue
            heapq.heappush(heap, (-c[0], c[1], lid, tg.leaf_gen[lid]))
            fresh.add(lid)
            continue
        fresh.discard(lid)
        new_lid = tg.split(lid, s)
        occupied += 1
        for l2 in (lid, new_lid):
            c2 = tg.candidate(l2)
            if c2 is not None:
                heapq.heappush(heap, (-c2[0], c2[1], l2, tg.leaf_gen[l2]))
                fresh.add(l2)
            else:
                fresh.discard(l2)
        times.append(time.time() - t_sp)
    total = time.time() - t_all
    return tg, occupied, np.array(times), total


def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    backends = sys.argv[2].split(",") if len(sys.argv) > 2 else ["numpy"]
    print(f"[bench] building synthetic S={S} vocab={VOCAB} ...", flush=True)
    t0 = time.time()
    sib_cnt, sib_F = make_synthetic(S, VOCAB)
    print(f"[bench] synthetic ready in {time.time()-t0:.1f}s "
          f"({sib_cnt.nbytes/1e9:.2f} GB)", flush=True)

    results = {}
    for backend in backends:
        tg, occ, times, total = run_splits(sib_cnt, sib_F, CAP, backend,
                                           n_splits)
        # per-split cost model: each split triggers candidate() on the two
        # resulting leaves; candidate cost ~ O(S_leaf + C*vocab). Early
        # splits hit the big root leaf; steady state has smaller leaves.
        # Projection: scale the MEASURED mean per-split time to R_3_REAL.
        mean_sp = float(times.mean())
        proj_s = mean_sp * R_3_REAL
        results[backend] = (mean_sp, proj_s, tg)
        print(f"[bench] backend={backend} cap={CAP} splits={occ} "
              f"mean={mean_sp*1e3:.1f} ms/split median={np.median(times)*1e3:.1f} "
              f"p90={np.quantile(times,0.9)*1e3:.1f} total={total:.1f}s",
              flush=True)
        print(f"[bench] backend={backend} projection n=3 "
              f"({R_3_REAL} splits): {proj_s/60:.1f} min", flush=True)
        print(f"[bench] backend={backend} steps_total={tg.steps_total} "
              f"shortlisted={tg.steps_shortlisted} "
              f"certified={tg.steps_certified}", flush=True)

    for b, (m, p, _tg) in results.items():
        status = "OK" if p < (30 if b == "numpy" else 10) * 60 else "MISS"
        budget = 30 if b == "numpy" else 10
        print(f"[bench] {b}: {p/60:.1f} min projected vs {budget} min "
              f"budget -> {status}", flush=True)


if __name__ == "__main__":
    main()
