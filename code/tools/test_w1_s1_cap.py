#!/usr/bin/env python3
"""Candidate-cap tests for w1_make_s1 (S1 shortlist + heap generations).

Covers the --candidate-cap machinery without touching the sibling synth test:

  1. cap=0 vs cap=1000000 on the shared fixture: bit-identical rows
     (both are full-exact; cap >= leaf size disables the shortlist).
  2. cap=0 vs cap=256 on the tiny fixture: bit-identical rows (every leaf
     is smaller than the cap, so the shortlist never activates).
  3. A planted many-sibling case with cap=2: rows in bounds, occupied <= R,
     determinism across two runs, certified_best_frac reported in meta, and
     the shortlist contains the exact-best split at every step (recall 1)
     because one sibling has a drastically different continuation q.
  4. Brute-force reference: an independent greedy reimplementation that
     evaluates ALL siblings every step with merge_loss_batch reproduces the
     cap=0 rows bit-identically.

Run: python3 code/tools/test_w1_s1_cap.py
"""
import os
import sys
import json
import shutil
import tempfile
import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import w1_common
from w1_make_s1 import build, merge_loss_batch, TrieGrouping, ALPHA
from test_w1_s1_synth import build_fixture

VOCAB = 32


# ------------------------------------------------------- brute-force reference
def brute_force_rows(sib_cnt, sib_F, target):
    """Independent greedy child-vs-rest reference: evaluate ALL siblings of
    ALL leaves every step with merge_loss_batch; split the global argmax
    (ties -> lowest sibling id). Returns sib->leaf row assignment."""
    S = int(sib_F.size)
    lc = np.log(sib_cnt, where=sib_cnt > 0, out=np.zeros_like(sib_cnt))
    sib_h = (-(sib_cnt * lc).sum(axis=1)
             + sib_F * np.log(np.where(sib_F > 0, sib_F, 1.0)))
    leaf_cnt = [sib_cnt.sum(axis=0)]
    leaf_sibs = [np.arange(S, dtype=np.int32)]
    occupied = 1
    while occupied < target:
        best = None  # (gain, sib, lid)
        for lid, sibs in enumerate(leaf_sibs):
            if sibs.size < 2:
                continue
            gains = merge_loss_batch(sib_cnt[sibs], sib_F[sibs],
                                     leaf_cnt[lid],
                                     float(leaf_cnt[lid].sum()),
                                     child_h=sib_h[sibs])
            i = int(np.argmax(gains))  # first max -> lowest sib (sorted)
            if not np.isfinite(gains[i]):
                continue
            cand = (float(gains[i]), int(sibs[i]), lid)
            if best is None or (cand[0], -cand[1]) > (best[0], -best[1]):
                best = cand
        if best is None:
            break
        gain, s, lid = best
        sibs = leaf_sibs[lid]
        leaf_sibs[lid] = sibs[sibs != s]
        child_cnt = sib_cnt[s].copy()
        leaf_cnt[lid] = leaf_cnt[lid] - child_cnt
        leaf_cnt.append(child_cnt)
        leaf_sibs.append(np.array([s], dtype=np.int32))
        occupied += 1
    sib_leaf = np.empty(S, dtype=np.int32)
    for lid, sibs in enumerate(leaf_sibs):
        for s in sibs:
            sib_leaf[s] = lid
    return sib_leaf, occupied


def sib_counts_for_order(index_path, stats_dir, n, vocab):
    """Rebuild the exact sibling count matrices build() uses for order n
    (mirrors w1_make_s1.build; test-only duplicate to drive TrieGrouping
    and the brute-force reference directly)."""
    sel, _ = w1_common.load_selected(index_path)
    uni = np.load(os.path.join(stats_dir, "unigram.npy"))
    u = uni[0].astype(np.float64) + uni[1].astype(np.float64) + 1.0
    q_uni = u / u.sum()
    q_uni = np.maximum(q_uni, 1e-12)
    q_uni /= q_uni.sum()

    lower_sel_keys = None
    lower_q = None
    for m in w1_common.ORDERS:
        z = np.load(os.path.join(stats_dir, f"hits_n{m}.npz"))
        sel_idx, y, fold, suffix = z["sel_idx"], z["y"], z["fold"], z["suffix"]
        keys = sel[m][0]
        K = keys.size
        from w1_make_s1 import reduce_hits
        ctx, tok, cnt, F, ctx_suffix = reduce_hits(sel_idx, y, fold, suffix,
                                                   K, vocab)
        if m == 2:
            q = np.broadcast_to(q_uni, (K, vocab)).copy()
            pos_c = None
        else:
            pos = np.searchsorted(lower_sel_keys, ctx_suffix)
            pos_c = np.clip(pos, 0, lower_sel_keys.size - 1)
            q = lower_q[pos_c].copy()
        q *= ALPHA
        q[ctx, tok] += cnt
        q /= (F + ALPHA)[:, None]
        if m == n:
            sib_key, sib_id = np.unique(ctx_suffix, return_inverse=True)
            S = sib_key.size
            sib_cnt = np.zeros((S, vocab), dtype=np.float64)
            if m == 2:
                members = np.bincount(sib_id, minlength=S).astype(np.float64)
                sib_cnt[:] = members[:, None] * (ALPHA * q_uni)[None, :]
            else:
                np.add.at(sib_cnt, sib_id, ALPHA * lower_q[pos_c])
            np.add.at(sib_cnt, (sib_id[ctx], tok), cnt)
            sib_F = sib_cnt.sum(axis=1)
            return sib_cnt, sib_F, sib_id
        lower_sel_keys = keys
        lower_q = q
    raise AssertionError(f"order {n} not found")


# --------------------------------------------------------------------- tests
def test_cap0_vs_capbig_bitidentical(index_path, stats_dir, tmp):
    out_a = os.path.join(tmp, "cap0.npz")
    out_b = os.path.join(tmp, "capbig.npz")
    build(index_path, stats_dir, out_a, m_total=12, candidate_cap=0)
    build(index_path, stats_dir, out_b, m_total=12, candidate_cap=1000000)
    a, b = np.load(out_a), np.load(out_b)
    for n in w1_common.ORDERS:
        assert np.array_equal(a[f"rows_{n}"], b[f"rows_{n}"]), \
            f"n={n}: cap=0 vs cap=1000000 rows differ"
    print("[test] cap=0 vs cap=1000000 bit-identical OK")


def test_cap0_vs_cap256_bitidentical(index_path, stats_dir, tmp):
    out_a = os.path.join(tmp, "cap0b.npz")
    out_b = os.path.join(tmp, "cap256.npz")
    build(index_path, stats_dir, out_a, m_total=12, candidate_cap=0)
    build(index_path, stats_dir, out_b, m_total=12, candidate_cap=256)
    a, b = np.load(out_a), np.load(out_b)
    for n in w1_common.ORDERS:
        assert np.array_equal(a[f"rows_{n}"], b[f"rows_{n}"]), \
            f"n={n}: cap=0 vs cap=256 rows differ (fixture leaves < cap)"
    print("[test] cap=0 vs cap=256 bit-identical OK (cap >= leaf size)")


def _greedy_run(tg, target):
    """Drive a TrieGrouping through the same gen-counter heap loop build()
    uses. Returns (sib_leaf, occupied, split_sequence)."""
    import heapq
    S = tg.S
    heap = []
    fresh = set()
    c = tg.candidate(0)
    if c is not None:
        heapq.heappush(heap, (-c[0], c[1], 0, 0))
        fresh.add(0)
    occupied = 1
    seq = []
    while occupied < target and heap:
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
        seq.append((lid, s))
        new_lid = tg.split(lid, s)
        occupied += 1
        for l2 in (lid, new_lid):
            c2 = tg.candidate(l2)
            if c2 is not None:
                heapq.heappush(heap, (-c2[0], c2[1], l2, tg.leaf_gen[l2]))
                fresh.add(l2)
            else:
                fresh.discard(l2)
    sib_leaf = np.empty(S, dtype=np.int32)
    for lid, sibs in enumerate(tg.leaf_sibs):
        for s in sibs:
            sib_leaf[s] = lid
    return sib_leaf, occupied, seq


def test_cap2_planted_best():
    """Many siblings, cap=2, one drastically-different sibling plus exactly
    identical quiet siblings: the shortlist keeps the exact-best split at
    every step (recall 1): the capped split sequence and rows are identical
    to the full-exact run."""
    S = 40
    V = VOCAB
    # quiet siblings: EXACTLY identical q -> exact ties broken by lowest
    # sibling id in both full and capped modes -> identical sequences.
    sib_cnt = np.full((S, V), 100.0)
    # one planted outlier sibling with a drastically different q: its bound
    # and its exact gain are both the unique maximum -> shortlisted first.
    outlier = 17
    sib_cnt[outlier] = 1.0
    sib_cnt[outlier, 5] = 5000.0
    sib_F = sib_cnt.sum(axis=1)

    tg_full = TrieGrouping(sib_cnt, sib_F, candidate_cap=0,
                           exact_backend="numpy")
    rows_full, occ_full, seq_full = _greedy_run(tg_full, 8)
    tg_cap = TrieGrouping(sib_cnt, sib_F, candidate_cap=2,
                          exact_backend="numpy")
    rows_cap, occ_cap, seq_cap = _greedy_run(tg_cap, 8)

    # rows in bounds, occupied <= target
    assert rows_cap.min() >= 0 and rows_cap.max() < occ_cap
    assert np.unique(rows_cap).size == occ_cap
    assert occ_cap <= 8

    # determinism: a second capped run is identical
    tg_cap2 = TrieGrouping(sib_cnt, sib_F, candidate_cap=2,
                           exact_backend="numpy")
    rows_cap2, occ_cap2, _ = _greedy_run(tg_cap2, 8)
    assert occ_cap2 == occ_cap and np.array_equal(rows_cap2, rows_cap)

    # recall 1: the capped run reproduces the full-exact split sequence and
    # rows exactly (planted clear-best is always in the shortlist)
    assert seq_cap == seq_full, \
        f"cap=2 split sequence diverged: {seq_cap} vs {seq_full}"
    assert np.array_equal(rows_cap, rows_full), \
        "cap=2 shortlist lost the exact-best split on the planted case"
    # the outlier is peeled first (it is the clear best)
    assert seq_cap[0][1] == outlier

    # shortlist actually activated and certification counters are reported
    assert tg_cap.steps_total > 0
    assert tg_cap.steps_shortlisted > 0, "shortlist never activated"
    assert 0 <= tg_cap.steps_certified <= tg_cap.steps_total
    frac = tg_cap.steps_certified / tg_cap.steps_total
    assert 0.0 <= frac <= 1.0
    print(f"[test] cap=2 planted-best recall 1 OK "
          f"(steps={tg_cap.steps_total}, shortlisted={tg_cap.steps_shortlisted}, "
          f"certified={tg_cap.steps_certified}, frac={frac:.3f})")


def test_certificate_logic():
    """Certificate bookkeeping: steps_total/steps_shortlisted/steps_certified
    are consistent with a manual recomputation of bounds and gains."""
    V = VOCAB
    A = np.full(V, 100000.0)          # dominates the leaf
    B = np.full(V, 1.0); B[3] = 400.0  # small, very different q (best)
    C = np.full(V, 5000.0); C[9] += 2000.0
    sib_cnt = np.stack([A, B, C])
    sib_F = sib_cnt.sum(axis=1)
    tg = TrieGrouping(sib_cnt, sib_F, candidate_cap=2, exact_backend="numpy")
    best_gain, best_sib = tg.candidate(0)
    assert tg.steps_total == 1 and tg.steps_shortlisted == 1
    # manual: bounds, shortlist (top-2 by bound, ties -> lowest sib),
    # excluded max bound, certificate predicate
    pcnt = tg.leaf_cnt[0]
    Ft = tg.leaf_F[0]
    lp = np.log(pcnt, where=pcnt > 0, out=np.zeros_like(pcnt))
    HP = float(-(pcnt * lp).sum()) + Ft * np.log(Ft)
    bounds = HP - tg.sib_h
    order = np.lexsort((np.arange(3), -bounds))
    short = np.sort(order[:2])
    excl = np.setdiff1d(np.arange(3), short)
    gains = merge_loss_batch(sib_cnt, sib_F, pcnt, Ft, child_h=tg.sib_h)
    go = np.lexsort((short, -gains[short]))
    exp_sib = int(short[go[0]])
    exp_gain = float(gains[short][go[0]])
    assert best_sib == exp_sib and abs(best_gain - exp_gain) < 1e-9
    max_excl = float(bounds[excl].max()) if excl.size else -np.inf
    exp_cert = 1 if exp_gain >= max_excl else 0
    assert tg.steps_certified == exp_cert, \
        f"certificate mismatch: got {tg.steps_certified}, want {exp_cert}"
    # full-exact run agrees on the best sibling
    tg0 = TrieGrouping(sib_cnt, sib_F, candidate_cap=0, exact_backend="numpy")
    g0, s0 = tg0.candidate(0)
    assert s0 == exp_sib and abs(g0 - exp_gain) < 1e-9
    assert tg0.steps_certified == 1 and tg0.steps_shortlisted == 0
    print(f"[test] certificate logic OK (fired={bool(exp_cert)})")


def test_bruteforce_reference(index_path, stats_dir, tmp):
    """cap=0 TrieGrouping rows == independent brute-force greedy rows."""
    out = os.path.join(tmp, "cap0_ref.npz")
    m_total = 12
    build(index_path, stats_dir, out, m_total=m_total, candidate_cap=0)
    o = np.load(out)
    for n in w1_common.ORDERS:
        sib_cnt, sib_F, sib_id = sib_counts_for_order(index_path, stats_dir,
                                                      n, VOCAB)
        R_n = int(o[f"R_{n}"])
        sib_leaf, occ = brute_force_rows(sib_cnt, sib_F, R_n)
        ref_rows = sib_leaf[sib_id].astype(np.int32)
        got = o[f"rows_{n}"]
        assert occ == int(json.loads(str(o["meta_json"]))[f"occupied_{n}"]), \
            f"n={n}: occupied mismatch vs brute force"
        assert np.array_equal(got, ref_rows), \
            f"n={n}: cap=0 rows differ from brute-force reference"
    print("[test] brute-force reference == cap=0 rows bit-identical OK")


def test_certified_frac_in_meta(index_path, stats_dir, tmp):
    out = os.path.join(tmp, "cap_meta.npz")
    build(index_path, stats_dir, out, m_total=12, candidate_cap=2)
    meta = json.loads(str(np.load(out)["meta_json"]))
    for n in w1_common.ORDERS:
        assert f"certified_best_frac_{n}" in meta
        assert f"steps_total_{n}" in meta
        assert f"steps_certified_{n}" in meta
        assert f"steps_shortlisted_{n}" in meta
        frac = meta[f"certified_best_frac_{n}"]
        assert 0.0 <= frac <= 1.0
    assert meta["candidate_cap"] == 2
    assert meta["exact_backend"] in ("numpy", "torch")
    print("[test] certified_best_frac + step counters in meta OK")


def main():
    tmp = tempfile.mkdtemp(prefix="w1_s1_cap_", dir="/tmp")
    try:
        index_path, stats_dir, ctxs, fam = build_fixture(tmp)
        test_cap0_vs_capbig_bitidentical(index_path, stats_dir, tmp)
        test_cap0_vs_cap256_bitidentical(index_path, stats_dir, tmp)
        test_cap2_planted_best()
        test_certificate_logic()
        test_bruteforce_reference(index_path, stats_dir, tmp)
        test_certified_frac_in_meta(index_path, stats_dir, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL CAP TESTS PASSED")


if __name__ == "__main__":
    main()
