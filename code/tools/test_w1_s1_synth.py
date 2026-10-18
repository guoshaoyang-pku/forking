#!/usr/bin/env python3
"""Synthetic-data unit test for w1_make_s1 (S1 suffix-structured grouping).

Builds a tiny stats dir + freq index npz (vocab 32, orders 2..5, ~200
contexts) with planted structure: two families of contexts with
(near-)identical continuation distributions but DIFFERENT suffixes, plus
high-distortion outlier sibling groups. Asserts:

  1. identical-q contexts merge (share a row when the budget is tight)
  2. merge loss of two identical q == 0; of distinct q > 0
  3. rows in bounds; every selected context has exactly one row;
     occupied <= R_n
  4. determinism: two runs produce identical outputs
  5. suffix closure: every suffix maps into the order-(n-1) selected set

Run: python3 code/tools/test_w1_s1_synth.py
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
from w1_make_s1 import build, js_merge_loss, merge_loss_from_counts, ALPHA

VOCAB = 32
FMIN = w1_common.FMIN
FIT_PER_CTX = 240          # fold-0 + fold-1 hits per selected context
DEV_PER_CTX = 20           # fold-2 hits per selected context
FAM_TOK = 20               # shared continuation token of the quiet families


def pack_key(tokens):
    """tokens = (first, ..., last) -> base-VOCAB key, last token = k=0."""
    k = 0
    for i, t in enumerate(reversed(tokens)):
        k += int(t) * (VOCAB ** i)
    return k


def make_hits(ctx_list, y_of_ctx, seed_rng):
    """Fabricate hits arrays for one order.

    ctx_list: list of (key, suffix_key, first_tok).
    y_of_ctx: function key -> continuation token for fit hits.
    Each context gets FIT_PER_CTX fit hits (half fold 0, half fold 1) on
    y_of_ctx(key) and DEV_PER_CTX fold-2 hits on the same token.
    """
    sel_idx, y, fold, first, suffix = [], [], [], [], []
    for i, (key, suf, ft) in enumerate(ctx_list):
        yy = y_of_ctx(key)
        nf0 = FIT_PER_CTX // 2
        nf1 = FIT_PER_CTX - nf0
        sel_idx += [i] * (nf0 + nf1 + DEV_PER_CTX)
        y += [yy] * (nf0 + nf1 + DEV_PER_CTX)
        fold += [0] * nf0 + [1] * nf1 + [2] * DEV_PER_CTX
        first += [ft] * (nf0 + nf1 + DEV_PER_CTX)
        suffix += [suf] * (nf0 + nf1 + DEV_PER_CTX)
    keys = np.array([c[0] for c in ctx_list], dtype=np.int64)
    order = np.argsort(keys)
    assert (keys[order] == np.sort(keys)).all()
    return {
        "sel_idx": np.array(sel_idx, dtype=np.int32),
        "y": np.array(y, dtype=np.uint16),
        "fold": np.array(fold, dtype=np.uint8),
        "first_tok": np.array(first, dtype=np.uint16),
        "suffix": np.array(suffix, dtype=np.int64),
        "keys": keys,
    }


def build_fixture(root):
    """Write index npz + stats dir under root. Returns (index_path, stats_dir,
    ctxs dict n -> ctx_list, fam dict with planted family member indices)."""
    stats_dir = os.path.join(root, "stats")
    os.makedirs(stats_dir)
    rng = np.random.default_rng(0)

    # ---------------- order 2: bigrams (a,b), key = a*V + b, suffix = b ----
    # quiet families X (suffix 3) and Y (suffix 7): identical continuations
    # (all fit hits on FAM_TOK) but different suffix tokens -> different
    # sibling groups. 4 outlier sibling groups with distinct, concentrated
    # continuations (high merge loss -> peeled first under a tight budget).
    famX_first = [1, 2, 4, 5, 6]
    famY_first = [8, 9, 10, 11, 12]
    out_suffixes = [11, 15, 19, 23]          # outlier last-tokens
    out_firsts = [3, 13, 14, 17, 18, 21]     # several bigrams per outlier group
    filler_suffixes = [0, 1, 2, 5, 9, 27]    # extra quiet-ish sibling groups
    ctx2 = []
    for a in famX_first:
        ctx2.append((pack_key((a, 3)), 3, a))
    for a in famY_first:
        ctx2.append((pack_key((a, 7)), 7, a))
    for s in out_suffixes:
        for a in out_firsts:
            ctx2.append((pack_key((a, s)), s, a))
    for s in filler_suffixes:
        for a in (22, 24, 26):
            ctx2.append((pack_key((a, s)), s, a))
    ctx2 = sorted(set(ctx2), key=lambda c: c[0])
    key2 = np.array([c[0] for c in ctx2], dtype=np.int64)
    idxX = [i for i, c in enumerate(ctx2) if c[1] == 3]
    idxY = [i for i, c in enumerate(ctx2) if c[1] == 7]
    idxOut = [i for i, c in enumerate(ctx2) if c[1] in out_suffixes]

    def y2(key):
        b = key % VOCAB
        if b in (3, 7):
            return FAM_TOK                       # quiet families: identical q
        if b in out_suffixes:
            # distinct concentrated token per outlier group, far from FAM_TOK
            return 24 + out_suffixes.index(b)
        return (key // VOCAB + b) % VOCAB        # filler: varied
    hits2 = make_hits(ctx2, y2, rng)

    # ---------------- order 3: trigrams, suffix in selected bigrams --------
    # pick a subset of bigrams as suffixes; plant the same quiet/outlier
    # structure one level up (quiet trigram families share FAM_TOK).
    suf_pool = [c for c in ctx2 if c[1] in (3, 7)][:8]   # quiet bigram suffixes
    out_pool = [c for c in ctx2 if c[1] in out_suffixes][:3]
    ctx3 = []
    for (skey, _suf, _ft) in suf_pool:
        for a in (2, 9, 16):
            ctx3.append((pack_key((a, skey // VOCAB, skey % VOCAB)), skey, a))
    for (skey, _suf, _ft) in out_pool:
        for a in (5, 12):
            ctx3.append((pack_key((a, skey // VOCAB, skey % VOCAB)), skey, a))
    ctx3 = sorted(set(ctx3), key=lambda c: c[0])
    key3 = np.array([c[0] for c in ctx3], dtype=np.int64)
    quiet3_suf = set(c[0] for c in suf_pool)
    out3_suf = [c[0] for c in out_pool]

    def y3(key):
        skey = key % (VOCAB ** 2)
        if skey in quiet3_suf:
            return FAM_TOK
        if skey in out3_suf:
            return 24 + out3_suf.index(skey)     # distinct per outlier group
        return (skey + 3) % VOCAB
    hits3 = make_hits(ctx3, y3, rng)

    # ---------------- orders 4,5: minimal closed chains --------------------
    suf4 = [ctx3[0], ctx3[1]]
    ctx4 = sorted({(pack_key((7, c[0] // VOCAB ** 2, (c[0] // VOCAB) % VOCAB,
                             c[0] % VOCAB)), c[0], 7) for c in suf4},
                  key=lambda c: c[0])
    key4 = np.array([c[0] for c in ctx4], dtype=np.int64)
    hits4 = make_hits(ctx4, lambda k: 4, rng)
    # n=5 keys are splitmix64 hashes in the real pipeline; the tool never
    # decodes keys, so arbitrary distinct sorted int64 stand in.
    suf5 = [ctx4[0], ctx4[1]]
    raw5 = sorted(rng.integers(1, 1 << 62, size=len(suf5), dtype=np.int64).tolist())
    ctx5 = [(int(raw5[i]), suf5[i][0], suf5[i][2]) for i in range(len(suf5))]
    key5 = np.array([c[0] for c in ctx5], dtype=np.int64)
    hits5 = make_hits(ctx5, lambda k: 6, rng)

    # ---------------- index npz --------------------------------------------
    index = {}
    for n, keys in ((2, key2), (3, key3), (4, key4), (5, key5)):
        # one unselected low-freq key per order (filtering check)
        extra = np.array([keys.max() + 1], dtype=np.int64)
        index[f"keys_{n}"] = np.concatenate([keys, extra])
        index[f"counts_{n}"] = np.concatenate([
            np.full(keys.size, FIT_PER_CTX + DEV_PER_CTX, dtype=np.int32),
            np.array([FMIN - 1], dtype=np.int32)])
    index_path = os.path.join(root, "index.npz")
    np.savez(index_path, **index)

    # ---------------- stats dir ---------------------------------------------
    for n, hits in ((2, hits2), (3, hits3), (4, hits4), (5, hits5)):
        np.savez(os.path.join(stats_dir, f"hits_n{n}.npz"), **hits)
    uni = np.zeros((3, VOCAB), dtype=np.int64)
    for f in (0, 1, 2):
        uni[f] = np.bincount(hits2["y"][hits2["fold"] == f], minlength=VOCAB)
    np.save(os.path.join(stats_dir, "unigram.npy"), uni)

    fam = {"idxX": idxX, "idxY": idxY, "idxOut": idxOut,
           "quiet3_suf": quiet3_suf, "ctx2": ctx2, "ctx3": ctx3}
    return index_path, stats_dir, {2: ctx2, 3: ctx3, 4: ctx4, 5: ctx5}, fam


# ------------------------------------------------------------------ helpers
def rows_by_key(out, n):
    return {int(k): int(r) for k, r in zip(out[f"keys_{n}"], out[f"rows_{n}"])}


def run_build(index_path, stats_dir, out_path, m_total):
    return build(index_path, stats_dir, out_path, m_total=m_total)


# --------------------------------------------------------------------- tests
def test_merge_loss_identical_vs_distinct():
    q = np.full(VOCAB, 1.0 / VOCAB)
    p = np.full(VOCAB, 1.0 / VOCAB)
    p[0] = 0.5
    p[1:] = 0.5 / (VOCAB - 1)
    d_same = js_merge_loss(q, 100.0, q.copy(), 200.0)
    d_diff = js_merge_loss(q, 100.0, p, 200.0)
    assert abs(d_same) < 1e-12, f"identical q merge loss {d_same} != 0"
    assert d_diff > 1e-6, f"distinct q merge loss {d_diff} <= 0"
    # scalar vs batch consistency is exercised implicitly via the grouping
    print("[test] merge loss identical==0, distinct>0 OK")


def test_fixture_closure(ctxs):
    keys = {n: np.array([c[0] for c in ctxs[n]], dtype=np.int64) for n in ctxs}
    for n in (3, 4, 5):
        lower = set(keys[n - 1].tolist())
        for (_k, suf, _ft) in ctxs[n]:
            assert int(suf) in lower, f"n={n}: suffix {suf} not in selected n-1"
    print("[test] suffix closure OK")


def test_s1_properties(index_path, stats_dir, ctxs, fam, tmp):
    out1 = os.path.join(tmp, "s1_a.npz")
    out2 = os.path.join(tmp, "s1_b.npz")
    m_total = 12
    rows_map, meta = run_build(index_path, stats_dir, out1, m_total)
    rows_map2, meta2 = run_build(index_path, stats_dir, out2, m_total)

    o1 = np.load(out1)
    o2 = np.load(out2)

    # ---- determinism: identical arrays + meta (modulo elapsed time) -------
    for k in o1.files:
        a, b = o1[k], o2[k]
        if k == "meta_json":
            ja, jb = json.loads(str(a)), json.loads(str(b))
            ja.pop("elapsed_s", None)
            jb.pop("elapsed_s", None)
            assert ja == jb, "meta_json mismatch across runs"
        else:
            assert np.array_equal(a, b), f"array {k} differs across runs"
    print("[test] determinism OK")

    R = {n: int(o1[f"R_{n}"]) for n in w1_common.ORDERS}
    for n in w1_common.ORDERS:
        rows = o1[f"rows_{n}"]
        keys = o1[f"keys_{n}"]
        occ = int(meta[f"occupied_{n}"])
        # ---- rows in bounds, one row per context, occupied <= R_n ---------
        assert rows.shape == keys.shape and rows.dtype == np.int32
        assert rows.min() >= 0 and rows.max() < occ, f"n={n} rows out of bounds"
        assert np.unique(rows).size == occ, f"n={n} unused rows present"
        assert occ <= R[n], f"n={n} occupied {occ} > R {R[n]}"
    print("[test] row bounds / coverage / occupied<=R OK")

    # ---- planted structure: quiet families merge under tight budget -------
    assert R[2] < len({c[1] for c in ctxs[2]}), "test needs budget < #siblings"
    r2 = rows_by_key(o1, 2)
    rowsX = {r2[ctxs[2][i][0]] for i in fam["idxX"]}
    rowsY = {r2[ctxs[2][i][0]] for i in fam["idxY"]}
    rowsOut = {r2[ctxs[2][i][0]] for i in fam["idxOut"]}
    assert len(rowsX) == 1 and len(rowsY) == 1, "family internally split"
    assert rowsX == rowsY, f"identical-q families not merged: {rowsX} vs {rowsY}"
    assert not (rowsX & rowsOut), "outliers merged with quiet families"
    print("[test] identical-q families share a row; outliers split OK")

    # ---- order 3: quiet trigram families (shared FAM_TOK) merge -----------
    r3 = rows_by_key(o1, 3)
    q3 = [c for c in ctxs[3] if c[1] in fam["quiet3_suf"]]
    o3 = [c for c in ctxs[3] if c[1] not in fam["quiet3_suf"]]
    rows_q3 = {r3[c[0]] for c in q3}
    rows_o3 = {r3[c[0]] for c in o3}
    assert len(rows_q3) == 1, f"quiet trigram families split: {rows_q3}"
    assert not (rows_q3 & rows_o3), "order-3 outliers merged with quiet"
    print("[test] order-3 quiet families share a row OK")

    # ---- grouped dev NLL >= exact dev NLL (grouping can only lose info) ---
    for n in w1_common.ORDERS:
        ng, ne = meta[f"dev_nll_grouped_{n}"], meta[f"dev_nll_exact_{n}"]
        assert ng >= ne - 1e-9, f"n={n}: grouped NLL {ng} < exact {ne}"
    print("[test] dev NLL grouped >= exact OK")

    print("[test] meta:", json.dumps({k: v for k, v in meta.items()
                                      if k != "elapsed_s"}, sort_keys=True))


def main():
    tmp = tempfile.mkdtemp(prefix="w1_s1_synth_", dir="/tmp")
    try:
        index_path, stats_dir, ctxs, fam = build_fixture(tmp)
        test_merge_loss_identical_vs_distinct()
        test_fixture_closure(ctxs)
        test_s1_properties(index_path, stats_dir, ctxs, fam, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
