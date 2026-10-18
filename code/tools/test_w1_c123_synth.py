#!/usr/bin/env python3
"""Synthetic-data unit test for w1_make_c123 (C1/C2/C3 clustering routers).

Builds a tiny stats dir + freq index npz (vocab 32, orders 2..5, ~2000
contexts) with planted cluster families: groups of contexts whose
continuation distributions are drawn from a small set of family prototypes
(so PROB-space clustering should recover them), plus filler noise contexts.

Asserts (plan v2 section 6 unit-test list, adapted to C1/C2/C3):
  1. q rows finite, non-negative, sum to 1 (within 1e-6); unseen-suffix
     backoff correct (a context with zero fit counts backs off to its suffix)
  2. merge loss of identical q == 0, of distinct q > 0
  3. merge loss equals the direct before/after fit-NLL difference (toy)
  4. C3 exact update never increases J across accepted moves (monotone/sweep)
  5. sum(local K) == R_n exactly (or unused budget reported)
  6. determinism: two runs bit-identical
  7. random-label shuffle control: clustering planted families recovers
     >95% family purity; permuted labels do not

Run: python3 code/tools/test_w1_c123_synth.py
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
import w1_make_c123 as W

VOCAB = 32
FMIN = w1_common.FMIN
N_FAM = 6                  # planted families
FAM_SIZE2 = 40             # contexts per family at order 2  (240)
FAM_SIZE3 = 45             # contexts per family at order 3  (270)
FILL2 = 700                # filler contexts order 2 (vocab^2=1024 caps total)
FILL3 = 1730               # filler contexts order 3  (~2000/order)
FIT_PER_CTX = 240
DEV_PER_CTX = 20


def pack_key(tokens):
    k = 0
    for i, t in enumerate(reversed(tokens)):
        k += int(t) * (VOCAB ** i)
    return k


def family_prototypes(rng):
    """N_FAM sparse, well-separated continuation prototypes over VOCAB."""
    protos = np.zeros((N_FAM, VOCAB))
    toks = rng.permutation(VOCAB)[:N_FAM * 3].reshape(N_FAM, 3)
    for f in range(N_FAM):
        protos[f, toks[f]] = [0.7, 0.2, 0.1]
    return protos


def make_hits(ctx_list, q_of_key, rng):
    """Fabricate hits arrays. ctx_list: (key, suffix_key, first_tok).
    q_of_key: key -> (VOCAB,) continuation distribution for fit hits.
    Dev hits are drawn from the same distribution."""
    sel_idx, y, fold, first, suffix = [], [], [], [], []
    for i, (key, suf, ft) in enumerate(ctx_list):
        q = q_of_key(key)
        nf0 = FIT_PER_CTX // 2
        nf1 = FIT_PER_CTX - nf0
        nd = DEV_PER_CTX
        draws0 = rng.choice(VOCAB, size=nf0, p=q)
        draws1 = rng.choice(VOCAB, size=nf1, p=q)
        drawsd = rng.choice(VOCAB, size=nd, p=q)
        yy = np.concatenate([draws0, draws1, drawsd])
        ff = np.array([0] * nf0 + [1] * nf1 + [2] * nd, dtype=np.uint8)
        sel_idx += [i] * yy.size
        y += yy.tolist()
        fold += ff.tolist()
        first += [ft] * yy.size
        suffix += [suf] * yy.size
    keys = np.array([c[0] for c in ctx_list], dtype=np.int64)
    return {
        "sel_idx": np.array(sel_idx, dtype=np.int32),
        "y": np.array(y, dtype=np.uint16),
        "fold": np.array(fold, dtype=np.uint8),
        "first_tok": np.array(first, dtype=np.uint16),
        "suffix": np.array(suffix, dtype=np.int64),
        "keys": keys,
    }


def build_fixture(root):
    """Write index npz + stats dir. Returns (index_path, stats_dir, info)."""
    stats_dir = os.path.join(root, "stats")
    os.makedirs(stats_dir)
    rng = np.random.default_rng(1234)
    protos = family_prototypes(rng)
    uni_q = np.full(VOCAB, 1.0 / VOCAB)

    # ---------------- order 2: bigrams (a,b), key = a*V + b, suffix = b ----
    ctx2 = []
    fam2 = {}                 # key -> family id (None = common suffix pool)
    # families: members share a prototype (up to sampling noise)
    for f in range(N_FAM):
        for m in range(FAM_SIZE2):
            a = (f * FAM_SIZE2 + m) % VOCAB
            b = (f * 5 + m * 7 + 1) % VOCAB
            key = pack_key((a, b))
            if key in fam2:
                continue
            ctx2.append((key, b, a))
            fam2[key] = f
    # common suffix pool: dedicated bigrams with a shared, family-independent
    # continuation distribution; order-3 families use these as suffixes so
    # the backoff q_2(suffix) is (nearly) identical across order-3 families
    # -> families separable in BOTH PROB and RESIDUAL spaces.
    POOL_PROTO = np.full(VOCAB, 1.0 / VOCAB)
    POOL_PROTO[21] += 0.1
    POOL_PROTO[22] += 0.05
    POOL_PROTO /= POOL_PROTO.sum()
    common_pool_keys = []
    for i in range(8):
        a = (3 + i * 5) % VOCAB
        b = (9 + i * 3) % VOCAB
        key = pack_key((a, b))
        if key in fam2:
            continue
        common_pool_keys.append(key)
        ctx2.append((key, b, a))
        fam2[key] = None
    pool_set = set(common_pool_keys)
    # filler: unique keys, near-uniform noisy distributions
    used = set(fam2)
    filler2 = []
    k_try = 0
    while len(filler2) < FILL2 and k_try < VOCAB * VOCAB * 4:
        a = (k_try * 13 + 5) % VOCAB
        b = (k_try * 29 + 3) % VOCAB
        k_try += 1
        key = pack_key((a, b))
        if key in used:
            continue
        used.add(key)
        filler2.append((key, b, a))
    ctx2 = sorted(ctx2 + filler2, key=lambda c: c[0])

    def q2(key):
        if key in pool_set:
            return POOL_PROTO
        if fam2.get(key) is not None:
            return protos[fam2[key]]
        # near-uniform filler with mild token preference
        q = np.full(VOCAB, 1.0 / VOCAB)
        q[key % VOCAB] += 0.05
        return q / q.sum()
    hits2 = make_hits(ctx2, q2, rng)

    # ---------------- order 3: trigrams, suffix in selected bigrams --------
    ctx3 = []
    fam3 = {}
    for f in range(N_FAM):
        for m in range(FAM_SIZE3):
            skey = common_pool_keys[m % len(common_pool_keys)]
            a = (f * 11 + m * 3 + 2) % VOCAB
            key = pack_key((a, skey // VOCAB, skey % VOCAB))
            if key in fam3:
                continue
            ctx3.append((key, skey, a))
            fam3[key] = f
    used3 = set(fam3)
    filler3 = []
    k_try = 0
    while len(filler3) < FILL3 and k_try < VOCAB * VOCAB * VOCAB * 4:
        skey = ctx2[(k_try * 7 + 1) % len(ctx2)][0]
        a = (k_try * 17 + 9) % VOCAB
        k_try += 1
        key = pack_key((a, skey // VOCAB, skey % VOCAB))
        if key in used3:
            continue
        used3.add(key)
        filler3.append((key, skey, a))
    ctx3 = sorted(ctx3 + filler3, key=lambda c: c[0])

    def q3(key):
        if key in fam3:
            return protos[fam3[key]]
        q = np.full(VOCAB, 1.0 / VOCAB)
        q[(key // VOCAB) % VOCAB] += 0.04
        return q / q.sum()
    hits3 = make_hits(ctx3, q3, rng)

    # ---------------- orders 4,5: minimal closed chains --------------------
    suf4 = ctx3[:4]
    ctx4 = sorted({(pack_key((7, c[0] // VOCAB ** 2, (c[0] // VOCAB) % VOCAB,
                             c[0] % VOCAB)), c[0], 7) for c in suf4},
                  key=lambda c: c[0])
    hits4 = make_hits(ctx4, lambda k: np.eye(VOCAB)[4], rng)
    suf5 = ctx4[:2]
    raw5 = sorted(rng.integers(1, 1 << 62, size=len(suf5), dtype=np.int64).tolist())
    ctx5 = [(int(raw5[i]), suf5[i][0], suf5[i][2]) for i in range(len(suf5))]
    hits5 = make_hits(ctx5, lambda k: np.eye(VOCAB)[6], rng)

    # ---------------- index npz ---------------------------------------------
    index = {}
    for n, ctxs in ((2, ctx2), (3, ctx3), (4, ctx4), (5, ctx5)):
        keys = np.array([c[0] for c in ctxs], dtype=np.int64)
        extra = np.array([keys.max() + 1], dtype=np.int64)
        index[f"keys_{n}"] = np.concatenate([keys, extra])
        index[f"counts_{n}"] = np.concatenate([
            np.full(keys.size, FIT_PER_CTX + DEV_PER_CTX, dtype=np.int32),
            np.array([FMIN - 1], dtype=np.int32)])
    index_path = os.path.join(root, "index.npz")
    np.savez(index_path, **index)

    for n, hits in ((2, hits2), (3, hits3), (4, hits4), (5, hits5)):
        np.savez(os.path.join(stats_dir, f"hits_n{n}.npz"), **hits)
    uni = np.zeros((3, VOCAB), dtype=np.int64)
    for f in (0, 1, 2):
        uni[f] = np.bincount(hits2["y"][hits2["fold"] == f], minlength=VOCAB)
    np.save(os.path.join(stats_dir, "unigram.npy"), uni)

    info = {"ctx2": ctx2, "ctx3": ctx3, "fam2": fam2, "fam3": fam3,
            "protos": protos, "q2": q2, "q3": q3}
    return index_path, stats_dir, info


# --------------------------------------------------------------------- tests
def test_q_properties(index_path, stats_dir):
    """q finite, non-negative, sum to 1; unseen-suffix backoff correct."""
    sel, _ = w1_common.load_selected(index_path)
    uni = np.load(os.path.join(stats_dir, "unigram.npy"))
    vocab = uni.shape[1]
    u = uni[0].astype(np.float64) + uni[1].astype(np.float64) + 1.0
    q_uni = u / u.sum()
    q_uni = np.maximum(q_uni, W.UNIGRAM_FLOOR)
    q_uni /= q_uni.sum()
    lower_keys, lower_q = None, None
    for n in (2, 3):
        z = np.load(os.path.join(stats_dir, f"hits_n{n}.npz"))
        K = sel[n][0].size
        ctx, tok, cnt, F, ctx_suffix = W.reduce_hits(
            z["sel_idx"], z["y"], z["fold"], z["suffix"], K, vocab)
        if n == 2:
            back = np.broadcast_to(q_uni, (K, vocab)).astype(np.float64)
        else:
            srows = W.suffix_rows(ctx_suffix, lower_keys)
            back = lower_q[srows].astype(np.float64)
        q = W.build_q(ctx, tok, cnt, F, K, vocab, 128.0, back.copy())
        assert np.isfinite(q).all(), f"n={n} q not finite"
        assert (q >= 0).all(), f"n={n} q negative"
        s = q.sum(axis=1)
        assert np.abs(s - 1).max() < 1e-6, f"n={n} q rows not normalized"
        # unseen-suffix backoff: a zero-count context equals its backoff row
        zero = np.nonzero(F == 0)[0]
        if zero.size:
            c = int(zero[0])
            assert np.allclose(q[c], back[c] / back[c].sum(), atol=1e-6), \
                f"n={n} zero-count context did not back off"
        lower_keys, lower_q = sel[n][0], q
    print("[test] q finite/nonneg/normalized + backoff OK")


def test_merge_loss():
    q = np.full(VOCAB, 1.0 / VOCAB)
    p = np.full(VOCAB, 1.0 / VOCAB)
    p[0] = 0.5
    p[1:] = 0.5 / (VOCAB - 1)
    d_same = W.js_merge_loss(q, 100.0, q.copy(), 200.0)
    d_diff = W.js_merge_loss(q, 100.0, p, 200.0)
    assert abs(d_same) < 1e-12, f"identical q merge loss {d_same} != 0"
    assert d_diff > 1e-6, f"distinct q merge loss {d_diff} <= 0"
    print("[test] merge loss identical==0, distinct>0 OK")


def test_merge_loss_equals_nll_diff():
    """Merge loss == before/after fit-NLL difference (toy, no smoothing change).

    Two clusters A,B with count-weighted distributions; forcing them to share
    the mixture raises the count-level NLL by exactly Delta_merge.
    """
    rng = np.random.default_rng(7)
    qA = rng.dirichlet(np.ones(VOCAB))
    qB = rng.dirichlet(np.ones(VOCAB))
    FA, FB = 300.0, 700.0
    # sample counts exactly proportional to qA/qB (toy: expectations)
    vA, vB = qA * FA, qB * FB
    # before: each token's NLL under its own distribution
    nll_before = -(vA * np.log(qA)).sum() - (vB * np.log(qB)).sum()
    qAB = (vA + vB) / (FA + FB)
    nll_after = -(vA * np.log(qAB)).sum() - (vB * np.log(qAB)).sum()
    delta = W.merge_loss_from_counts(vA, FA, vB, FB)
    assert abs((nll_after - nll_before) - delta) < 1e-8, \
        f"merge loss {delta} != NLL diff {nll_after - nll_before}"
    print("[test] merge loss == before/after NLL difference OK")


def _purity(rows, fam_of_key, keys):
    """Best-match family purity: for each cluster, majority family fraction."""
    from collections import defaultdict
    cl = defaultdict(list)
    for k, r in zip(keys, rows):
        f = fam_of_key.get(int(k))
        if f is not None:
            cl[int(r)].append(f)
    if not cl:
        return 0.0
    correct = 0
    total = 0
    for r, fs in cl.items():
        vals, cnts = np.unique(fs, return_counts=True)
        correct += int(cnts.max())
        total += len(fs)
    return correct / total


def test_c1_c2_c3(index_path, stats_dir, info, tmp):
    m_total = 96
    for algo in ("c1", "c2", "c3"):
        out1 = os.path.join(tmp, f"{algo}_a.npz")
        out2 = os.path.join(tmp, f"{algo}_b.npz")
        rows_map, meta = W.build(index_path, stats_dir, algo, out1,
                                 seed=w1_common.ROUTING_SEED, m_total=m_total,
                                 log=lambda *a: None)
        rows_map2, meta2 = W.build(index_path, stats_dir, algo, out2,
                                   seed=w1_common.ROUTING_SEED, m_total=m_total,
                                   log=lambda *a: None)
        o1 = np.load(out1)
        o2 = np.load(out2)

        # ---- determinism: bit-identical arrays + meta (minus elapsed) ------
        for k in o1.files:
            a, b = o1[k], o2[k]
            if k == "meta_json":
                ja, jb = json.loads(str(a)), json.loads(str(b))
                ja.pop("elapsed_s", None)
                jb.pop("elapsed_s", None)
                assert ja == jb, f"{algo}: meta_json mismatch across runs"
            else:
                assert np.array_equal(a, b), f"{algo}: array {k} differs"
        # rows_map returned must match file
        for n in rows_map:
            assert np.array_equal(rows_map[n], rows_map2[n])

        R = {n: int(o1[f"R_{n}"]) for n in w1_common.ORDERS}
        for n in w1_common.ORDERS:
            rows = o1[f"rows_{n}"]
            keys = o1[f"keys_{n}"]
            occ = int(meta[f"occupied_{n}"])
            unused = int(meta[f"unused_{n}"])
            # ---- rows in bounds, one row per context, occupied+unused==R ---
            assert rows.shape == keys.shape and rows.dtype == np.int32
            assert rows.min() >= 0 and rows.max() < occ, f"{algo} n={n} bounds"
            assert np.unique(rows).size == occ, f"{algo} n={n} unused rows"
            assert occ + unused == R[n], \
                f"{algo} n={n}: occupied {occ} + unused {unused} != R {R[n]}"
            assert occ <= R[n]

        # ---- C3: J monotone non-increasing across sweeps -------------------
        if algo == "c3":
            for n in w1_common.ORDERS:
                traj = meta[f"J_traj_{n}"]
                for i in range(1, len(traj)):
                    assert traj[i] <= traj[i - 1] + 1e-6, \
                        f"c3 n={n}: J increased {traj[i-1]} -> {traj[i]}"

        # ---- planted family purity (orders 2,3) ----------------------------
        for n, fam in ((2, info["fam2"]), (3, info["fam3"])):
            keys = o1[f"keys_{n}"]
            rows = o1[f"rows_{n}"]
            pur = _purity(rows, fam, keys)
            assert pur > 0.95, f"{algo} n={n}: family purity {pur:.3f} <= 0.95"
            # shuffle control: permuted labels must NOT recover purity
            rng = np.random.default_rng(0)
            perm = rng.permutation(rows)
            pur_perm = _purity(perm, fam, keys)
            assert pur_perm < pur - 0.3, \
                f"{algo} n={n}: shuffled purity {pur_perm:.3f} too high"
        print(f"[test] {algo}: determinism/bounds/budget/purity"
              + ("/J-monotone" if algo == "c3" else "") + " OK")


def main():
    tmp = tempfile.mkdtemp(prefix="w1_c123_synth_", dir="/tmp")
    try:
        index_path, stats_dir, info = build_fixture(tmp)
        test_q_properties(index_path, stats_dir)
        test_merge_loss()
        test_merge_loss_equals_nll_diff()
        test_c1_c2_c3(index_path, stats_dir, info, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
