#!/usr/bin/env python3
"""W1 S1 arm: suffix-structured grouping (plan v2 section 4, "S1").

Contexts of order n that share their order-(n-1) suffix form sibling groups
(reverse suffix trie). Within each order, starting from the coarsest grouping
(one group per order; plan v2 S1: "budget内没有必要区分的 context 共用较短
suffix 对应的 group" + child-vs-rest binary splits when a multi-way split
would exceed the remaining budget), greedily apply the child-vs-rest binary
split with the largest count-weighted predictive distortion (merge loss
    Delta = F_A*KL(q_A||q_AB) + F_B*KL(q_B||q_AB),  q_AB the count mixture)
until the order's row budget R_n is exhausted or no candidates remain
(occupied < R_n is reported, never silently padded).

Smoothed continuation distributions use fixed recursive backoff
    q_n(y|c) = [F_fit(c,y) + alpha * q_{n-1}(y|suffix(c))] / [F_fit(c) + alpha]
with alpha=128 (S1-fixed), fit counts = folds 0+1 only; unigram q_1 is the
add-1-smoothed fit-fold unigram (floor 1e-12, renormalized). Fold 2
(routing-dev) is used only for the diagnostic held-out NLL of grouped vs
ungrouped models.

Merge losses use the exact entropy identity
    Delta(x|P) = F_P*H(q_P) - F_x*H(q_x) - F_r*H(q_r)
with per-sibling F_x*H(q_x) precomputed once per order; candidate evaluation
is vectorized over sibling blocks (bounded peak memory).

Candidate shortlist (default on, --candidate-cap C): the exact entropy
identity gives the cheap per-sibling upper bound
    Delta(x|P) <= F_P*H(q_P) - F_x*H(q_x)        (since H(q_r) >= 0)
computable for ALL siblings of a leaf in O(S_leaf) (F_P*H(q_P) is one
scalar per leaf, F_x*H(q_x) is precomputed once per order). Only the top-C
siblings by bound (ties broken by sibling id ascending) get the exact
O(C * vocab) evaluation; the best of those is split. This keeps exact
merge-loss semantics for every evaluated candidate while cutting the
per-split cost from O(S * vocab) to O(S + C * vocab). A step is
"certified" when the selected sibling's exact gain >= the largest bound
among non-shortlisted siblings (bound >= exact => provably the global
best); the certified fraction is a lower bound on the true shortlist
recall and is reported in meta_json. With --candidate-cap 0 (or cap >=
leaf size) every sibling is evaluated exactly, reproducing the pre-cap
behavior bit-for-bit.

CLI: python3 w1_make_s1.py <index_path> <stats_dir> <out_path> [--seed SEED]
     [--candidate-cap C] [--exact-backend {auto,numpy,torch}]
Deterministic: fixed seed; all tie-breaking by key ascending. No RNG is used
anywhere; the seed is recorded in meta for provenance only.

Scaling note: exact candidate re-evaluation costs O(C * vocab) per split
with the shortlist (C = candidate cap), so an order with R_n splits costs
~O(R_n * (S + C * vocab)). See the deliverable report for measured
constants and real-data projections.
"""
import os
import sys
import time
import heapq
import argparse
import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None
    _HAVE_TORCH = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w1_common import (ORDERS, ROUTING_SEED, load_selected, allocate_R,
                       write_assignment)

ALPHA = 128.0          # S1-fixed backoff mass (plan v2 section 3.3 grid point)
UNIGRAM_FLOOR = 1e-12  # renormalized floor for q_1


# ---------------------------------------------------------------- merge loss
def merge_loss_from_counts(vA, FA, vB, FB):
    """Delta = FA*KL(qA||qAB) + FB*KL(qB||qAB) from smoothed count vectors.

    vA/vB are smoothed (alpha-augmented) count vectors with totals FA/FB, so
    the mixture qAB = (vA+vB)/(FA+FB) is exactly the smoothed mixture. qAB>0
    wherever qA/qB>0 (mixture), so the KLs are finite.
    """
    qA = vA / FA
    qB = vB / FB
    qAB = (vA + vB) / (FA + FB)
    la = np.where(qA > 0, np.log(qA, where=qA > 0, out=np.zeros_like(qA)), 0.0)
    lb = np.where(qB > 0, np.log(qB, where=qB > 0, out=np.zeros_like(qB)), 0.0)
    lab = np.where(qAB > 0, np.log(qAB, where=qAB > 0, out=np.zeros_like(qAB)), 0.0)
    klA = float((qA * (la - lab)).sum())
    klB = float((qB * (lb - lab)).sum())
    return max(FA * klA + FB * klB, 0.0)  # numerical floor


def merge_loss_batch(child_cnt, child_F, parent_cnt, parent_F, child_h=None,
                     block=8192):
    """Vectorized merge loss of peeling each row of child_cnt off the parent.

    child_cnt (S, vocab) float64 smoothed counts, child_F (S,) totals;
    parent_cnt (vocab,), parent_F scalar; child_h (S,) optional precomputed
    F_x*H(q_x) per sibling (recomputed when not given). Returns (S,) gains =
    F_c*KL(q_c||q_m) + F_r*KL(q_r||q_m) with rest = parent - child, using the
    exact entropy identity
        Delta = F_P*H(q_P) - F_c*H(q_c) - F_r*H(q_r),
        F_r*H(q_r) = -G(V_P - V_c) + F_r*log(F_r),  G(V) = sum V log V.
    Rows with nonpositive child or rest mass get -inf (invalid). Processes
    siblings in blocks to bound peak memory.
    """
    Ft = float(parent_F)
    S = child_cnt.shape[0]
    # F_P*H(q_P), constant across candidates
    lp = np.log(parent_cnt, where=parent_cnt > 0,
                out=np.zeros_like(parent_cnt))
    HP = float(-(parent_cnt * lp).sum()) + Ft * np.log(Ft)  # F_P*H(q_P)
    out = np.full(S, -np.inf, dtype=np.float64)
    for a in range(0, S, block):
        b = min(a + block, S)
        Vc = child_cnt[a:b]
        Fc = child_F[a:b].astype(np.float64)
        Fr = Ft - Fc
        valid = (Fc > 0) & (Fr > 0)
        if child_h is None:
            lc = np.log(Vc, where=Vc > 0, out=np.zeros_like(Vc))
            hc = -(Vc * lc).sum(axis=1) + Fc * np.log(np.where(Fc > 0, Fc, 1.0))
        else:
            hc = child_h[a:b]
        # G(V_P - V_c): rest = parent - child, clipped for float drift
        rest = parent_cnt[None, :] - Vc
        np.maximum(rest, 0.0, out=rest)
        lr = np.log(rest, where=rest > 0, out=np.zeros_like(rest))
        G = (rest * lr).sum(axis=1)
        FrH = -G + Fr * np.log(np.where(Fr > 0, Fr, 1.0))
        gain = HP - hc - FrH
        out[a:b] = np.where(valid, np.maximum(gain, 0.0), -np.inf)
    return out


def js_merge_loss(qA, FA, qB, FB):
    """Same as merge_loss_from_counts but from normalized rows (test helper)."""
    return merge_loss_from_counts(qA * FA, FA, qB * FB, FB)


# ------------------------------------------------------- torch exact backend
class _TorchExact(object):
    """Torch (CUDA if available) exact merge-loss evaluator for a shortlist.

    Holds the sibling count matrix on device as a dense (S, vocab) fp64
    tensor; shortlisted rows are gathered and scored with the exact entropy
    identity in chunks. fp64 everywhere keeps results bit-comparable to the
    numpy reference (same operation order is NOT guaranteed, so tiny last-ulp
    differences are possible; the committed split decision re-checks the
    chosen candidate through the same backend that produced the heap entry,
    and tie-breaking by sibling id makes the selection deterministic).

    Determinism: torch CPU and CUDA reductions are deterministic for fixed
    shapes/device; no RNG is used. The numpy backend remains the reference
    and is what --exact-backend numpy forces.
    """

    def __init__(self, sib_cnt, sib_F, sib_h, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.cnt = torch.from_numpy(np.ascontiguousarray(sib_cnt)).to(self.device)
        self.F = torch.from_numpy(np.ascontiguousarray(sib_F)).to(self.device)
        self.h = torch.from_numpy(np.ascontiguousarray(sib_h)).to(self.device)
        self.vocab = int(self.cnt.shape[1])

    def gains(self, sibs, parent_cnt, parent_F, chunk=128):
        """Exact entropy-identity gains for shortlisted siblings."""
        idx = torch.from_numpy(np.ascontiguousarray(sibs, dtype=np.int64)).to(self.device)
        Vp = torch.from_numpy(np.ascontiguousarray(parent_cnt)).to(self.device)
        Ft = float(parent_F)
        lp = torch.where(Vp > 0, torch.log(torch.clamp(Vp, min=1e-300)),
                         torch.zeros_like(Vp))
        HP = float(-(Vp * lp).sum()) + Ft * np.log(Ft)
        h = self.h[idx]                       # (C,)
        Fc = self.F[idx]                      # (C,)
        Fr = Ft - Fc
        out = torch.full((idx.numel(),), float("-inf"), dtype=torch.float64,
                         device=self.device)
        for a in range(0, idx.numel(), chunk):
            b = min(a + chunk, idx.numel())
            Vc = self.cnt[idx[a:b]]           # (c, vocab)
            rest = torch.clamp(Vp[None, :] - Vc, min=0.0)
            lr = torch.where(rest > 0, torch.log(torch.clamp(rest, min=1e-300)),
                             torch.zeros_like(rest))
            G = (rest * lr).sum(dim=1)
            fr = Fr[a:b]
            FrH = -G + fr * torch.log(torch.clamp(fr, min=1e-300))
            gain = HP - h[a:b] - FrH
            valid = (Fc[a:b] > 0) & (fr > 0)
            out[a:b] = torch.where(valid, torch.clamp(gain, min=0.0),
                                   torch.full_like(gain, float("-inf")))
        return out.cpu().numpy()


# ------------------------------------------------------------- trie grouping
class TrieGrouping(object):
    """Greedy binary-split forest over the sibling (shared-suffix) partition.

    Sibling ids come from np.unique(suffix_key). The root groups ALL contexts
    of the order; each split peels one sibling child off a leaf. Leaves are
    parallel arrays; per-leaf sibling membership is a sorted int32 array plus
    a set for O(1) staleness checks. Sibling count vectors are precomputed
    once (S, vocab); leaf vectors are maintained incrementally.

    Candidate evaluation uses the entropy-identity upper bound
        Delta(x|P) <= F_P*H(q_P) - F_x*H(q_x)        (H(q_r) >= 0)
    to shortlist the top-`candidate_cap` siblings by bound (ties at the
    threshold broken by lowest sibling id); only the shortlist gets the exact
    O(C * vocab) evaluation. cap=0 or cap >= leaf size evaluates every
    sibling exactly (pre-cap behavior, bit-identical). Selection among
    evaluated candidates is by (gain desc, sib asc) via np.lexsort, which in
    full-cap mode reproduces the old argmax-first-max rule (lowest sibling
    id among ties). A step is "certified" when the selected exact gain >=
    the largest bound among excluded siblings (bound >= exact => provably
    the global best); full-cap steps count as certified.

    exact_backend: "numpy" (reference) or "torch" (_TorchExact evaluator for
    the shortlist; same fp64 numerics).
    """

    def __init__(self, sib_cnt, sib_F, candidate_cap=0, exact_backend="numpy"):
        self.S = int(sib_F.size)
        self.sib_cnt = sib_cnt          # (S, vocab) float64 smoothed counts
        self.sib_F = sib_F              # (S,) float64
        self.candidate_cap = int(candidate_cap)
        self.exact_backend = exact_backend
        # per-sibling F*H(q), precomputed once (constant across split steps)
        lc = np.log(sib_cnt, where=sib_cnt > 0, out=np.zeros_like(sib_cnt))
        self.sib_h = (-(sib_cnt * lc).sum(axis=1)
                      + sib_F * np.log(np.where(sib_F > 0, sib_F, 1.0)))
        self._tex = None
        if exact_backend == "torch":
            self._tex = _TorchExact(sib_cnt, sib_F, self.sib_h)
        root = sib_cnt.sum(axis=0)
        self.leaf_cnt = [root]
        self.leaf_F = [float(root.sum())]
        self.leaf_sibs = [np.arange(self.S, dtype=np.int32)]
        self.leaf_sibset = [set(range(self.S))]
        self.leaf_gen = [0]             # bumped on split; heap staleness
        # shortlist certification counters
        self.steps_total = 0
        self.steps_certified = 0
        self.steps_shortlisted = 0

    def _exact_gains(self, sibs, lid):
        """Exact entropy-identity gains for sibling ids `sibs` of leaf lid."""
        if self._tex is not None:
            return self._tex.gains(sibs, self.leaf_cnt[lid], self.leaf_F[lid])
        return merge_loss_batch(self.sib_cnt[sibs], self.sib_F[sibs],
                                self.leaf_cnt[lid], self.leaf_F[lid],
                                child_h=self.sib_h[sibs])

    def candidate(self, lid):
        """Best child-vs-rest split of leaf lid -> (gain, sib) or None.

        With an active cap, only the top-C siblings by upper bound are
        evaluated exactly; the certification counters are updated here.
        """
        sibs = self.leaf_sibs[lid]
        if sibs.size < 2:
            return None
        cap = self.candidate_cap
        if cap <= 0 or cap >= sibs.size:
            # full exact evaluation (pre-cap behavior)
            gains = self._exact_gains(sibs, lid)
            # (gain desc, sib asc): lexsort primary key = last = -gains
            order = np.lexsort((sibs, -gains))
            i = int(order[0])
            if not np.isfinite(gains[i]):
                return None
            self.steps_total += 1
            self.steps_certified += 1   # full-cap steps are certified
            return float(gains[i]), int(sibs[i])

        # ---- upper-bound shortlist --------------------------------------
        # bound(x) = F_P*H(q_P) - F_x*H(q_x); F_P*H(q_P) is one scalar.
        pcnt = self.leaf_cnt[lid]
        Ft = self.leaf_F[lid]
        lp = np.log(pcnt, where=pcnt > 0, out=np.zeros_like(pcnt))
        HP = float(-(pcnt * lp).sum()) + Ft * np.log(Ft)
        bounds = HP - self.sib_h[sibs]              # (S_leaf,)
        # deterministic top-C: C-th largest bound via argpartition, then
        # include everything strictly above the threshold plus threshold
        # ties chosen by LOWEST sibling id.
        kth = sibs.size - cap                        # index of C-th largest
        thr = np.partition(bounds, kth)[kth]
        above = bounds > thr
        n_above = int(above.sum())
        need = cap - n_above                         # threshold ties to take
        tie_pos = np.flatnonzero(~above & (bounds == thr))
        # lowest sibling id among ties (sibs is sorted ascending, so the
        # first `need` positions in tie_pos are the lowest sibling ids)
        take = tie_pos[:need]
        short_pos = np.concatenate([np.flatnonzero(above), take])
        short_sibs = sibs[short_pos]
        short_bounds = bounds[short_pos]
        # order shortlist by (bound desc, sib asc)
        so = np.lexsort((short_sibs, -short_bounds))
        short_sibs = short_sibs[so]
        # exact evaluation on the shortlist only
        gains = self._exact_gains(short_sibs, lid)
        go = np.lexsort((short_sibs, -gains))
        i = int(go[0])
        if not np.isfinite(gains[i]):
            return None
        best_gain = float(gains[i])
        best_sib = int(short_sibs[i])
        # certificate: best exact gain >= max bound among EXCLUDED siblings
        excluded_mask = np.ones(sibs.size, dtype=bool)
        excluded_mask[short_pos] = False
        max_excl = float(bounds[excluded_mask].max()) if excluded_mask.any() \
            else -np.inf
        self.steps_total += 1
        self.steps_shortlisted += 1
        if best_gain >= max_excl:
            self.steps_certified += 1
        return best_gain, best_sib

    def split(self, lid, s):
        """Peel sibling s off leaf lid -> new leaf id (child)."""
        sibs = self.leaf_sibs[lid]
        keep = sibs != s
        self.leaf_sibs[lid] = sibs[keep]
        self.leaf_sibset[lid].discard(s)
        child_cnt = self.sib_cnt[s].copy()
        self.leaf_cnt[lid] = self.leaf_cnt[lid] - child_cnt
        self.leaf_F[lid] = float(self.leaf_cnt[lid].sum())
        self.leaf_cnt.append(child_cnt)
        self.leaf_F.append(float(child_cnt.sum()))
        self.leaf_sibs.append(np.array([s], dtype=np.int32))
        self.leaf_sibset.append({s})
        self.leaf_gen[lid] += 1
        self.leaf_gen.append(0)
        return len(self.leaf_cnt) - 1


# ------------------------------------------------------------------ dev NLL
def dev_nll(sel_idx, y, fold, rows, q_grouped, q_full):
    """Count-level held-out NLL (nats/token) on fold 2, grouped vs exact."""
    m = fold == 2
    if not m.any():
        return float("nan"), float("nan")
    si = sel_idx[m].astype(np.int64)
    yy = y[m].astype(np.int64)
    p_exact = q_full[si, yy]
    p_group = q_grouped[rows[si], yy]
    return (float(-np.log(p_group).sum() / m.sum()),
            float(-np.log(p_exact).sum() / m.sum()))


# ------------------------------------------------------- per-order reduction
def reduce_hits(sel_idx, y, fold, suffix, K, vocab):
    """Per-context fit counts + first-hit suffix key.

    Returns ctx, tok, cnt (nz fit entries, folds 0+1), F (K,) fit totals,
    ctx_suffix (K,) suffix key from the context's first hit over ALL folds
    (topology must exist for every selected context; counts use fit folds
    only). Every selected context has >= FMIN hits overall, so ctx_suffix is
    always populated.
    """
    m = fold < 2
    si = sel_idx[m].astype(np.int64)
    yy = y[m].astype(np.int64)
    lin = si * vocab + yy
    u, c = np.unique(lin, return_counts=True)
    ctx = (u // vocab).astype(np.int64)
    tok = (u % vocab).astype(np.int64)
    F = np.zeros(K, dtype=np.float64)
    np.add.at(F, ctx, c.astype(np.float64))
    # first hit per context over all folds (stable: hits are in scan order)
    si_all = sel_idx.astype(np.int64)
    srt = np.argsort(si_all, kind="stable")
    si_s = si_all[srt]
    first = np.r_[True, si_s[1:] != si_s[:-1]]
    ctx_suffix = np.zeros(K, dtype=np.int64)
    ctx_suffix[si_s[first]] = suffix[srt][first]
    return ctx, tok, c.astype(np.float64), F, ctx_suffix


# ---------------------------------------------------------------------- main
def build(index_path, stats_dir, out_path, seed=ROUTING_SEED, vocab=None,
          m_total=None, candidate_cap=256, exact_backend="auto"):
    t0 = time.time()
    if exact_backend == "auto":
        exact_backend = "torch" if (_HAVE_TORCH and torch.cuda.is_available()) \
            else "numpy"
    if exact_backend == "torch" and not _HAVE_TORCH:
        print("[s1] WARNING: --exact-backend torch but torch unavailable; "
              "falling back to numpy", flush=True)
        exact_backend = "numpy"
    sel, counts_total = load_selected(index_path)
    R = allocate_R(sel) if m_total is None else allocate_R(sel, m_total=m_total)
    print(f"[s1] selected: " + ", ".join(f"n{n}={sel[n][0].size}" for n in ORDERS), flush=True)
    print(f"[s1] R allocation: {R}", flush=True)
    print(f"[s1] candidate_cap={candidate_cap} exact_backend={exact_backend}",
          flush=True)

    # ---- unigram q_1 from stats (folds 0+1, add-1 smoothing, floor) --------
    uni = np.load(os.path.join(stats_dir, "unigram.npy"))  # (3, vocab) int64
    if vocab is None:
        vocab = int(uni.shape[1])
    assert uni.shape[1] == vocab
    u = uni[0].astype(np.float64) + uni[1].astype(np.float64) + 1.0
    q_uni = u / u.sum()
    q_uni = np.maximum(q_uni, UNIGRAM_FLOOR)
    q_uni /= q_uni.sum()

    rows_map = {}
    meta = {"algo": "s1", "seed": seed, "alpha": ALPHA, "fmin": 256,
            "m_total": int(sum(R.values())),
            "candidate_cap": int(candidate_cap),
            "exact_backend": exact_backend,
            "R": {str(n): int(R[n]) for n in ORDERS}}
    lower_sel_keys = None
    lower_q = None

    for n in ORDERS:
        tn = time.time()
        z = np.load(os.path.join(stats_dir, f"hits_n{n}.npz"))
        sel_idx = z["sel_idx"]
        y = z["y"]
        fold = z["fold"]
        suffix = z["suffix"]
        keys = sel[n][0]
        K = keys.size

        ctx, tok, cnt, F, ctx_suffix = reduce_hits(sel_idx, y, fold, suffix, K, vocab)

        # ---- smoothed q (K, vocab) ----------------------------------------
        if n == 2:
            q = np.broadcast_to(q_uni, (K, vocab)).copy()
            pos_c = None
        else:
            pos = np.searchsorted(lower_sel_keys, ctx_suffix)
            pos_c = np.clip(pos, 0, lower_sel_keys.size - 1)
            ok = lower_sel_keys[pos_c] == ctx_suffix
            if not ok.all():
                raise AssertionError(
                    f"n={n}: suffix closure violated for {int((~ok).sum())} contexts")
            q = lower_q[pos_c].copy()
        q *= ALPHA
        q[ctx, tok] += cnt
        q /= (F + ALPHA)[:, None]

        # ---- sibling partition ---------------------------------------------
        sib_key, sib_id = np.unique(ctx_suffix, return_inverse=True)
        S = sib_key.size
        # sibling smoothed count vector = alpha * sum(member lower-q) + counts
        sib_cnt = np.zeros((S, vocab), dtype=np.float64)
        if n == 2:
            members = np.bincount(sib_id, minlength=S).astype(np.float64)
            sib_cnt[:] = members[:, None] * (ALPHA * q_uni)[None, :]
        else:
            np.add.at(sib_cnt, sib_id, ALPHA * lower_q[pos_c])
        np.add.at(sib_cnt, (sib_id[ctx], tok), cnt)
        sib_F = sib_cnt.sum(axis=1)

        # ---- greedy splits ---------------------------------------------------
        # Heap entries: (-gain, sib, lid, gen). candidate() is deterministic
        # given leaf state, so a fresh-gen entry can be split directly without
        # recompute; a stale-gen entry is dropped if the leaf already has a
        # fresh entry queued (the fresh one carries the current best), else
        # recomputed and re-pushed with the current gen. Split decisions are
        # identical to the recompute-on-pop reference.
        tg = TrieGrouping(sib_cnt, sib_F, candidate_cap=candidate_cap,
                          exact_backend=exact_backend)
        heap = []
        fresh = set()                    # lids with a fresh entry in the heap
        cand = tg.candidate(0)
        if cand is not None:
            heapq.heappush(heap, (-cand[0], cand[1], 0, 0))
            fresh.add(0)
        target = R[n]
        occupied = 1
        while occupied < target and heap:
            neggain, s, lid, gen = heapq.heappop(heap)
            if lid >= len(tg.leaf_cnt) or s not in tg.leaf_sibset[lid]:
                fresh.discard(lid)
                continue  # stale entry (sibling no longer in this leaf)
            if gen != tg.leaf_gen[lid]:
                # stale generation: the leaf was split after this entry
                if lid in fresh:
                    continue  # a fresh entry already carries the current best
                cand = tg.candidate(lid)
                if cand is None:
                    continue
                heapq.heappush(heap, (-cand[0], cand[1], lid,
                                      tg.leaf_gen[lid]))
                fresh.add(lid)
                continue
            fresh.discard(lid)           # consuming the fresh entry
            new_lid = tg.split(lid, s)
            occupied += 1
            for l2 in (lid, new_lid):
                cand2 = tg.candidate(l2)
                if cand2 is not None:
                    heapq.heappush(heap, (-cand2[0], cand2[1], l2,
                                          tg.leaf_gen[l2]))
                    fresh.add(l2)
                else:
                    fresh.discard(l2)

        # ---- rows -----------------------------------------------------------
        sib_leaf = np.empty(S, dtype=np.int32)
        for lid, sibs in enumerate(tg.leaf_sibs):
            for s in sibs:
                sib_leaf[s] = lid
        rows = sib_leaf[sib_id].astype(np.int32)
        assert rows.min() >= 0 and rows.max() < occupied

        # ---- dev NLL ---------------------------------------------------------
        leaf_cnt = np.stack(tg.leaf_cnt)
        leaf_F = np.asarray(tg.leaf_F)
        q_grouped = leaf_cnt / leaf_F[:, None]
        nll_g, nll_e = dev_nll(sel_idx, y, fold, rows, q_grouped, q)

        rows_map[n] = rows
        meta[f"occupied_{n}"] = int(occupied)
        meta[f"R_{n}"] = int(target)
        meta[f"dev_nll_grouped_{n}"] = nll_g
        meta[f"dev_nll_exact_{n}"] = nll_e
        meta[f"steps_total_{n}"] = int(tg.steps_total)
        meta[f"steps_certified_{n}"] = int(tg.steps_certified)
        meta[f"steps_shortlisted_{n}"] = int(tg.steps_shortlisted)
        meta[f"certified_best_frac_{n}"] = (
            float(tg.steps_certified) / float(tg.steps_total)
            if tg.steps_total else 1.0)
        print(f"[s1] n={n}: K={K} siblings={S} occupied={occupied}/{target} "
              f"devNLL grouped={nll_g:.4f} exact={nll_e:.4f} "
              f"certified={tg.steps_certified}/{tg.steps_total} "
              f"({time.time()-tn:.0f}s)", flush=True)

        lower_sel_keys = keys
        lower_q = q

    meta["elapsed_s"] = round(time.time() - t0, 1)
    write_assignment(out_path, sel, R, rows_map, counts_total, meta)
    return rows_map, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("index_path")
    ap.add_argument("stats_dir")
    ap.add_argument("out_path")
    ap.add_argument("--seed", type=int, default=ROUTING_SEED)
    ap.add_argument("--candidate-cap", type=int, default=256,
                    help="shortlist size for exact candidate evaluation; "
                         "0 = evaluate all siblings exactly (pre-cap behavior)")
    ap.add_argument("--exact-backend", choices=("auto", "numpy", "torch"),
                    default="auto",
                    help="backend for exact merge-loss evaluation of the "
                         "shortlist; auto = torch iff CUDA is available")
    a = ap.parse_args()
    build(a.index_path, a.stats_dir, a.out_path, seed=a.seed,
          candidate_cap=a.candidate_cap, exact_backend=a.exact_backend)
