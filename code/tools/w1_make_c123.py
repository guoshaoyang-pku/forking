#!/usr/bin/env python3
"""W1 clustering routers C1 / C2 / C3 (plan v2 sections 3-6).

C1: Hellinger weighted clustering -- PROB features z(c)=sqrt(q_n(.|c)),
    L2 k-means with count weights w_c=F(c), weighted-mean centers.
C2: identical optimizer/seed/budget on RESIDUAL features
    r(c)=sqrt(q_n(.|c))-sqrt(q_{n-1}(.|suffix(c))) (NOT unit-normed).
C3: predictive-KL exchange refinement of the C1 assignment:
    J = sum_c F(c) KL(q_c || q_{a(c)}),  q_j = weighted mean of member q's,
    exact KL evaluated in the ORIGINAL probability space; the 128-d PROB
    projection is used only to propose candidate centroids (ANN via cdist).

Smoothing (plan v2 3.3): recursive backoff
    q_n(y|c) = [F_fit(c,y) + alpha_n q_{n-1}(y|suffix(c))] / [F_fit(c)+alpha_n]
down to unigram q_1 (add-1, floored, renormalized). alpha_n is selected per
order from {32,128,512} by held-out NLL on routing-dev fold 2 (fit on folds
0+1), then q is RE-FIT on ALL folds with the chosen alpha for the clustering
input (declared in meta; the LM validation is the real judge).

Two-level weighted k-means (plan v2 6): ~256 coarse groups, per-group local
K by deterministic largest remainder of group weight, sum(local K) == R_n
exactly; groups lacking points take fewer and the shortfall is reported
(never silently padded). Deterministic given --seed (routing randomness,
distinct from LM randomness).

Memory discipline: continuation counts are kept sparse (ctx/tok/cnt); q rows
are materialized dense (K, vocab) float32 one order at a time -- with vocab
8192 this is <= ~13GB for K<=408k, fine on the target host. No K x R or
K x K distance matrix is ever materialized; all distance computations are
chunked (chunk x centers).

CLI: python3 w1_make_c123.py <index_path> <stats_dir> <c1|c2|c3> <out_path>
     [--seed ROUTING_SEED] [--order N]  (default: all orders 2..5)
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w1_common import (ORDERS, ROUTING_SEED, load_selected, allocate_R,
                       write_assignment)

ALPHA_GRID = (32.0, 128.0, 512.0)
UNIGRAM_FLOOR = 1e-12
PROJ_DIM = 128
COARSE_K = 256
PCA_SAMPLE = 100_000        # count-weighted sample cap for PCA fit
N_CAND = 24                 # C3 ANN candidates per point (incl. own centroid)
C3_SWEEPS = 2
KM_ITERS_COARSE = 25
KM_ITERS_LOCAL = 25
DIST_CHUNK = 65536          # rows per cdist / projection chunk
CENT_CHUNK = 8192           # centroid columns per cdist chunk (C3 ANN)


# ----------------------------------------------------------------- rng utils
def rng_u64(seed, n, tag):
    """Deterministic uint64 scalar stream: splitmix64(seed ^ const(n) ^ tag)."""
    z = (int(seed) & 0xFFFFFFFFFFFFFFFF) ^ ((0x9E3779B97F4A7C15 * n) & 0xFFFFFFFFFFFFFFFF) ^ (tag & 0xFFFFFFFFFFFFFFFF)
    z &= 0xFFFFFFFFFFFFFFFF
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


def torch_gen(seed, n, tag):
    g = torch.Generator(device="cpu")   # cpu generator: identical draws everywhere
    g.manual_seed(rng_u64(seed, n, tag))
    return g


def weighted_choice(prob, k, gen):
    """Deterministic weighted sampling WITHOUT replacement (torch.multinomial)."""
    k = min(k, int(prob.numel()))
    return torch.multinomial(prob, k, replacement=False, generator=gen)


# ---------------------------------------------------------------- merge loss
def js_merge_loss(qA, FA, qB, FB):
    """Delta = FA*KL(qA||qAB) + FB*KL(qB||qAB) (test helper, dense rows)."""
    qAB = (FA * qA + FB * qB) / (FA + FB)
    klA = float((qA * (np.log(qA) - np.log(qAB))).sum())
    klB = float((qB * (np.log(qB) - np.log(qAB))).sum())
    return max(FA * klA + FB * klB, 0.0)


def merge_loss_from_counts(vA, FA, vB, FB):
    """Same from (smoothed) count vectors with totals FA/FB."""
    return js_merge_loss(vA / FA, FA, vB / FB, FB)


# ------------------------------------------------------------- data reduction
def reduce_hits(sel_idx, y, fold, suffix, K, vocab):
    """Per-context sparse continuation counts over FIT folds (0+1).

    Returns ctx, tok, cnt (nz fit entries), F (K,) fit totals, and
    ctx_suffix (K,) = suffix key of each context's first hit over ALL folds
    (topology exists for every selected context; counts are fit-only).
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
    si_all = sel_idx.astype(np.int64)
    srt = np.argsort(si_all, kind="stable")
    si_s = si_all[srt]
    first = np.r_[True, si_s[1:] != si_s[:-1]]
    ctx_suffix = np.zeros(K, dtype=np.int64)
    ctx_suffix[si_s[first]] = suffix[srt][first]
    return ctx, tok, c.astype(np.float64), F, ctx_suffix


def suffix_rows(ctx_suffix, lower_sel_keys):
    """Map each context's suffix key -> row in the order-(n-1) selected list."""
    pos = np.searchsorted(lower_sel_keys, ctx_suffix)
    pos_c = np.clip(pos, 0, lower_sel_keys.size - 1)
    ok = lower_sel_keys[pos_c] == ctx_suffix
    if not ok.all():
        raise AssertionError(
            f"suffix closure violated for {int((~ok).sum())} contexts")
    return pos_c


def build_q(ctx, tok, cnt, F, K, vocab, alpha, back_rows):
    """Smoothed q_n (K, vocab) float32: (counts + alpha*backoff)/(F+alpha).

    back_rows: (K, vocab) backoff distribution per context (unigram broadcast
    for n=2, else lower_q[suffix_rows]); not modified (a fresh product is
    allocated)."""
    q = back_rows * alpha
    q[ctx, tok] += cnt
    q /= (F + alpha)[:, None].astype(q.dtype)
    return q.astype(np.float32)


def dev_nll_of(q, sel_idx, y, fold):
    """Held-out NLL (nats/token) of q on fold 2 (routing-dev)."""
    m = fold == 2
    if not m.any():
        return float("nan")
    si = sel_idx[m].astype(np.int64)
    yy = y[m].astype(np.int64)
    p = q[si, yy]
    return float(-np.log(p).sum() / m.sum())


# --------------------------------------------------------------------- PCA
def fit_pca(feat_np, w, seed, n, device):
    """Randomized PCA (PROJ_DIM) fit ONLY on a count-weighted sample of rows.

    feat_np: (K, vocab) float32 feature matrix (PROB or RESIDUAL).
    w: (K,) float64 fit counts (sample probability proportional to w).
    Returns mean (vocab,) float32 and comps (vocab, d) float32 (col-orthonormal).
    """
    K = feat_np.shape[0]
    gen = torch_gen(seed, n, 0x9CA0)
    p = torch.from_numpy(w / w.sum()).double()
    idx = weighted_choice(p, min(PCA_SAMPLE, K), gen).numpy()
    idx.sort()
    S = torch.from_numpy(feat_np[idx]).to(device)
    mean = S.mean(dim=0)
    Sc = S - mean
    C = (Sc.T @ Sc).double() / max(S.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(C)
    d = min(PROJ_DIM, feat_np.shape[1])
    comps = evecs[:, -d:].flip(dims=[1]).float()   # descending variance
    return mean.cpu().numpy().astype(np.float32), comps.cpu().numpy().astype(np.float32)


def project(feat_np, mean, comps, device):
    """(K, vocab) @ (vocab, d) -> (K, d) float32, chunked."""
    K = feat_np.shape[0]
    d = comps.shape[1]
    out = np.empty((K, d), dtype=np.float32)
    comp_t = torch.from_numpy(comps).to(device)
    mean_t = torch.from_numpy(mean).to(device)
    for a in range(0, K, DIST_CHUNK):
        b = min(a + DIST_CHUNK, K)
        X = torch.from_numpy(feat_np[a:b]).to(device) - mean_t
        out[a:b] = (X @ comp_t).cpu().numpy()
    return out


# ------------------------------------------------------- weighted k-means
def cdist_argmin(X, C):
    """Row-wise argmin ||x - c||^2, chunked over rows AND centroid columns."""
    K = X.shape[0]
    k = C.shape[0]
    out = torch.empty(K, dtype=torch.long, device=X.device)
    c2 = (C * C).sum(dim=1)
    for a in range(0, K, DIST_CHUNK):
        b = min(a + DIST_CHUNK, K)
        x = X[a:b]
        x2 = (x * x).sum(dim=1, keepdim=True)
        best_d = None
        best_j = None
        for c0 in range(0, k, CENT_CHUNK):
            c1 = min(c0 + CENT_CHUNK, k)
            d = x2 - 2.0 * (x @ C[c0:c1].T) + c2[c0:c1][None, :]
            j = d.argmin(dim=1)
            md = d.gather(1, j[:, None]).squeeze(1)
            if best_d is None:
                best_d, best_j = md, j + c0
            else:
                upd = md < best_d
                best_d = torch.where(upd, md, best_d)
                best_j = torch.where(upd, j + c0, best_j)
        out[a:b] = best_j
    return out


def kmeanspp_init(X, w, k, gen):
    """k-means++ style init from a weighted sample, deterministic via gen."""
    K = X.shape[0]
    k = min(k, K)
    p = torch.from_numpy(w / w.sum()).double()
    i0 = int(weighted_choice(p, 1, gen)[0])
    centers = [X[i0:i0 + 1]]
    d2 = ((X - centers[0]) ** 2).sum(dim=1)
    wt = torch.from_numpy(w).to(X.device)
    for _ in range(1, k):
        probs = d2 * wt
        tot = float(probs.sum())
        if tot <= 0:                        # all remaining points identical
            idx = weighted_choice(p, 1, gen)
        else:
            # cpu generator -> draw on cpu (same values => same deterministic idx)
            idx = torch.multinomial((probs / tot).cpu(), 1, generator=gen)
        i = int(idx[0])
        centers.append(X[i:i + 1])
        d2 = torch.minimum(d2, ((X - centers[-1]) ** 2).sum(dim=1))
    return torch.cat(centers, dim=0)


def weighted_kmeans(X, w, k, gen, iters):
    """Plain weighted L2 k-means (weighted-mean centers, no normalization).

    X (K, d) torch tensor on device, w (K,) float64 numpy. Returns
    (labels (K,) int64 numpy, occupied int). Empty centers are re-seeded on
    the current highest-distortion points (deterministic: descending
    distortion, ties by index).
    """
    K = X.shape[0]
    k = min(k, K)
    wt = torch.from_numpy(w).to(X.device)
    wt32 = wt.float()
    C = kmeanspp_init(X, w, k, gen)
    labels = torch.zeros(K, dtype=torch.long, device=X.device)
    for _ in range(iters):
        labels = cdist_argmin(X, C)
        newC = torch.zeros_like(C)
        cw = torch.zeros(k, dtype=torch.float64, device=X.device)
        newC.scatter_add_(0, labels[:, None].expand(-1, C.shape[1]),
                          X * wt32[:, None])
        cw.scatter_add_(0, labels, wt)
        nonempty = cw > 0
        newC[nonempty] = newC[nonempty] / cw[nonempty].float()[:, None]
        n_empty = int((~nonempty).sum())
        if n_empty > 0:
            d2 = ((X - C[labels]) ** 2).sum(dim=1)
            order = torch.argsort(d2, descending=True)
            newC[~nonempty] = X[order[:n_empty]]
        shift = float((newC - C).abs().max())
        C = newC
        if shift < 1e-7:
            break
    labels = cdist_argmin(X, C)
    cw = torch.zeros(k, dtype=torch.float64, device=X.device)
    cw.scatter_add_(0, labels, wt)
    occupied = int((cw > 0).sum())
    return labels.cpu().numpy(), occupied


def allocate_local_K(group_w, sizes, R):
    """Deterministic largest-remainder split of R across coarse groups.

    Every NONEMPTY group is guaranteed >= 1 (groups are <= R by construction),
    then the remaining budget is split by largest remainder of group weight.
    Returns localK (G,) int64 with sum == min(R, sum(sizes)),
    localK[g] <= sizes[g], and localK[g] >= 1 wherever sizes[g] > 0.
    """
    G = group_w.shape[0]
    R = int(min(R, sizes.sum()))
    nonempty = sizes > 0
    Gne = int(nonempty.sum())
    lk = np.zeros(G, dtype=np.int64)
    if Gne == 0 or R == 0:
        return lk
    # 1 seat per nonempty group first (feasible: Gne <= R by construction)
    lk[nonempty] = 1
    rem = R - Gne
    if rem > 0:
        w = np.where(nonempty, group_w, 0.0)
        raw = rem * w / w.sum() if w.sum() > 0 else np.full(G, rem / G)
        add = np.floor(raw).astype(np.int64)
        add = np.minimum(add, sizes - lk)
        add = np.maximum(add, 0)
        lk += add
        rem2 = R - int(lk.sum())
        frac_order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        i = 0
        while rem2 > 0 and i < 10 * G + 10:
            g = int(frac_order[i % G])
            if lk[g] < sizes[g]:
                lk[g] += 1
                rem2 -= 1
            i += 1
    return lk


def two_level_kmeans(Z, w, R, seed, n, device):
    """~256 coarse groups -> local weighted k-means with largest-remainder K.

    Returns rows (K,) int32 (cluster ids 0..occupied-1, dense), occupied,
    unused budget. Deterministic given (seed, n).
    """
    K = Z.shape[0]
    R = int(min(R, K))
    X = torch.from_numpy(Z).to(device)
    g_coarse = min(COARSE_K, K, R)
    gen = torch_gen(seed, n, 0xC0A15E)
    glabels, _ = weighted_kmeans(X, w, g_coarse, gen, KM_ITERS_COARSE)
    sizes = np.bincount(glabels, minlength=g_coarse).astype(np.int64)
    group_w = np.zeros(g_coarse, dtype=np.float64)
    np.add.at(group_w, glabels, w)
    localK = allocate_local_K(group_w, sizes, R)
    rows = np.empty(K, dtype=np.int32)
    offset = 0
    for g in range(g_coarse):
        kg = int(localK[g])
        members = np.nonzero(glabels == g)[0]
        if kg == 0 or members.size == 0:
            continue
        kg = min(kg, members.size)
        if kg == 1:
            rows[members] = offset
            offset += 1
            continue
        gen_l = torch_gen(seed, n, 0x10CA1 ^ (g * 0x9E3779B9))
        lab, occ = weighted_kmeans(X[members], w[members], kg, gen_l,
                                   KM_ITERS_LOCAL)
        rows[members] = (offset + lab).astype(np.int32)
        offset += kg
    # compact to dense 0..occupied-1 (local re-seeding makes empties rare,
    # but remap defensively); unused = budget never materialized as a row
    uniq = np.unique(rows)
    remap = np.full(int(rows.max()) + 1, -1, dtype=np.int32)
    remap[uniq] = np.arange(uniq.size, dtype=np.int32)
    rows = remap[rows]
    occupied = int(uniq.size)
    unused = int(R - occupied)
    return rows, occupied, unused


# ------------------------------------------------------------------ C3 core
def sparse_rows(q):
    """(K, vocab) dense -> list of (idx, val) per row (only nonzero entries)."""
    K = q.shape[0]
    out_idx = [None] * K
    out_val = [None] * K
    for r in range(K):
        nz = np.nonzero(q[r])[0]
        out_idx[r] = nz
        out_val[r] = q[r, nz].astype(np.float64)
    return out_idx, out_val


def c3_refine(q, w, rows_init, Z, mean, comps, device, log):
    """Predictive-KL exchange refinement of the C1 assignment.

    q (K, vocab) float32 probability rows; w (K,) fit counts; rows_init (K,)
    int32 initial labels (dense 0..K0-1); Z (K, d) PROB projection for ANN
    candidate proposal; mean/comps define the PCA transform used to embed
    centroid PROB vectors sqrt(Q) into the SAME d-dim ANN space (KL itself is
    always evaluated in the original probability space). Per-point exchange
    with IMMEDIATE sufficient-stat commit: for each point (count-descending
    order), propose <=N_CAND candidate centroids by ANN in the projection
    space (including the point's OWN current centroid), evaluate exact KL in
    probability space, accept only a strictly improving move, and correctly
    decrement the old cluster's stats BEFORE evaluating its leave-one-out
    centroid. Empty clusters are rebuilt by splitting the highest-distortion
    contexts (total K fixed). Returns (rows (K,) int32 dense, J trajectory
    list, occupied).
    """
    K, vocab = q.shape
    eps = 1e-12
    q64 = q.astype(np.float64)
    sp_idx, sp_val = sparse_rows(q64)
    negH = np.array([float((v * np.log(v)).sum()) for v in sp_val])

    Kc = int(rows_init.max()) + 1
    labels = rows_init.astype(np.int64).copy()
    S = np.zeros((Kc, vocab), dtype=np.float64)
    W = np.zeros(Kc, dtype=np.float64)
    np.add.at(W, labels, w)
    for a in range(0, K, 8192):
        b = min(a + 8192, K)
        np.add.at(S, labels[a:b], q64[a:b] * w[a:b, None])
    Q = np.maximum(S / np.maximum(W, eps)[:, None], eps)
    Q /= Q.sum(axis=1, keepdims=True)

    comp_t = torch.from_numpy(comps).to(device)
    mean_t = torch.from_numpy(mean).to(device)

    def centroid_ann_vectors():
        """sqrt(Q) projected into the d-dim ANN space (PROB geometry)."""
        sq = torch.from_numpy(np.sqrt(Q).astype(np.float32)).to(device)
        return (sq - mean_t) @ comp_t                    # (Kc, d)

    def point_kl(c, j):
        """w_c * KL(q_c || Q[j]) with current centroid j."""
        v = sp_val[c]
        at = Q[j][sp_idx[c]]
        return float(w[c] * (negH[c] - (v * np.log(at)).sum()))

    def total_J():
        J = 0.0
        for c in range(K):
            J += point_kl(c, labels[c])
        return J

    J = total_J()
    traj = [J]
    log(f"    [c3] init J={J:.6f} K={Kc}")

    order = np.argsort(-w, kind="stable")      # count-descending, ties by idx
    Zt = torch.from_numpy(Z).to(device)
    n_cand = min(N_CAND, Kc)

    for sweep in range(C3_SWEEPS):
        moved = 0
        # ANN proposal in the 128-d projection space (KL stays in prob space)
        Zc = centroid_ann_vectors()
        for a in range(0, K, 4096):
            b = min(a + 4096, K)
            pts = order[a:b]
            x = Zt[pts]
            # chunked cdist against all centroids -> top n_cand-1 nearest
            k_near = min(n_cand - 1, Kc)
            if k_near > 0:
                best_d = None
                best_j = None
                c2 = (Zc * Zc).sum(dim=1)
                x2 = (x * x).sum(dim=1, keepdim=True)
                for c0 in range(0, Kc, CENT_CHUNK):
                    c1 = min(c0 + CENT_CHUNK, Kc)
                    d = x2 - 2.0 * (x @ Zc[c0:c1].T) + c2[c0:c1][None, :]
                    td, tj = torch.topk(d, min(k_near, c1 - c0), largest=False)
                    tj = tj + c0
                    if best_d is None:
                        best_d, best_j = td, tj
                    else:
                        cd = torch.cat([best_d, td], dim=1)
                        cj = torch.cat([best_j, tj], dim=1)
                        best_d, bi = torch.topk(cd, k_near, largest=False)
                        best_j = cj.gather(1, bi)
                near = best_j.cpu().numpy()
            else:
                near = np.empty((pts.size, 0), dtype=np.int64)
            cand = np.concatenate([labels[pts][:, None], near], axis=1)

            # per-point exact-KL exchange with immediate commit
            for i in range(pts.size):
                c = int(pts[i])
                j_old = labels[c]
                v = sp_val[c]
                idx = sp_idx[c]
                cur = float(w[c] * (negH[c] - (v * np.log(Q[j_old][idx])).sum()))
                best_val = cur
                best_j = j_old
                for j in cand[i]:
                    j = int(j)
                    if j == j_old or W[j] <= 0:
                        continue
                    val = float(w[c] * (negH[c] - (v * np.log(Q[j][idx])).sum()))
                    if val < best_val - 1e-12:
                        best_val = val
                        best_j = j
                if best_j != j_old:
                    # decrement old cluster FIRST (leave-one-out correctness),
                    # then credit the new cluster
                    wc = w[c]
                    W[j_old] -= wc
                    S[j_old] -= wc * q64[c]
                    W[best_j] += wc
                    S[best_j] += wc * q64[c]
                    labels[c] = best_j
                    for j in (j_old, best_j):
                        if W[j] > 0:
                            Q[j] = np.maximum(S[j] / W[j], eps)
                            Q[j] /= Q[j].sum()
                    moved += 1
            # refresh ANN centroids after each chunk (stats changed)
            if moved:
                Zc = centroid_ann_vectors()
        J_new = total_J()
        if J_new > J + 1e-6 * max(1.0, abs(J)):
            raise AssertionError(f"C3 J increased: {J} -> {J_new} (sweep {sweep})")
        J = J_new
        traj.append(J)
        log(f"    [c3] sweep {sweep}: moved={moved} J={J:.6f}")
        if moved == 0:
            break

    # rebuild empty clusters by splitting highest-distortion contexts
    occ_mask = W > 0
    n_empty = int((~occ_mask).sum())
    if n_empty > 0:
        dist = np.array([point_kl(c, labels[c]) for c in range(K)])
        worst = np.argsort(-dist, kind="stable")
        empties = np.nonzero(~occ_mask)[0]
        ei = 0
        for c in worst:
            if ei >= n_empty:
                break
            c = int(c)
            j_old = labels[c]
            if W[j_old] - w[c] <= 0:
                continue                       # would empty the donor
            j_new = int(empties[ei])
            W[j_old] -= w[c]
            W[j_new] += w[c]
            S[j_old] -= w[c] * q64[c]
            S[j_new] += w[c] * q64[c]
            labels[c] = j_new
            ei += 1
        for j in range(Kc):
            if W[j] > 0:
                Q[j] = np.maximum(S[j] / W[j], eps)
                Q[j] /= Q[j].sum()
        J = total_J()
        traj.append(J)
        log(f"    [c3] rebuilt {ei} empty clusters, J={J:.6f}")

    occ_ids = np.unique(labels)
    remap = np.full(Kc, -1, dtype=np.int64)
    remap[occ_ids] = np.arange(occ_ids.size)
    labels = remap[labels]
    return labels.astype(np.int32), traj, int(occ_ids.size)


# ---------------------------------------------------------------------- main
def build(index_path, stats_dir, algo, out_path, seed=ROUTING_SEED,
          order=None, vocab=None, m_total=None, log=print):
    assert algo in ("c1", "c2", "c3")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    log(f"[{algo}] device={device} seed={seed}")
    sel, counts_total = load_selected(index_path)
    R = allocate_R(sel) if m_total is None else allocate_R(sel, m_total=m_total)
    log(f"[{algo}] selected: " + ", ".join(f"n{n}={sel[n][0].size}" for n in ORDERS))
    log(f"[{algo}] R allocation: {R}")

    uni = np.load(os.path.join(stats_dir, "unigram.npy"))
    if vocab is None:
        vocab = int(uni.shape[1])
    u_fit = uni[0].astype(np.float64) + uni[1].astype(np.float64) + 1.0
    q_uni_fit = u_fit / u_fit.sum()
    q_uni_fit = np.maximum(q_uni_fit, UNIGRAM_FLOOR)
    q_uni_fit /= q_uni_fit.sum()
    u_all = uni.sum(axis=0).astype(np.float64) + 1.0
    q_uni_all = u_all / u_all.sum()
    q_uni_all = np.maximum(q_uni_all, UNIGRAM_FLOOR)
    q_uni_all /= q_uni_all.sum()

    orders = [order] if order is not None else list(ORDERS)
    rows_map = {}
    meta = {"algo": algo, "seed": seed, "fmin": 256,
            "m_total": int(sum(R.values())),
            "R": {str(n): int(R[n]) for n in ORDERS},
            "alpha_grid": [int(a) for a in ALPHA_GRID],
            "alpha_refit_on_all_folds": True,
            "proj_dim": PROJ_DIM, "coarse_k": COARSE_K, "n_cand": N_CAND,
            "c3_sweeps": C3_SWEEPS, "device": device}
    profile = {}
    lower_keys = None
    lower_q_fit = None
    lower_q_all = None

    for n in ORDERS:
        if n not in orders:
            if any(o > n for o in orders):
                raise SystemExit("subset runs must include consecutive orders "
                                 "from 2 up (suffix backoff needs order n-1)")
            continue
        tn = time.time()
        z = np.load(os.path.join(stats_dir, f"hits_n{n}.npz"))
        sel_idx, y, fold, suffix = z["sel_idx"], z["y"], z["fold"], z["suffix"]
        keys = sel[n][0]
        K = keys.size
        log(f"[{algo}] n={n}: K={K} hits={sel_idx.size}")

        ctx, tok, cnt, F, ctx_suffix = reduce_hits(sel_idx, y, fold, suffix, K, vocab)
        if n == 2:
            back_fit = np.broadcast_to(q_uni_fit.astype(np.float32), (K, vocab))
            back_all = np.broadcast_to(q_uni_all.astype(np.float32), (K, vocab))
        else:
            srows = suffix_rows(ctx_suffix, lower_keys)
            back_fit = lower_q_fit[srows]          # float32 view-copy (gather)
            back_all = lower_q_all[srows]

        # alpha selection (fit folds 0+1, dev fold 2), then refit on all folds
        alpha_curve = {}
        best_alpha, best_nll = None, np.inf
        q_fit_best = None
        for alpha in ALPHA_GRID:
            q_fit = build_q(ctx, tok, cnt, F, K, vocab, alpha, back_fit)
            nll = dev_nll_of(q_fit, sel_idx, y, fold)
            alpha_curve[str(int(alpha))] = nll
            if nll < best_nll:
                best_nll, best_alpha, q_fit_best = nll, alpha, q_fit
        log(f"[{algo}] n={n} alpha curve (dev NLL): "
            + " ".join(f"a={a}:{v:.4f}" for a, v in alpha_curve.items())
            + f" -> alpha={int(best_alpha)}")

        # refit counts on ALL folds with the chosen alpha (declared in meta)
        si = sel_idx.astype(np.int64)
        yy = y.astype(np.int64)
        lin = si * vocab + yy
        u, c = np.unique(lin, return_counts=True)
        ctx_a = (u // vocab).astype(np.int64)
        tok_a = (u % vocab).astype(np.int64)
        F_all = np.zeros(K, dtype=np.float64)
        np.add.at(F_all, ctx_a, c.astype(np.float64))
        q_all = build_q(ctx_a, tok_a, c.astype(np.float64), F_all, K, vocab,
                        best_alpha, back_all)

        sqrt_q = np.sqrt(q_all)
        if algo == "c2":
            if n == 2:
                feat = sqrt_q - np.sqrt(q_uni_all)[None, :]
            else:
                feat = sqrt_q - np.sqrt(lower_q_all[srows])
            feat = feat.astype(np.float32)
        else:
            feat = sqrt_q.astype(np.float32)

        t_pca = time.time()
        mean, comps = fit_pca(feat, F, seed, n, device)
        Z = project(feat, mean, comps, device)
        t_pca = time.time() - t_pca

        t_km = time.time()
        if algo == "c3":
            rows0, occ0, unused0 = two_level_kmeans(Z, F, R[n], seed, n, device)
            log(f"[{algo}] n={n} C1-init occupied={occ0}/{R[n]} unused={unused0}")
            rows, traj, occ = c3_refine(q_all, F, rows0, Z, mean, comps,
                                        device, log)
            unused = max(int(R[n] - occ), 0)
            meta[f"J_traj_{n}"] = [round(j, 6) for j in traj]
        else:
            rows, occ, unused = two_level_kmeans(Z, F, R[n], seed, n, device)
        t_km = time.time() - t_km

        assert rows.min() >= 0 and rows.max() < occ
        assert np.unique(rows).size == occ
        rows_map[n] = rows
        meta[f"alpha_{n}"] = int(best_alpha)
        meta[f"alpha_curve_{n}"] = {k: round(v, 6) for k, v in alpha_curve.items()}
        meta[f"R_{n}"] = int(R[n])
        meta[f"occupied_{n}"] = int(occ)
        meta[f"unused_{n}"] = int(unused)
        meta[f"dev_nll_exact_{n}"] = round(best_nll, 6)
        profile[f"n{n}"] = {"pca_s": round(t_pca, 1), "cluster_s": round(t_km, 1),
                            "total_s": round(time.time() - tn, 1), "K": int(K)}
        log(f"[{algo}] n={n}: occupied={occ}/{R[n]} unused={unused} "
            f"pca={t_pca:.0f}s cluster={t_km:.0f}s total={time.time()-tn:.0f}s")

        lower_keys = keys
        lower_q_fit = q_fit_best
        lower_q_all = q_all

    meta["profile"] = profile
    meta["elapsed_s"] = round(time.time() - t0, 1)
    write_assignment(out_path, sel, R, rows_map, counts_total, meta)
    return rows_map, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("index_path")
    ap.add_argument("stats_dir")
    ap.add_argument("algo", choices=["c1", "c2", "c3"])
    ap.add_argument("out_path")
    ap.add_argument("--seed", type=int, default=ROUTING_SEED)
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--m-total", type=int, default=None,
                    help="physical row budget summed over orders (default 131072)")
    a = ap.parse_args()
    build(a.index_path, a.stats_dir, a.algo, a.out_path, seed=a.seed,
          order=a.order, m_total=a.m_total)
