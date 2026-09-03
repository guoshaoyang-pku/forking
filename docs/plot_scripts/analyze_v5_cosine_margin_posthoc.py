"""Post-hoc analysis from replay snapshots (CPU).

A. Table per-row write trajectory (runs_theory/input_snap/rows_*.npy):
   cos(row@337, row@e) and ||row@e||/||row@337|| per hit-count f-bin,
   plus per-pass write intensity ||W_e - W_{e-1}||_F / ||W_{e-1}||_F
   (answer: does the table keep rotating or only grow in scale?)

B. Seen/novel/margin decomposition (decomp.jsonl of input/nogram/freezebb):
   per pass e, per f-bin:
     d_seen(f)  = seen-token mean loss(input) - mean loss(nogram)
     d_novel(f) = novel-token mean loss(input) - mean loss(nogram)
     share_novel(f) from offline counts (GT: Mis(f)/(Tok(f)+Mis(f)))
     margin(f) for seen / novel tokens (logit dominant - logit true)
   Tests: (i) does per-token novel loss keep growing (suppression dynamics);
          (ii) does margin grow monotonically with passes.
Output: docs/figs/theory/fig_v5_table_cosine_and_margin.png + printed tables.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data/runs_theory/input_snap"
OUT = ROOT / "docs/figs/theory/fig_v5_table_cosine_and_margin.png"
P = 337
EDGES = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 180, 320, 560, 1000, 1800, 3200, 5600, 10000, 20000, 300000])
STEPS = [337, 674, 1011, 1348, 1685, 2022]

# ---------- A. table trajectory ----------
z = np.load(ROOT / "data/freq_index.npz")
R = 1 << 20
MASK64 = (1 << 64) - 1
P_BI = (2654435761, 2246822519)      # _BASE_BIGRAM_PRIMES[0][0], layer1 K=1
P_TRI = (16777619, 2166136261, 3432918353)  # _BASE_TRIGRAM_PRIMES[0][:3], layer1 K=1
bk, bc = z["bigram_keys"], z["bigram_counts"].astype(np.int64)
tk, tc = z["trigram_keys"], z["trigram_counts"].astype(np.int64)
V = int(z["vocab_size"][0])
c1 = bk // V; c2 = bk % V
b_row = ((c1.astype(np.uint64) * P_BI[0] & MASK64) ^ (c2.astype(np.uint64) * P_BI[1] & MASK64)) % R
t_ctx = tk // V
t1 = t_ctx // V; t2 = t_ctx % V
t_row = (((t1.astype(np.uint64) * P_TRI[0] & MASK64)
          ^ (t2.astype(np.uint64) * P_TRI[1] & MASK64)
          ^ ((t_ctx % V).astype(np.uint64) * P_TRI[2] & MASK64)) % R)
f_b_row = np.bincount(b_row, weights=bc, minlength=R)
f_t_row = np.bincount(t_row, weights=tc, minlength=R)

def load_rows(name, step):
    return np.load(SNAP / f"rows_{name}_step{step}.npy").astype(np.float32)

W = {br: [load_rows(br, s) for s in STEPS] for br in ("bigram", "trigram")}

def cos_per_bin(W0, We, f_row, edges):
    a = W0 / (np.linalg.norm(W0, axis=1, keepdims=True) + 1e-12)
    b = We / (np.linalg.norm(We, axis=1, keepdims=True) + 1e-12)
    cos = (a * b).sum(1)
    nrm = np.linalg.norm(We, axis=1) / (np.linalg.norm(W0, axis=1) + 1e-12)
    hit = f_row > 0
    idx = np.searchsorted(edges, f_row[hit], side="right") - 1
    nb = len(edges) - 1
    cs = np.array([np.median(cos[hit][idx == i]) if (idx == i).any() else np.nan for i in range(nb)])
    ns = np.array([np.median(nrm[hit][idx == i]) if (idx == i).any() else np.nan for i in range(nb)])
    return cs, ns

print("== A. table trajectory (rows vs end-of-pass-1 snapshot) ==")
for br, f_row in (("bigram", f_b_row), ("trigram", f_t_row)):
    Ws = W[br]
    for e in range(2, 7):
        cs, ns = cos_per_bin(Ws[0], Ws[e - 1], f_row, EDGES)
        sel = [1, 3, 5, 7, 9, 11, 13, 15, 17]
        print(f"  {br} cos(row@e1,row@e{e}) bins {[str(EDGES[i]) for i in sel]}:",
              " ".join(f"{cs[i]:.3f}" for i in sel))
        print(f"  {br} norm ratio e{e}/e1:", " ".join(f"{ns[i]:.2f}" for i in sel))
    # per-pass write intensity
    for e in range(2, 7):
        d = np.linalg.norm((Ws[e - 1] - Ws[e - 2]).ravel()) / np.linalg.norm(Ws[e - 2].ravel())
        print(f"  {br} relative write pass {e}: ||ΔW||/||W|| = {d:.4f}")

# ---------- B. decomp ----------
def load_decomp(path):
    rows = {}
    for l in open(path):
        r = json.loads(l)
        rows.setdefault(r["step"], {})[r["f"]] = r
    return rows
DI = load_decomp(ROOT / "data/runs_theory/input_snap/decomp.jsonl")
DN = load_decomp(ROOT / "data/runs_theory/nogram_snap/decomp.jsonl")
DF = load_decomp(ROOT / "data/runs_theory/freezebb_snap/decomp.jsonl")

# offline novel share per f
n1_by_ctx = np.bincount(tk // V, weights=(tc == 1))
mass_ctx = np.bincount(tk // V, weights=tc)
f_ctx = mass_ctx.copy()
def novel_share(f):
    sel = f_ctx == f
    tok = mass_ctx[sel].sum(); mis = n1_by_ctx[sel].sum()
    return mis / (tok + mis) if tok > 0 else np.nan

def bin_stats(D, step, f_lo, f_hi):
    n_s = n_n = ls_t = ln_t = ls_c = ln_c = 0.0
    mg_s = mg_sn = mg_n = mg_nn = 0.0
    for f, r in D[step].items():
        if not (f_lo <= f < f_hi):
            continue
        n_s += r["n_seen"]; n_n += r["n_novel"]
        ls_t += r["loss_seen_t"]; ln_t += r["loss_novel_t"]
        ls_c += r["loss_seen_c"]; ln_c += r["loss_novel_c"]
        mg_s += r["marg_seen_sum"]; mg_sn += r["marg_seen_n"]
        mg_n += r["marg_novel_sum"]; mg_nn += r["marg_novel_n"]
    return dict(n_seen=n_s, n_novel=n_n,
                seen_t=ls_t / max(n_s, 1), novel_t=ln_t / max(n_n, 1),
                seen_c=ls_c / max(n_s, 1), novel_c=ln_c / max(n_n, 1),
                marg_seen=mg_s / max(mg_sn, 1), marg_novel=mg_n / max(mg_nn, 1))

print("\n== B. seen/novel decomposition (input − nogram), per pass ==")
print(f"{'pass':>4} {'f-bin':>10} {'d_seen':>7} {'d_novel':>8} {'novel_share_GT':>14} {'marg_seen_in':>12} {'marg_novel_in':>13}")
for e, s in enumerate(STEPS, 1):
    if s not in DI or s not in DN:
        continue
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        if lo > 10000:
            break
        a = bin_stats(DI, s, lo, hi); b = bin_stats(DN, s, lo, hi)
        if a["n_seen"] + a["n_novel"] < 50:
            continue
        ns = np.nanmean([novel_share(f) for f in range(lo, min(hi, 2000))])
        print(f"{e:>4} {f'[{lo},{hi})':>10} {a['seen_t']-b['seen_t']:7.2f} {a['novel_t']-b['novel_t']:8.2f} "
              f"{ns:14.2f} {a['marg_seen']:12.2f} {a['marg_novel']:13.2f}")
    print()
