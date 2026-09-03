"""Figure: table row cosine trajectory + seen/novel/margin decomposition.

Sources: data/runs_theory/{input,nogram,freezebb}_snap/decomp.jsonl (GPU replay,
2022 steps, seed 42, 128x, step-for-step identical to the authoritative runs)
+ rows_{bigram,trigram}_step{337..2022}.npy snapshots (input arm).
Output: docs/figs/theory/fig_v5_table_cosine_and_margin.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
THE = ROOT / "data/runs_theory"
OUT = ROOT / "docs/figs/theory/fig_v5_table_cosine_and_margin.png"
EDGES = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 180, 320, 560, 1000, 1800, 3200, 5600, 10000, 20000])
STEPS = [337, 674, 1011, 1348, 1685, 2022]
XC = np.sqrt(EDGES[:-1] * EDGES[1:])

z = np.load(ROOT / "data/freq_index.npz")
V = int(z["vocab_size"][0]); R = 1 << 20; M64 = (1 << 64) - 1
tk, tc = z["trigram_keys"], z["trigram_counts"].astype(np.int64)
bk, bc = z["bigram_keys"], z["bigram_counts"].astype(np.int64)
c1, c2 = bk // V, bk % V
b_row = ((c1.astype(np.uint64) * 2654435761 & M64) ^ (c2.astype(np.uint64) * 2246822519 & M64)) % R
t1, t2 = tk // (V * V), (tk % (V * V)) // V
c3 = tk % V
t_row = ((t1.astype(np.uint64) * 16777619 & M64) ^ (t2.astype(np.uint64) * 2166136261 & M64)
         ^ (c3.astype(np.uint64) * 3432918353 & M64)) % R
f_b = np.bincount(b_row, weights=bc, minlength=R)
f_t = np.bincount(t_row, weights=tc, minlength=R)

def load_rows(br, s):
    return np.load(THE / "input_snap" / f"rows_{br}_step{s}.npy").astype(np.float32)

def cos_bins(W0, We, f_row):
    a = W0 / (np.linalg.norm(W0, axis=1, keepdims=True) + 1e-12)
    b = We / (np.linalg.norm(We, axis=1, keepdims=True) + 1e-12)
    cos = (a * b).sum(1)
    hit = f_row > 0
    idx = np.searchsorted(EDGES, f_row[hit], side="right") - 1
    return np.array([np.median(cos[hit][idx == i]) if (idx == i).any() else np.nan
                     for i in range(len(EDGES) - 1)])

def load_decomp(run):
    d = {}
    for l in open(THE / f"{run}_snap/decomp.jsonl"):
        r = json.loads(l); d.setdefault(r["step"], {})[r["f"]] = r
    return d
DI, DN, DF = load_decomp("input"), load_decomp("nogram"), load_decomp("freezebb")

def binned(D, step, key_a, key_b):
    num = np.zeros(len(EDGES) - 1); den = np.zeros(len(EDGES) - 1)
    for f, r in D[step].items():
        i = np.searchsorted(EDGES, f, side="right") - 1
        if 0 <= i < len(num):
            num[i] += r[key_a]; den[i] += r[key_b]
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)

fig, ax = plt.subplots(1, 4, figsize=(21, 5.2))
cols = plt.cm.viridis(np.linspace(0, 0.9, 6))

# (a) table cos per f-bin, bigram + trigram
for br, f_row, ls in (("bigram", f_b, "-"), ("trigram", f_t, "--")):
    W0 = load_rows(br, 337)
    for e, c in zip((2, 3, 4, 6), (cols[1], cols[2], cols[3], cols[5])):
        ax[0].plot(XC, cos_bins(W0, load_rows(br, STEPS[e - 1]), f_row), ls, color=c, lw=1.4,
                   label=f"{br} e{e}" if br == "bigram" else f"{br} e{e}")
ax[0].set_xscale("log"); ax[0].set_xlabel("context hit count f (per table row)")
ax[0].set_ylabel("median cos( row@end pass 1 , row@end pass e )")
ax[0].set_title("(a) table rows KEEP ROTATING after pass 1\nwrite-once falsified: cos falls to 0.52–0.57 by pass 6")
ax[0].legend(fontsize=7, ncol=2); ax[0].set_ylim(0.4, 1.0)

# (b) relative write intensity per pass
for br, c in (("bigram", "#1f77b4"), ("trigram", "#d62728")):
    Ws = [load_rows(br, s) for s in STEPS]
    wr = [np.linalg.norm((Ws[e] - Ws[e - 1]).ravel()) / np.linalg.norm(Ws[e - 1].ravel())
          for e in range(1, 6)]
    ax[1].plot(range(2, 7), wr, "o-", color=c, label=f"{br}: ||W_e−W_{{e−1}}||_F / ||W_{{e−1}}||_F")
    # scale-only reference: what the write would be if only norms grew (cos=1)
    nr = [np.linalg.norm(Ws[e].ravel()) / np.linalg.norm(Ws[e - 1].ravel()) - 1 for e in range(1, 6)]
    ax[1].plot(range(2, 7), nr, "o:", color=c, label=f"{br} scale-only reference (cos=1)")
ax[1].set_xlabel("pass"); ax[1].set_ylabel("relative write per pass")
ax[1].set_title("(b) write intensity per pass:\n3–6× the scale-only reference → genuine re-writing")
ax[1].legend(fontsize=7)

# (c) seen vs novel val damage (input − nogram), pass 2,4,6
for e, c in zip((2, 4, 6), (cols[1], cols[3], cols[5])):
    s = STEPS[e - 1]
    ds_i = binned(DI, s, "loss_seen_t", "n_seen"); ds_n = binned(DN, s, "loss_seen_t", "n_seen")
    dn_i = binned(DI, s, "loss_novel_t", "n_novel"); dn_n = binned(DN, s, "loss_novel_t", "n_novel")
    ax[2].plot(XC, dn_i - dn_n, "-", color=c, lw=1.6, label=f"novel tokens, pass {e}")
    ax[2].plot(XC, ds_i - ds_n, "--", color=c, lw=1.2, label=f"seen tokens, pass {e}")
ax[2].set_xscale("log"); ax[2].set_xlabel("f"); ax[2].set_ylabel("val loss(input) − val loss(nogram)")
ax[2].set_title("(c) damage split by continuation identity:\nnovel-token suppression dominates; seen damage smaller,\nhigh-f seen damage = sampling-error term")
ax[2].legend(fontsize=7, ncol=2); ax[2].axhline(0, color="k", lw=0.6)

# (d) margin vs pass, input vs freeze_backbone_e1, novel tokens
for fsel, mk in ((1, "o"), (13, "s"), (100, "^")):
    i = list(EDGES).index(fsel)
    mg_in, mg_fb = [], []
    for s in STEPS:
        a = binned(DI, s, "marg_novel_sum", "marg_novel_n")[i]
        b = binned(DF, s, "marg_novel_sum", "marg_novel_n")[i]
        mg_in.append(a); mg_fb.append(b)
    ax[3].plot(range(1, 7), mg_in, mk + "-", color="#1f77b4", ms=5, lw=1.2,
               label=f"input f={fsel}")
    ax[3].plot(range(1, 7), mg_fb, mk + "--", color="#d62728", ms=5, lw=1.2,
               label=f"freeze_backbone@e1 f={fsel}")
ax[3].set_xlabel("pass"); ax[3].set_ylabel("margin: logit(train-dominant cont.) − logit(true novel token)")
ax[3].set_title("(d) suppression margin keeps growing with passes\n(backbone-driven: flat after freeze_backbone@e1)")
ax[3].legend(fontsize=7, ncol=2)

fig.suptitle("v5 128x · seed 42 · table write trajectory + seen/novel/margin decomposition (step-for-step replay, val fixed pool)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("saved", OUT)
