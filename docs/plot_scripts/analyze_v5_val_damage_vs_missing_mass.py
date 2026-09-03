"""Zero-GPU: is memorization 'democratic' across context frequency f, or more
stubborn (偏执) on rare contexts?  Bigram branch (2-token context), 1x train shard.

Offline from data/freq_index.npz (trigram keys give continuation counts of every
bigram context):  per exact f  H_f = token-weighted empirical conditional entropy,
M_f = Good-Turing missing mass = N1/f.
From exact_freq_loss.jsonl (input vs nogram fd arms, 128x, seed 42) at pass
boundaries: val damage d_e(f) = L_val^input(f) - L_val^nogram(f), compared with M_f.
Test of the factorization  d_e(f) = A_e * M_f  (shape = coverage, amplitude = dynamics).
NOTE: a 'memorization fraction' relative to H_f is NOT well defined here — the backbone
uses the full 2048-token context, so its loss is below the 2-token empirical entropy H_f
for f >~ 100 already at pass 1 (checked; not plotted).
Output: docs/figs/theory/fig_v5_memorization_fraction_bigram.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/runs_fixed"
IN, NOG = "nglab1x_input_v5_128x_freq10_fd_fixed", "nglab1x_nogram_v5_128x_freq10_fd_fixed"
P = 337

z = np.load(ROOT / "data/freq_index.npz"); V = int(z["vocab_size"][0])
tk, tc = z["trigram_keys"], z["trigram_counts"].astype(np.int64)
bk, bc = z["bigram_keys"], z["bigram_counts"].astype(np.int64)
ctx = tk // V                                     # bigram context of each trigram key
uctx, inv = np.unique(ctx, return_inverse=True)
f_ctx = np.bincount(inv, weights=tc).astype(np.int64)
bmap = dict(zip(bk.tolist(), bc.tolist()))
chk = np.array([bmap.get(int(c), -1) for c in uctx[:20000]])
assert np.mean(chk == f_ctx[:20000]) > 0.99, "trigram->bigram context sum mismatch"
n1 = np.bincount(inv, weights=(tc == 1)).astype(np.int64)
nlogn = np.bincount(inv, weights=tc * np.log(tc))
H_ctx = np.log(f_ctx) - nlogn / f_ctx             # empirical conditional entropy (nats)
# per exact f, token-weighted
fmax = int(f_ctx.max())
tok_f = np.bincount(f_ctx, weights=f_ctx, minlength=fmax + 1)
H_f = np.divide(np.bincount(f_ctx, weights=f_ctx * H_ctx, minlength=fmax + 1), tok_f, out=np.zeros(fmax + 1), where=tok_f > 0)
M_f = np.divide(np.bincount(f_ctx, weights=n1, minlength=fmax + 1), tok_f, out=np.zeros(fmax + 1), where=tok_f > 0)

def load(run):
    out = {}
    for l in open(RUNS / run / "exact_freq_loss.jsonl"):
        r = json.loads(l); out[r["step"]] = r
    return out
A, B = load(IN), load(NOG)
steps = sorted(A)
def at(e): return min(steps, key=lambda s: abs(s - min(e * P, 2020)))

edges = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 180, 320, 560, 1000, 1800, 3200, 5600, 10000, 20000, 200000])
def binned(rec, side, key):
    """token-weighted mean over exact-f entries falling in each log bin"""
    d = rec[side]["bigram"]; num = np.zeros(len(edges) - 1); den = np.zeros(len(edges) - 1)
    for k, v in d.items():
        f = int(k); i = np.searchsorted(edges, f, side="right") - 1
        if 0 <= i < len(num): num[i] += v[key] * v["token_count"]; den[i] += v["token_count"]
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)
def binned_offline(arr):
    num = np.zeros(len(edges) - 1); den = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        f = np.arange(edges[i], min(edges[i + 1], fmax + 1)); w = tok_f[f]
        num[i] = (arr[f] * w).sum(); den[i] = w.sum()
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)
Hb, Mb = binned_offline(H_f), binned_offline(M_f)
xc = np.sqrt(edges[:-1] * edges[1:])

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
mid = slice(1, 11)   # f in [2, 320): the region where M_f is well estimated and d/M is flat
amp = []
print(f"{'pass':>4} | d_e / M_f at f-bins  [1] [3-4] [8-12] [20-34] [60-99] [180-319] [560-999] [1800-3199] [5600-9999] | A_e=median(f∈[2,320))  CV")
for e, c in zip(range(1, 7), cols):
    s = at(e); ra, rb = A[s], B[s]
    Vin, Vnog = binned(ra, "val", "mean_loss"), binned(rb, "val", "mean_loss")
    d = Vin - Vnog; r = d / Mb
    Ae = np.nanmedian(r[mid]); cv = np.nanstd(r[mid]) / abs(Ae); amp.append((e, Ae, cv))
    ax[0].plot(xc, d, "o-", ms=4, lw=1, color=c, label=f"pass {e} (step {s})")
    ax[1].plot(xc, r, "o-", ms=4, lw=1, color=c, label=f"pass {e}")
    pick = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    print(f"{e:>4} | " + " ".join(f"{r[i]:5.1f}" for i in pick) + f" | {Ae:5.2f}  {cv:.2f}")
ax[0].set_xscale("log"); ax[0].set_xlabel("exact train hit-count f of bigram context"); ax[0].set_ylabel("val damage d_e(f) = L_val^input − L_val^nogram (nats)")
ax[0].set_title("(a) val damage per context frequency"); ax[0].legend(fontsize=8)
axb = ax[0].twinx(); axb.plot(xc, Mb, "k--", lw=1, label="Good-Turing missing mass M_f = N1/f"); axb.plot(xc, Hb, "k-.", lw=1, label="empirical cond. entropy H_f (nats)")
axb.set_ylabel("offline data statistics (1x train shard)"); axb.legend(fontsize=7, loc="upper right")
ax[1].set_xscale("log"); ax[1].axvspan(2, 320, color="0.9", zorder=0); ax[1].set_xlabel("f"); ax[1].set_ylabel("d_e(f) / M_f")
ax[1].set_title("(b) val damage per unit missing mass\nflat over f∈[2,320) ⇒ d_e(f) ≈ A_e·M_f; rises at high f (misweighted-seen term)")
ax[1].legend(fontsize=8)
ep = np.array([a[0] for a in amp]); Ae = np.array([a[1] for a in amp])
ax[2].plot(ep, Ae, "o-", color="#d62728", label="A_e = median d_e/M_f, f∈[2,320)")
k = ep >= 2; a_, b_ = np.polyfit(ep[k], Ae[k], 1); ax[2].plot(ep, a_ * ep + b_, "k--", lw=1, label=f"linear fit e≥2: {a_:.2f}/pass")
ax[2].set_xlabel("pass"); ax[2].set_ylabel("amplitude A_e (nats per unit missing mass)")
ax[2].set_title("(c) amplitude grows with training, shape does not"); ax[2].legend(fontsize=8)
fig.suptitle("bigram branch (2-token context) · nglab1x_{input,nogram}_v5_128x_freq10_fd · seed 42 · offline stats from 1x train shard freq_index", fontsize=10)
fig.tight_layout()
out = ROOT / "docs/figs/theory/fig_v5_val_damage_vs_missing_mass_bigram.png"; fig.savefig(out, dpi=140); print("saved", out)
print("offline: contexts", len(uctx), "tokens", int(tok_f.sum()), "| token mass with f=1:", round(tok_f[1] / tok_f.sum(), 4), "| overall GT missing mass:", round(n1.sum() / tok_f.sum(), 4))
print("M_f bins:", np.round(Mb, 3))
