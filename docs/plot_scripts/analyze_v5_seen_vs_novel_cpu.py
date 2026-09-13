"""CPU analysis of per-continuation decomposition at one eval step (2020).

From offline npz: per-context continuation rows (c3, count, mass), GT missing mass.
From exact_freq_loss.jsonl: per-f seen-continuation loss (val mean_loss) + token counts.
Weighting: d_e(f) = [Tok/Tok+Mis] * mean_loss_seen + [Mis/Tok+Mis] * mass_loss_novel,
so per-f: seen contribution d_seen(f) = share*mean_loss_seen (with mass_loss_novel
estimated from the aggregate), and novel share Mis(f)/(Tok(f)+Mis(f)) vs missing
mass M_f tells whether *amplitude* or *share* drives the growth.
Also bigram-trigram comparison (trigram has no offline GT here -> shape only).

Run: ophis CPU (network small), outputs numbers only; GPU batch comes after.
"""
import json, glob
import numpy as np

def load(run):
    return {r["step"]: r for l in open(glob.glob(f"data/runs_fixed/{run}_fixed/exact_freq_loss.jsonl")[0])
            for r in [json.loads(l)]}

IN = load("nglab1x_input_v5_128x_freq10_fd")
NOG = load("nglab1x_nogram_v5_128x_freq10_fd")
z = np.load("data/freq_index.npz"); V = int(z["vocab_size"][0])
tk, tc = z["trigram_keys"], z["trigram_counts"].astype(np.int64)
ctx = tk // V
uctx, inv = np.unique(ctx, return_inverse=True)
f_ctx = np.bincount(inv, weights=tc)
n1 = np.bincount(inv, weights=(tc == 1)).astype(np.int64)
mass = np.bincount(inv, weights=tc).astype(np.int64)
print("check mass==f:", np.all(mass == f_ctx))

edges = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 180, 320, 560, 1000, 1800, 3200, 5600, 10000, 20000, 300000])
def bins(f):
    i = np.searchsorted(edges, f, side="right") - 1
    return np.clip(i, 0, len(edges) - 2)
bi = bins(f_ctx)
NB = len(edges) - 1
def agg(arr):
    return np.array([arr[bi == i].sum() for i in range(NB)])
TOKb, MISb = agg(f_ctx), agg(n1)          # token mass, missing(=novel) mass per bin

def perbin(rec, branch, side):
    d = rec[side][branch]
    num = np.zeros(NB); tok = np.zeros(NB)
    for k, v in d.items():
        i = np.searchsorted(edges, int(k), side="right") - 1
        if 0 <= i < NB: num[i] += v["loss_sum"]; tok[i] += v["token_count"]
    return num, tok

avail = sorted(set(IN) & set(NOG))
S = [min(avail, key=lambda x: abs(x - t)) for t in (337, 674, 1011, 1348, 1685, 2020)]
print(f"\n== bigram val damage d_seen(f) = loss_sum^IN/loss_mass-weight? see below ==")
print("rows: pass; cols: f-bin; value = d_seen(f) = (loss_sum_IN - loss_sum_NOG)/ (TOK+ MIS) per bin (nats, contribution to gap)")
print("   and [share of novel mass] = MIS/(TOK+MIS)")
for s in S:
    a, ta = perbin(IN[s], "bigram", "val"); b, tb = perbin(NOG[s], "bigram", "val")
    assert np.allclose(ta, tb, rtol=1e-3)
    dtot = a - b
    w = TOKb + MISb
    d_norm = dtot / w
    print(f"p{round(s/337)}: " + " ".join(f"{x:5.2f}" for x in d_norm[:14]))
print("novel share: " + " ".join(f"{x:4.2f}" for x in (MISb / (TOKb + MISb))[:14]))
print("\n== same, trigram (share unknown -> d_seen only, per seen-token) ==")
for s in S:
    a, ta = perbin(IN[s], "trigram", "val"); b, tb = perbin(NOG[s], "trigram", "val")
    print(f"p{round(s/337)}: " + " ".join(f"{x:5.2f}" for x in ((a - b) / ta)[:14]))
