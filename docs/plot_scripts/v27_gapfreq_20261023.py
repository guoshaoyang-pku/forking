#!/usr/bin/env python3
"""Per-order gap-vs-context-frequency at E3 end (v27, 50M, same-seed paired dumps).

Inputs (.tmp/tokstats/): token_stats_step*_{val,train}.npz of an exact arm and its same-seed
nogram (same val chunks / same final train batch => position-aligned), the 1B-pool f30 index
(counts >= 30) and the arm's assign npz (table rows).
Frequency axis: expected count of the context in the run's own pool = c_1B * D / 0.9999e9
(prefix pools are subsets of the 1B pool; c<30 in the 1B pool is the "rare" bin).
Per order n and bin: dTrain = nog - arm on the final train batch, dVal = nog - arm on val
(positive = OE wins), dGap = dTrain - dVal (positive = forking added). mK.
Output: docs/figs/v27_gapfreq_20261023/{fig_gapfreq.png,svg,gapfreq.json}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab"
TS = f"{ROOT}/.tmp/tokstats"
OUT = f"{ROOT}/docs/figs/v27_gapfreq_20261023"
os.makedirs(OUT, exist_ok=True)
IDX = np.load(f"{TS}/freq_index_o2to5_1b_f30.npz")
POOL1B = 999_899_136
CASES = [  # label, exact run, nogram run, assign, D tokens, colour, ls
    ("D/N=2  R/N=8", "v27_50m_exact_r400k_dn2_tls1_s42_e3_fixed", "v27_50m_nogram_r400k_dn2_tls1_s42_e3_fixed",
     "assign_exact_r400k.npz", 1.0e8, "#bf4d43", "-"),
    ("D/N=5  R/N=8", "v27_50m_exact_r400k_dn5_tls1_s42_e3_fixed", "v27_50m_nogram_r400k_dn5_tls1_s42_e3_fixed",
     "assign_exact_r400k.npz", 2.5e8, "#d97757", "-"),
    ("D/N=20 R/N=8", "v27_50m_exact_r400k_dn20_tls1_s43_e3_fixed", "v27_50m_nogram_r400k_dn20_tls1_s43_e3_fixed",
     "assign_exact_r400k.npz", 1.0e9, "#6a9bcc", "-"),
    ("D/N=20 R/N=1", "v27_50m_exact_r50k_dn20_tls1_s43_e3_fixed", "v27_50m_nogram_r400k_dn20_tls1_s43_e3_fixed",
     "assign_exact_r50k.npz", 1.0e9, "#788c5d", "--"),
]
EDGES = np.array([0, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 1e12])


def counts(n, k):
    ik, ic = IDX[f"keys_{n}"], IDX[f"counts_{n}"]
    i = np.clip(np.searchsorted(ik, k), 0, len(ik) - 1)
    return np.where(ik[i] == k, ic[i], 0).astype(np.float64)


def load(run, split):
    z = np.load(f"{TS}/{run}/token_stats_step{[f for f in os.listdir(f'{TS}/{run}') if f.endswith(f'_{split}.npz')][0].split('step')[1]}")
    return z


def analyse(label, ex, ng, asg, D):
    res = {"label": label, "D": D, "orders": {}, "depth": []}
    A = np.load(f"{TS}/{asg}")
    data = {}
    for split in ("val", "train"):
        fe = [f for f in os.listdir(f"{TS}/{ex}") if f.endswith(f"_{split}.npz")][0]
        fg = [f for f in os.listdir(f"{TS}/{ng}") if f.endswith(f"_{split}.npz")][0]
        e, g = np.load(f"{TS}/{ex}/{fe}"), np.load(f"{TS}/{ng}/{fg}")
        le = e["loss"].astype(np.float64)
        lg = g["loss"].astype(np.float64)[: len(le)]
        data[split] = (le, lg, {n: e[f"key_{n}"] for n in (2, 3, 4, 5)})
    scale = D / POOL1B
    for n in (2, 3, 4, 5):
        rows = []
        cv = counts(n, data["val"][2][n]) * scale
        ct = counts(n, data["train"][2][n]) * scale
        rare_v = counts(n, data["val"][2][n]) == 0
        rare_t = counts(n, data["train"][2][n]) == 0
        # rare bin (count < 30 in the 1B pool): its own row
        bins = [("rare", rare_v, rare_t, None)]
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            bins.append((f"{lo:g}-{hi:g}", (~rare_v) & (cv >= lo) & (cv < hi), (~rare_t) & (ct >= lo) & (ct < hi),
                         float(np.sqrt(max(lo, 1) * min(hi, 3e5)))))
        for name, mv, mt, mid in bins:
            if mv.sum() < 400 or mt.sum() < 150:
                continue
            le, lg = data["val"][0], data["val"][1]
            te, tg = data["train"][0], data["train"][1]
            dva = (lg[mv].mean() - le[mv].mean()) * 1000
            dtr = (tg[mt].mean() - te[mt].mean()) * 1000
            rows.append(dict(bin=name, mid=mid, frac=float(mv.mean()), dtrain=dtr, dval=dva, dgap=dtr - dva,
                             covered=float(np.isin(data["val"][2][n][mv], A[f"keys_{n}"]).mean())))
        res["orders"][n] = rows
    for split in ("val", "train"):
        depth = np.zeros(len(data[split][0]), int)
        for n in (2, 3, 4, 5):
            depth = np.where(np.isin(data[split][2][n], A[f"keys_{n}"]), n, depth)
        data[split] = data[split] + (depth,)
    for d in (0, 2, 3, 4, 5):
        mv, mt = data["val"][3] == d, data["train"][3] == d
        dva = (data["val"][1][mv].mean() - data["val"][0][mv].mean()) * 1000
        dtr = (data["train"][1][mt].mean() - data["train"][0][mt].mean()) * 1000
        res["depth"].append(dict(depth=d, frac=float(mv.mean()), dtrain=dtr, dval=dva, dgap=dtr - dva))
    res["total"] = dict(dval=(data["val"][1].mean() - data["val"][0].mean()) * 1000,
                        dtrain=(data["train"][1].mean() - data["train"][0].mean()) * 1000)
    return res


def main():
    results = [analyse(*c[:5]) for c in CASES]
    fig, axs = plt.subplots(2, 4, figsize=(18, 8.5), sharex="row")
    for j, n in enumerate((2, 3, 4, 5)):
        for (label, *_r, col, ls), res in zip(CASES, results):
            rows = [r for r in res["orders"][n] if r["mid"] is not None]
            if not rows:
                continue
            x = [r["mid"] for r in rows]
            axs[0, j].plot(x, [r["dgap"] for r in rows], ls, color=col, marker="o", ms=3.5, label=label)
            axs[1, j].plot(x, [r["dval"] for r in rows], ls, color=col, marker="o", ms=3.5, label=label)
            rare = [r for r in res["orders"][n] if r["mid"] is None]
            if rare:
                axs[0, j].plot(0.6, rare[0]["dgap"], "s", color=col, mfc="none", ms=6)
                axs[1, j].plot(0.6, rare[0]["dval"], "s", color=col, mfc="none", ms=6)
        for i in (0, 1):
            ax = axs[i, j]
            ax.set_xscale("log"); ax.axhline(0, color="#191919", lw=0.7)
            ax.grid(True, color="#e8e6dc", lw=0.6)
        axs[0, j].set_title(f"order n={n}", loc="left", fontsize=11)
        axs[1, j].set_xlabel("context count in the run's own pool (□ = rare, <30 in 1B)")
    axs[0, 0].set_ylabel("dGap (mK)  ↑ = OE adds forking")
    axs[1, 0].set_ylabel("dVal (mK)  ↑ = OE wins on val")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("v27 50M, E3 end, per-token paired dumps: where the forking cost and the val gain live", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/fig_gapfreq.png", dpi=150)
    fig.savefig(f"{OUT}/fig_gapfreq.svg")
    json.dump(results, open(f"{OUT}/gapfreq.json", "w"), indent=1)
    for res in results:
        print(f"\n== {res['label']}  total dTrain {res['total']['dtrain']:.1f} dVal {res['total']['dval']:.1f}")
        for d in res["depth"]:
            print(f"   depth {d['depth']}: frac {d['frac']:.3f} dTrain {d['dtrain']:7.1f} dVal {d['dval']:7.1f} dGap {d['dgap']:7.1f}")


if __name__ == "__main__":
    main()
