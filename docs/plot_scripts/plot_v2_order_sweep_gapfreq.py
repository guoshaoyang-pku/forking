"""v2 order sweep: diagonal gap-vs-frequency power laws and collision load.

Inputs (docs/appendices/v2_order_sweep_20261017/):
  gapfreq_step2000.csv  bucket-level gap per (arm, seed, branch), step 2000
  gapfit_step2000.csv   per-(arm, branch) power-law fits (analyze + fit scripts)

Output: docs/figs/v2_order_sweep_20261017/fig_gap_vs_F_and_alpha.{png,svg}

Panel (a): diagonal gap(F) = val - online train loss for the order-n branch in
the order-n arm, mean +- SD over seeds 42/43/44, buckets with >=300 tokens on
both sides.  Panel (b): fitted exponent alpha_n against table load
distinct_n / R (R = 2^20), the collision confound noted in the appendix.
"""
import csv
import os
import collections
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
APP = os.path.join(ROOT, "docs", "appendices", "v2_order_sweep_20261017")
FIG = os.path.join(ROOT, "docs", "figs", "v2_order_sweep_20261017")
os.makedirs(FIG, exist_ok=True)

MIN_TOK = 300
R = 1048576
# distinct train contexts per order, shard 1 with chunk semantics (index build log)
DISTINCT = {2: 3538293, 3: 18989467, 4: 34822438, 5: 42952218,
            6: 46350667, 7: 47745239, 8: 48392919}
DIAG = {2: ("n2", "bigram"), 3: ("n3", "trigram"), 4: ("n4", "4gram"), 5: ("n5", "5gram"),
        6: ("n6", "6gram"), 7: ("n7", "7gram"), 8: ("n8", "8gram")}

rows = list(csv.DictReader(open(os.path.join(APP, "gapfreq_step2000.csv"))))
fits = {(r["arm"], r["branch"]): r for r in csv.DictReader(open(os.path.join(APP, "gapfit_step2000.csv")))}

cmap = plt.get_cmap("viridis")
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

for n, (arm, br) in DIAG.items():
    by = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == arm and r["branch"] == br and min(int(r["n_train"]), int(r["n_val"])) >= MIN_TOK:
            by[(int(r["F_lo"]), int(r["F_hi"]))].append(float(r["gap"]))
    pts = sorted((np.sqrt(lo * hi), st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, lo, hi)
                 for (lo, hi), v in by.items() if len(v) == 3)
    pos = [p for p in pts if p[1] > 0]
    if not pos:
        continue
    F, g, sd = map(np.array, zip(*[(p[0], p[1], p[2]) for p in pos]))
    ax_a.errorbar(F, g, yerr=sd, fmt="o-", ms=3.5, lw=1, capsize=2, color=cmap((n - 2) / 6),
                  label=f"n={n}")
    a = fits[(arm, br)]
    alpha = float(a["alpha_mean"])
    ax_a.plot(F, np.exp(np.log(g[0]) - alpha * (np.log(F) - np.log(F[0]))), lw=0.6, ls=":",
              color=cmap((n - 2) / 6))

ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlabel("training context count F (bucket geometric centre)")
ax_a.set_ylabel("gap at order n (nats)")
ax_a.set_title("(a) diagonal gap vs F at step 2000 (dotted: fitted slope)")
ax_a.legend(fontsize=7, ncol=2)
ax_a.grid(alpha=0.25, which="both")

ns, loads, al, sds = [], [], [], []
for n, (arm, br) in DIAG.items():
    a = fits[(arm, br)]
    ns.append(n)
    loads.append(DISTINCT[n] / R)
    al.append(float(a["alpha_mean"]))
    sds.append(float(a["alpha_sd"]))
    ax_b.annotate(f"n={n}", (loads[-1], al[-1]), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax_b.errorbar(loads, al, yerr=sds, fmt="s-", ms=5, capsize=3, color="C3")
ax_b.axhline(0, color="0.5", lw=0.8)
ax_b.set_xscale("log")
ax_b.set_xlabel("table load = distinct train contexts / R (R = 2^20)")
ax_b.set_ylabel("power-law exponent alpha (gap ~ F^-alpha)")
ax_b.set_title("(b) decay exponent falls with collision load")
ax_b.grid(alpha=0.25, which="both")

for ext in ("png", "svg"):
    fig.savefig(os.path.join(FIG, f"fig_gap_vs_F_and_alpha.{ext}"), dpi=160)
print("wrote", os.path.join(FIG, "fig_gap_vs_F_and_alpha.png"))
