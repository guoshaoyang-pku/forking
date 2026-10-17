"""E=3 epoch-anchored envelope figures (§78) for the 2026-10-17 progress panel.

Reads output/harvest_e3 (flat <run_id>.json summaries) and the fixed-step
harvest output/harvest_v224 (<run_id>/summary.json) and writes:
  fig_e3_pool_envelope  : v2.3 order-sets, gap vs pool, E=3 solid vs 2000-step dashed
  fig_e3_modelscale     : v2.4 ladder (tri | n2..5), same overlay
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/Users/guoshaoyang/.verdent/skills/report-to-human/scripts")

import matplotlib.pyplot as plt
from panel_style import FIG_WIDE, OKABE_ITO, apply, save

apply()

H_E3 = os.path.join(REPO, "output", "harvest_e3")
H_224 = os.path.join(REPO, "output", "harvest_v224")
OUT = os.path.join(REPO, "docs", "figs", "sweeps_20261017")

ORDER_SETS = ["2", "2-3", "2-3-4", "2-3-4-5", "2-3-4-5-6", "2-3-4-5-6-7"]
SET_LABEL = {
    "2": "n=2", "2-3": "n=2,3", "2-3-4": "n=2,3,4", "2-3-4-5": "n=2,3,4,5",
    "2-3-4-5-6": "n=2..6", "2-3-4-5-6-7": "n=2..7",
}
LADDER = ["l4h4d512", "l6h5d640", "l8h6d768", "l10h7d896", "l12h8d1024"]
LAD_LABEL = {
    "l4h4d512": "17M", "l6h5d640": "35M", "l8h6d768": "63M",
    "l10h7d896": "104M", "l12h8d1024": "159M",
}


def load_e3():
    out = []
    for fp in glob.glob(os.path.join(H_E3, "*.json")):
        s = json.load(open(fp))
        rid = s["run_id"]
        if rid.startswith("v23d"):
            out.append(("v23", "-".join(map(str, s["ngram_orders"])), "-",
                        s["train_max_tokens"], s["final_gap"]))
        else:
            m = re.match(r"v24ms_(l\d+h\d+d\d+)_(tri|n25)_", rid)
            out.append(("v24", m.group(1), m.group(2),
                        s["train_max_tokens"], s["final_gap"]))
    return out


def load_fixed():
    out = []
    for d in glob.glob(os.path.join(H_224, "v23d_*")) + glob.glob(os.path.join(H_224, "v24ms_*")):
        sj = os.path.join(d, "summary.json")
        if not os.path.exists(sj):
            continue
        s = json.load(open(sj))
        rid = s["run_id"]
        if rid.startswith("v23d"):
            out.append(("v23", "-".join(map(str, s["ngram_orders"])), "-",
                        s["train_max_tokens"], s["final_gap"]))
        else:
            m = re.match(r"v24ms_(l\d+h\d+d\d+)_(tri|n25)_", rid)
            out.append(("v24", m.group(1), m.group(2),
                        s["train_max_tokens"], s["final_gap"]))
    return out


def curve(rows, kind, key, mode):
    pts = sorted((p, g) for k, a, b, p, g in rows if k == kind and a == key and b == mode)
    if not pts:
        return [], []
    return [p for p, _ in pts], [g for _, g in pts]


E3 = load_e3()
FIX = load_fixed()

# ---- fig 1: order-set envelope, E=3 vs fixed-step ----
fig, ax = plt.subplots(figsize=FIG_WIDE)
for i, o in enumerate(ORDER_SETS):
    c = OKABE_ITO[i % len(OKABE_ITO)]
    xs, ys = curve(E3, "v23", o, "-")
    ax.plot([x / 1e6 for x in xs], ys, "-o", color=c, ms=3.5, lw=1.6,
            label=SET_LABEL[o] + " (E=3)")
    xs, ys = curve(FIX, "v23", o, "-")
    ax.plot([x / 1e6 for x in xs], ys, "--", color=c, lw=1.1, alpha=0.55)
ax.annotate("2000-step (dashed): 295M tok,\n10M pool = 30 epochs", xy=(0.03, 0.70),
            xycoords="axes fraction", fontsize=8, color="0.35")
ax.annotate("10M pool: 10.53 → 1.67\nwhen epochs are matched", xy=(0.03, 0.16),
            xycoords="axes fraction", fontsize=8, color="0.35")
ax.set_xscale("log")
ax.set_xlabel("training pool (M tokens, log)")
ax.set_ylabel("final gap (nats)")
ax.set_title("E=3 envelope: gap vs pool, per order set (solid E=3, dashed 2000-step)")
ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
save(fig, os.path.join(OUT, "fig_e3_pool_envelope"))

# ---- fig 2: model ladder, E=3 vs fixed-step ----
fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE, sharey=True)
for ax, mode, ttl in zip(axes, ["tri", "n25"], ["trigram only", "n=2..5 joint"]):
    for i, lad in enumerate(LADDER):
        c = OKABE_ITO[i % len(OKABE_ITO)]
        xs, ys = curve(E3, "v24", lad, mode)
        ax.plot([x / 1e6 for x in xs], ys, "-o", color=c, ms=3.5, lw=1.6,
                label=LAD_LABEL[lad] + " (E=3)")
        xs, ys = curve(FIX, "v24", lad, mode)
        ax.plot([x / 1e6 for x in xs], ys, "--", color=c, lw=1.1, alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("training pool (M tokens, log)")
    ax.set_title(ttl, fontsize=9)
axes[0].set_ylabel("final gap (nats)")
axes[0].legend(fontsize=7, frameon=False, loc="upper right")
fig.suptitle("E=3 model ladder: backbone 17M–159M (solid E=3, dashed 2000-step)", y=1.02)
save(fig, os.path.join(OUT, "fig_e3_modelscale"))

print("figures written to", OUT)
