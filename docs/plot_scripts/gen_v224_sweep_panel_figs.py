#!/usr/bin/env python3
"""v2.2/v2.3/v2.4 + v21 sweep figures for the 2026-10-17 progress panel.

Reads harvested summaries (output/harvest_v224, output/harvest_v21) and writes
SVG+PNG pairs into docs/figs/sweeps_20261017/.

Style: report-to-human panel_style (Anthropic palette, Okabe-Ito for >5 series).
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/Users/guoshaoyang/.verdent/skills/report-to-human/scripts")
sys.path.insert(0, os.path.join(REPO, "code"))

import matplotlib.pyplot as plt
from panel_style import OKABE_ITO, C, FIG_WIDE, apply, save

apply()

H224 = os.path.join(REPO, "output", "harvest_v224")
H21 = os.path.join(REPO, "output", "harvest_v21")
OUT = os.path.join(REPO, "docs", "figs", "sweeps_20261017")
os.makedirs(OUT, exist_ok=True)

ORDER_SETS = ["2", "2-3", "2-3-4", "2-3-4-5", "2-3-4-5-6", "2-3-4-5-6-7"]
SET_LABEL = {
    "2": "n=2", "2-3": "n=2,3", "2-3-4": "n=2,3,4", "2-3-4-5": "n=2,3,4,5",
    "2-3-4-5-6": "n=2..6", "2-3-4-5-6-7": "n=2..7",
}
RCOL = {"2": 0, "2-3": 1, "2-3-4": 2, "2-3-4-5": 3, "2-3-4-5-6": 4, "2-3-4-5-6-7": 5}


def load(pattern, rx):
    out = {}
    for d in glob.glob(os.path.join(pattern)):
        sj = os.path.join(d, "summary.json")
        if not os.path.exists(sj):
            continue
        s = json.load(open(sj))
        m = re.search(rx, os.path.basename(d))
        if m:
            out[m.groups()] = {"gap": s["final_gap"], "dir": os.path.basename(d)}
    return out


v22 = load(os.path.join(H224, "v22r_*"), r"v22r_n([\d\-]+)_r(\w+)_s")
v23 = load(os.path.join(H224, "v23d_*"), r"v23d_n([\d\-]+)_p(\w+)_s")
v24 = load(os.path.join(H224, "v24ms_*"), r"v24ms_(l\d+h\d+d\d+)_(\w+)_(p\w+)_s")
v21 = load(os.path.join(H21, "v21_*"), r"v21_(dict|hash|nogram)(?:_n([\d\-]+))?_s")
v21 = {(k[1] or "-", k[0]): v for k, v in v21.items()}

RNUM = {"1": 1, "10": 10, "100": 100, "1k": 1e3, "10k": 1e4, "100k": 1e5,
        "300k": 3e5, "1m": 1e6, "2m": 2e6, "3m": 3e6}
POOLNUM = {"10m": 1e7, "14m": 1.4e7, "19m": 1.9e7, "27m": 2.7e7, "37m": 3.7e7}


def step_gap_curves(path):
    xs, ys = [], []
    if not os.path.exists(path):
        return xs, ys
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_loss" in r and "train_loss" in r and r.get("val_loss"):
            xs.append(r.get("step", len(xs)))
            ys.append(r["val_loss"] - r["train_loss"])
    return xs, ys


# ---- fig 1: v2.2 gap vs R ----
fig, ax = plt.subplots(figsize=FIG_WIDE)
for o in ORDER_SETS:
    pts = sorted((RNUM[r], v["gap"]) for (oo, r), v in v22.items() if oo == o)
    if not pts:
        continue
    ax.plot(*zip(*pts), "o-", color=OKABE_ITO[RCOL[o]], label=SET_LABEL[o])
ax.set_xscale("log")
ax.set_xlabel("每阶 n-gram 表行数 R（log 刻度）")
ax.set_ylabel("最终 gap（val − train）")
ax.axhline(0, color=C["mid"], lw=0.8, ls="--")
ax.legend(ncol=3, loc="upper left")
save(fig, os.path.join(OUT, "fig_v22_gap_vs_R"))

# ---- fig 2: v2.2 final gap at R=1M ----
fig, ax = plt.subplots(figsize=FIG_WIDE)
vals = [v22.get((o, "1m"), {}).get("gap") for o in ORDER_SETS]
bars = ax.bar(range(len(ORDER_SETS)), [v if v is not None else 0 for v in vals],
              color=[OKABE_ITO[RCOL[o]] for o in ORDER_SETS], width=0.62)
for i, v in enumerate(vals):
    if v is None:
        ax.text(i, 0.15, "OOM 重试中", rotation=90, ha="center", fontsize=8, color=C["muted"])
    else:
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=9, color=C["dark"])
ax.set_xticks(range(len(ORDER_SETS)))
ax.set_xticklabels([SET_LABEL[o] for o in ORDER_SETS])
ax.set_ylabel("最终 gap @ R=1M")
ax.set_ylim(0, 10.4)
save(fig, os.path.join(OUT, "fig_v22_final_gap_r1m"))

# ---- fig 3: v2.3 pool envelope ----
fig, ax = plt.subplots(figsize=FIG_WIDE)
for o in ORDER_SETS:
    pts = sorted((POOLNUM[p], v["gap"]) for (oo, p), v in v23.items() if oo == o)
    if not pts:
        continue
    ax.plot(*zip(*pts), "o-", color=OKABE_ITO[RCOL[o]], label=SET_LABEL[o])
ax.set_xscale("log")
ax.set_xticks([1e7, 1.4e7, 1.9e7, 2.7e7, 3.7e7])
ax.set_xticklabels(["10M", "14M", "19M", "27M", "37M"])
ax.set_xlabel("训练数据池大小（tokens，log 刻度）")
ax.set_ylabel("最终 gap")
ax.legend(ncol=3, loc="upper right")
save(fig, os.path.join(OUT, "fig_v23_pool_envelope"))

# ---- fig 4/5: v2.4 envelopes (tri / n25) ----
SIZES = [("l4h4d512", "4L·512d·17M"), ("l6h5d640", "6L·640d·35M"),
         ("l8h6d768", "8L·768d·63M"), ("l10h7d896", "10L·896d·104M"),
         ("l12h8d1024", "12L·1024d·159M")]
for mode, fname, color in (("tri", "fig_v24_model_tri_envelope", C["orange"]),
                           ("n25", "fig_v24_model_n25_envelope", C["blue"])):
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for i, (sz, lab) in enumerate(SIZES):
        pts = sorted((POOLNUM[p[1:]], v["gap"])
                     for (m, md, p), v in v24.items() if m == sz and md == mode)
        if not pts:
            continue
        ax.plot(*zip(*pts), "o-", color=OKABE_ITO[i], label=lab)
    ax.set_xscale("log")
    ax.set_xticks([1e7, 1.4e7, 1.9e7, 2.7e7, 3.7e7])
    ax.set_xticklabels(["10M", "14M", "19M", "27M", "37M"])
    ax.set_xlabel("训练数据池大小（tokens，log 刻度）")
    ax.set_ylabel("最终 gap")
    ax.set_ylim(0, 15)
    ax.legend(ncol=2, fontsize=8)
    save(fig, os.path.join(OUT, fname))

# ---- fig 6: v2.4 ladder at pool=10M ----
fig, ax = plt.subplots(figsize=FIG_WIDE)
PARAMS = {"l4h4d512": 17, "l6h5d640": 35, "l8h6d768": 63, "l10h7d896": 104, "l12h8d1024": 159}
for mode, color, lab in (("tri", C["orange"], "仅 trigram"),
                         ("n25", C["blue"], "n=2..5 联合")):
    pts = []
    for sz in PARAMS:
        g = v24.get((sz, mode, "p10m"), {}).get("gap")
        if g is not None:
            pts.append((PARAMS[sz], g))
    ax.plot(*zip(*pts), "o-", color=color, label=lab)
    for x, y in pts:
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color=C["dark"])
ax.set_xscale("log")
ax.set_xticks(list(PARAMS.values()))
ax.set_xticklabels([f"{v}M" for v in PARAMS.values()])
ax.set_xlabel("backbone 参数量（log 刻度）")
ax.set_ylabel("最终 gap @ 池 10M")
ax.set_ylim(0, 16)
ax.legend(loc="center left")
save(fig, os.path.join(OUT, "fig_v24_ladder_p10m"))

# ---- fig 7: v21 dict vs hash ----
SETS21 = ["2", "3", "4", "5", "2-3", "2-3-4"]
import numpy as np
x = np.arange(len(SETS21))
w = 0.36
fig, ax = plt.subplots(figsize=FIG_WIDE)
dv = [v21.get((o, "dict"), {}).get("gap", np.nan) for o in SETS21]
hv = [v21.get((o, "hash"), {}).get("gap", np.nan) for o in SETS21]
ax.bar(x - w / 2, dv, w, color=C["orange"], label="dict 无碰撞表（top-1M 唯一行）")
ax.bar(x + w / 2, hv, w, color=C["mid"], label="hash 表（1M slot，全碰撞）")
for xi, v in zip(x - w / 2, dv):
    if not np.isnan(v):
        ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, hv):
    if not np.isnan(v):
        ax.text(xi, v + 0.12, f"{v:.2f}", ha="center", fontsize=8)
ng = v21.get(("-", "nogram"), {}).get("gap")
if ng is not None:
    ax.axhline(ng, color=C["green"], lw=1.4, ls="--")
    ax.text(len(SETS21) - 0.5, ng + 0.12, f"无 n-gram {ng:.2f}", fontsize=8, color=C["green"])
ax.set_xticks(x)
ax.set_xticklabels(["n=2", "n=3", "n=4", "n=5", "n=2,3", "n=2,3,4"])
ax.set_ylabel("最终 gap")
ax.set_ylim(0, 9)
ax.legend(loc="upper left")
save(fig, os.path.join(OUT, "fig_v21_dict_vs_hash"))

# ---- fig 8: v21 n5 gap curves ----
fig, ax = plt.subplots(figsize=FIG_WIDE)
for key, color, lab in ((("5", "dict"), C["orange"], "dict 无碰撞 n=5"),
                        (("5", "hash"), C["mid"], "hash n=5"),
                        (("-", "nogram"), C["green"], "无 n-gram")):
    d = v21.get(key)
    if not d:
        continue
    xs, ys = step_gap_curves(os.path.join(H21, d["dir"], "train_log.jsonl"))
    if xs:
        ax.plot(xs, ys, color=color, label=lab)
ax.set_xlabel("训练步")
ax.set_ylabel("gap（val − train）")
ax.set_ylim(0, 8)
ax.legend(loc="upper left")
save(fig, os.path.join(OUT, "fig_v21_n5_gap_curves"))

# ---- fig 9: gap vs epoch (forking onset), extracted from existing curves ----
# v22r R=1M arms: pool = full shard 1 = 49,692,672 tok; 147,456 tok/step ->
# 337.0 steps per epoch.  v23d p10m: pool 9,879,552 tok -> 29.85 epochs.
fig, ax = plt.subplots(figsize=FIG_WIDE)
SPE_49M, SPE_10M = 49692672 / 147456, 9879552 / 147456
for o in ORDER_SETS:
    d = v22.get((o, "1m"))
    if not d:
        continue
    xs, ys = step_gap_curves(os.path.join(H224, d["dir"], "train_log.jsonl"))
    if xs:
        ax.plot([x / SPE_49M for x in xs], ys, color=OKABE_ITO[RCOL[o]],
                label=f"{SET_LABEL[o]} @R=1M")
d = v23.get(("2", "10m"))
if d:
    xs, ys = step_gap_curves(os.path.join(H224, d["dir"], "train_log.jsonl"))
    if xs:
        ax.plot([x / SPE_10M for x in xs], ys, color=C["mid"], lw=1.6, ls="--",
                label="n=2 @池10M（30 epoch）")
ax.axvline(3, color=C["orange"], lw=1.2, ls=":")
ax.set_xlabel("已见 epoch 数（重复遍数）")
ax.set_ylabel("gap（val − train）")
ax.set_xlim(0, 30)
ax.legend(ncol=2, fontsize=8, loc="center right")
save(fig, os.path.join(OUT, "fig_epoch_forking_curves"))

print("figures written to", OUT)
