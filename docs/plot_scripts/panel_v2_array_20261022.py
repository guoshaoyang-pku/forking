#!/usr/bin/env python3
"""Panel figures for the v2 experiment array (2026-10-22).

Data: v27 clean-pool R-scan + D/N envelope (this session's harvest),
multi-head 3-seed matrix, v26 iso-ratio ladder. Numbers from
docs/experiment-log.md §135/§141/§142 and the v27_a100 harvest.
"""
import sys

sys.path.insert(0, "/Users/guoshaoyang/.verdent/skills/report-to-human/scripts")
import matplotlib.pyplot as plt  # noqa: E402
from panel_style import C, SERIES, apply, save  # noqa: E402

OUT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/figs/v2_array_20261022"
apply()

# ---------- fig 1: R-scan decomposition (50M, D/N=20, s43) ----------
rn = [0.12, 0.24, 0.50, 1.00, 2.56, 5.12, 8.00]
dT = [-24.0, -29.1, -26.6, 15.6, 50.0, 72.2, 90.9]
dG = [-1.3, 2.6, 4.4, 10.8, 25.7, 44.6, 67.0]
dV = [-22.7, -31.7, -31.0, 4.8, 24.2, 27.6, 23.8]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.axhline(0, color=C["mid"], lw=0.9)
ax.plot(rn, dV, "o-", color=C["orange"], label="净收益 ΔVal")
ax.plot(rn, dT, "s-", color=C["blue"], label="训练项 Δtrain")
ax.plot(rn, dG, "^--", color=C["mid"], label="代价项 Δgap")
ax.plot([8.0], [29.8], "o", color=C["orange"], mfc="white")
ax.annotate("R/N=2 单臂发散", xy=(2.0, -55), xytext=(2.4, -48),
            fontsize=8.5, color=C["muted"],
            arrowprops=dict(arrowstyle="-", color=C["mid"], lw=0.8))
ax.plot([2.0], [-55], "x", color=C["dark"], ms=6)
ax.set_xscale("log")
ax.set_xticks(rn)
ax.set_xticklabels([str(x) for x in rn])
ax.set_xlabel("R/N（表行数 / 模型参数）")
ax.set_ylabel("mK")
ax.legend(loc="lower right")
save(fig, f"{OUT}/fig_rscan")

# ---------- fig 2: D/N envelope decomposition (50M, r400k, s42) ----------
dn = [10, 14, 20]
dn_dT = [115.2, 108.1, 100.2]
dn_dG = [98.7, 85.9, 70.4]
dn_dV = [16.5, 22.2, 29.8]
dn35 = [28, 56]
dn35_dV = [39.8, 33.5]

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.axhline(0, color=C["mid"], lw=0.9)
ax.plot(dn, dn_dV, "o-", color=C["orange"], label="净收益 ΔVal（50M）")
ax.plot(dn, dn_dT, "s-", color=C["blue"], label="训练项 Δtrain（50M）")
ax.plot(dn, dn_dG, "^--", color=C["mid"], label="代价项 Δgap（50M）")
ax.plot(dn35, dn35_dV, "o--", color=C["green"], label="净收益 ΔVal（35M，表 LR 减半）")
ax.set_xscale("log")
ax.set_xticks([10, 14, 20, 28, 56])
ax.set_xticklabels(["10", "14", "20", "28", "56"])
ax.set_xlabel("D/N（训练 token 数 / 模型参数）")
ax.set_ylabel("mK")
ax.legend(loc="center right")
save(fig, f"{OUT}/fig_dn")

# ---------- fig 3: multi-head 3-seed matrix ----------
arms = ["exact\n(零碰撞)", "multi-head\nH=4", "multi-head\nH=8"]
means = [40.9, 46.6, 47.1]
sds = [8.6, 7.0, 12.5]
seeds = [[33.7, 38.6, 50.5], [49.4, 38.7, 51.8], [54.1, 32.6, 54.5]]

fig, ax = plt.subplots(figsize=(7.2, 3.0))
x = range(3)
ax.bar(x, means, yerr=sds, capsize=4, color=[C["orange"], C["blue"], C["green"]],
       width=0.55, error_kw=dict(ecolor=C["muted"], lw=1.2))
for i, ss in enumerate(seeds):
    ax.plot([i - 0.18, i, i + 0.18], ss, "o", color=C["dark"], ms=3.5, alpha=0.65)
ax.set_xticks(list(x))
ax.set_xticklabels(arms)
ax.set_ylabel("ΔVal（mK）")
ax.set_ylim(0, 68)
save(fig, f"{OUT}/fig_mh")

# ---------- fig 4: iso-ratio ladder (D/N=20, R/N=8) ----------
N = [35, 50, 104]
iso_dV = [48.6, 37.7, 25.7]

fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(N, iso_dV, "o-", color=C["orange"], label="净收益 ΔVal")
for n, v in zip(N, iso_dV):
    ax.annotate(f"+{v}", xy=(n, v), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=C["dark"])
ax.set_xscale("log")
ax.set_xticks(N)
ax.set_xticklabels(["35M", "50M", "104M"])
ax.set_xlabel("模型参数 N（D/N=20、R/N=8 同步放大）")
ax.set_ylabel("ΔVal（mK）")
ax.set_ylim(0, 60)
save(fig, f"{OUT}/fig_iso")

print("figures written to", OUT)
