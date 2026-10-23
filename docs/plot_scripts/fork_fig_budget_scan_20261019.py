# -*- coding: utf-8 -*-
# fig_budget_scan: E1 单轴 + E3 断轴（负半轴 ≥50% 图高，正半轴压缩发散区）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, FixedLocator

plt.rcParams["font.family"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/figs/v26_w1_20261019"

def agg(pts, idx):
    vals = [p[idx] for p in pts if p is not None and (idx == 0 or p[1] < 0.05)]
    return np.mean(vals) if vals else None
def ndiverge(pts):
    return sum(1 for p in pts if p is not None and p[1] >= 0.05)

H2_4x = {128:[(-0.039,-0.022),(-0.039,-0.023),(-0.010,-0.008)],
         256:[(-0.039,0.078),(-0.054,-0.012),(-0.041,-0.027)],
         512:[(-0.051,-0.030),(-0.060,-0.039),(-0.037,-0.016)],
         974:[(-0.083,-0.049),(-0.057,-0.021),(-0.061,-0.021)]}
H2_1x = {128:[(-0.057,-0.033),(-0.056,-0.034),(-0.036,0.081)],
         512:[(-0.055,-0.040),(-0.054,-0.050),(-0.032,-0.023)]}
EXACT = {974:[(-0.089,0.447),(-0.073,0.070),(-0.068,2.660)]}
C3_128=[(-0.042,-0.002)]; C3_512=[(-0.072,-0.033),(-0.054,-0.042),(-0.041,-0.023)]
E1V=[(-0.038,-0.038),(-0.038,-0.030)]

fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.08], height_ratios=[0.34, 0.66],
                      hspace=0.045, wspace=0.22)
ax_e1 = fig.add_subplot(gs[:, 0])          # E1 全高单轴
ax_hi = fig.add_subplot(gs[0, 1])          # E3 正半轴（压缩）
ax_lo = fig.add_subplot(gs[1, 1], sharex=ax_hi)  # E3 负半轴（放大，占 66%）

def draw_e1(ax):
    xs4 = sorted(H2_4x); ys4 = [agg(H2_4x[m],0) for m in xs4]
    ax.plot(xs4, ys4, marker="o", color="#1f77b4", lw=1.8, ms=6, label="H2 @4x（W1 胜者）")
    xs1 = sorted(H2_1x); ys1 = [agg(H2_1x[m],0) for m in xs1]
    ax.plot(xs1, ys1, marker="s", color="#2ca02c", lw=1.6, ms=6, ls="--", label="H2 @1x")
    ax.scatter([128],[agg(C3_128,0)], marker="D", s=70, color="#9467bd", zorder=5, label="C3（语义 KL）@128k")
    ax.scatter([512],[agg(C3_512,0)], marker="D", s=70, color="#7e57c2", zorder=5,
               edgecolors="black", label="C3 @512k（3 seed 反超 H2）")
    ax.scatter([128],[agg(E1V,0)], marker="*", s=170, color="#ff7f0e", zorder=6,
               edgecolors="#8c4500", label="E1V（E1 向量聚类）@128k")
    ax.scatter([974],[agg(EXACT[974],0)], marker="X", s=110, color="#d62728", zorder=6,
               label="exact974k（零碰撞）")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylim(-0.09, 0.01)
    ax.set_ylabel("Δval vs nogram（负=收益）")
    ax.set_title("第 1 遍收益（E1 Δval）", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="lower left")

def draw_e3(ax, upper):
    xs4 = sorted(H2_4x); ys4 = [agg(H2_4x[m],1) for m in xs4]
    ax.plot(xs4, ys4, marker="o", color="#1f77b4", lw=1.8, ms=6,
            label="H2 @4x（W1 胜者）" if not upper else None)
    xs1 = sorted(H2_1x); ys1 = [agg(H2_1x[m],1) for m in xs1]
    ax.plot(xs1, ys1, marker="s", color="#2ca02c", lw=1.6, ms=6, ls="--",
            label="H2 @1x" if not upper else None)
    ax.scatter([128],[agg(C3_128,1)], marker="D", s=70, color="#9467bd", zorder=5)
    ax.scatter([512],[agg(C3_512,1)], marker="D", s=70, color="#7e57c2", zorder=5, edgecolors="black",
               label="C3 @512k（3 seed 反超 H2）" if not upper else None)
    ax.scatter([128],[agg(E1V,1)], marker="*", s=170, color="#ff7f0e", zorder=6, edgecolors="#8c4500",
               label="E1V（E1 向量聚类）@128k" if not upper else None)
    if upper:
        for v, tag in [(0.070,"s43 +0.070"), (0.447,"s42 +0.447"), (2.660,"s44 +2.660")]:
            ax.scatter([974],[v], marker="x", s=55, color="#d62728", zorder=6)
            ax.annotate(tag, (974, v), textcoords="offset points", xytext=(8,-3),
                        fontsize=7.5, color="#d62728")
        ax.scatter([974],[np.mean([0.447,0.070,2.660])], marker="X", s=120, color="#d62728",
                   zorder=6, label="exact974k：3/3 seed E3 发散")
        ax.set_ylim(0.03, 3.4); ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticks([0.05, 0.1, 0.5, 2.66]); ax.set_yticklabels(["+0.05","+0.10","+0.50","+2.66"])
    else:
        # 发散 seed 的 H2@128k s42 E1 是负的，E3 面板画发散标注
        for m, ys in [(128, H2_4x[128]), (256, H2_4x[256])]:
            nd = ndiverge(ys)
            if nd: ax.annotate(f"{nd}/3 seed 发散", (m, agg(ys,1)),
                               textcoords="offset points", xytext=(8,6), fontsize=8, color="#d62728")
        nd = ndiverge(H2_1x[128])
        if nd: ax.annotate(f"{nd}/3 seed 发散", (128, agg(H2_1x[128],1)),
                           textcoords="offset points", xytext=(10,10), fontsize=8, color="#d62728")
        ax.set_ylim(-0.062, 0.008)
        ax.set_ylabel("Δval vs nogram（负=收益）")
        ax.legend(fontsize=7.5, loc="lower left")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.grid(alpha=0.3)
    for m in [128,256,512,974]: ax.set_xlim(105, 1100)

draw_e1(ax_e1)
draw_e3(ax_hi, True)
draw_e3(ax_lo, False)

# x 轴样式
for ax in (ax_e1, ax_hi, ax_lo):
    ax.set_xscale("log")
    ax.set_xlim(105, 1100)
    ax.grid(alpha=0.3)
ax_e1.xaxis.set_major_locator(FixedLocator([128,256,512,974]))
ax_e1.xaxis.set_minor_locator(NullLocator())
ax_e1.set_xticklabels(["128k","256k","512k","974k"])
ax_e1.set_xlabel("物理行预算 M")
ax_lo.xaxis.set_major_locator(FixedLocator([128,256,512,974]))
ax_lo.xaxis.set_minor_locator(NullLocator())
ax_lo.set_xticklabels(["128k","256k","512k","974k"])
ax_lo.set_xlabel("物理行预算 M")
plt.setp(ax_hi.get_xticklabels(), visible=False)
ax_hi.set_title("第 3 遍收益（E3 Δval）｜上：发散区（log 压缩） 下：收益区（线性放大）", fontsize=10.5)

# 断轴斜线
d = 0.012
kw = dict(transform=ax_hi.transAxes, color="k", clip_on=False, lw=1)
ax_hi.plot((-d, +d), (-d*2.2, +d*2.2), **kw)
ax_hi.plot((1-d, 1+d), (-d*2.2, +d*2.2), **kw)
kw.update(transform=ax_lo.transAxes)
ax_lo.plot((-d, +d), (1-d*1.1, 1+d*1.1), **kw)
ax_lo.plot((1-d, 1+d), (1-d*1.1, 1+d*1.1), **kw)

fig.suptitle("表预算扫描：E1 收益随预算单调升，E3 在 512k 饱和；exact974k 零碰撞在 4x 下 3/3 seed 发散", fontsize=11)
fig.savefig(f"{OUT}/fig_budget_scan.svg", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_budget_scan.png", dpi=160, bbox_inches="tight")
print("fig1 broken-axis done")
