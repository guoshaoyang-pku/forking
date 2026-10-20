# -*- coding: utf-8 -*-
# fig_r_scaling_adamc: adam+clip 时代的 R-scaling（W1 H2 六点 + exact 对照 + 夜间 exact R/N 梯度）
# 数据源：experiment-log §115-§117 + ophis/360-2 runs_fixed（数值硬编码自收割脚本，注释给 run_id）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, FixedLocator

plt.rcParams["font.family"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/figs/v26_r_scaling_20261020"
import os
os.makedirs(OUT, exist_ok=True)

# ---- adamc 时代 H2 曲线（50M, 1B×E3, s42, tls1, adam+clip）run: v26w1_h2_m{64,128,256,512}kac / m974kac(360-2)
H2 = {64:  (-0.0268, -0.0176, -0.0102),
      128: (-0.0358, -0.0026, -0.0213),
      256: (-0.0462, -0.0386, -0.0320),
      512: (-0.0383, -0.0360, -0.0306),
      974: (-0.0468, -0.0414, -0.0303)}
# exact 零碰撞对照（974k 总物理行 ≈ 每阶 243.6k；e3exact 部署配方 3 seed mean）
EXACT974 = [(-0.074, -0.052, -0.0348)]  # s42 e3exact@1x §115
# RMSProp 时代 H2@4x（对照，灰虚线，不混画同轴结论只作背景）run: v26w1_h2_m*_tls4
H2_OLD = {128: -0.039, 256: -0.045, 512: -0.049, 974: -0.067}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
xs = sorted(H2)
labels = [f"{m}k" for m in xs]

for ax, idx, name, ylim in [(axes[0], 0, "第 1 遍收益（E1 Δval，负=收益）", (-0.085, 0.0)),
                            (axes[1], 2, "第 3 遍收益（E3 Δval，负=收益）", (-0.045, 0.005))]:
    ys = [H2[m][idx] for m in xs]
    ax.plot(xs, ys, marker="o", color="#1f77b4", lw=1.9, ms=6.5, label="H2 @1×（adam+clip）")
    if idx == 0:
        ax.plot(sorted(H2_OLD), [H2_OLD[m] for m in sorted(H2_OLD)], marker="s", ms=5, ls=":",
                color="#999999", lw=1.3, label="H2 @4× RMSProp 时代（对照，勿混读）")
    ax.scatter([974], [EXACT974[0][idx]], marker="X", s=130, color="#d62728", zorder=6,
               label="exact974k 零碰撞 @1×（e3exact 配方）")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(55, 1300)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(labels)
    ax.set_xlabel("R = 每阶表行数（H2 物理 bucket / exact = distinct+1）")
    ax.set_ylabel("Δval vs nogram")
    ax.set_title(name, fontsize=11)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left" if idx == 0 else "upper left")

axes[0].annotate("E1 升至 256k 后持平/微降", (256, -0.0462), textcoords="offset points",
                 xytext=(6, -16), fontsize=8.5, color="#1f77b4")
axes[1].annotate("E3 在 512k 饱和（974k 不加分）", (512, -0.0306), textcoords="offset points",
                 xytext=(-150, -24), fontsize=8.5, color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9))
axes[1].annotate("exact 红利 +0.004", (974, -0.0348), textcoords="offset points",
                 xytext=(-120, 22), fontsize=8.5, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9))

fig.suptitle("W1 R-scaling（adam+clip 时代）｜50M × 1B unique × E3, n=2–5 联合, f≥256/H2, s42", fontsize=11.5)
fig.text(0.5, 0.005, "源：v26w1_h2_m64k~m974kac（§115/§117 收割）、v26fix_exact974k_adamc（e3exact）；nogram 基线 E1/E2/E3 = 2.8527/2.7397/2.6930", ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(f"{OUT}/fig_r_scaling_adamc.svg")
fig.savefig(f"{OUT}/fig_r_scaling_adamc.png", dpi=160)
print(f"{OUT}/fig_r_scaling_adamc.png")
