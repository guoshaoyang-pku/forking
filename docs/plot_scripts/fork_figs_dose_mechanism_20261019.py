# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/figs/v26_w1_20261019"

def agg(pts):
    """pts: list of (e1,e3) or None; return means excluding EOS (>+0.15)"""
    e1 = [p[0] for p in pts if p is not None]
    e3 = [p[1] for p in pts if p is not None and p[1] < 0.15]
    e3all = [p[1] for p in pts if p is not None]
    return (np.mean(e1) if e1 else None, np.mean(e3) if e3 else None,
            len(e3), len(e3all))  # clean count, total count

# ---------- data (all dVal vs same-host same-seed nogram) ----------
# M-scan H2@4x: budget -> [(s42),(s43),(s44)]
H2_4x = {
    128:  [(-0.039,-0.022),(-0.039,-0.023),(-0.010,-0.008)],
    256:  [(-0.039, 0.078),(-0.054,-0.012),(-0.041,-0.027)],
    512:  [(-0.051,-0.030),(-0.060,-0.039),(-0.037,-0.016)],
    974:  [(-0.083,-0.049),(-0.057,-0.021),(-0.061,-0.021)],
}
H2_1x = {
    128:  [(-0.057,-0.033),(-0.056,-0.034),(-0.036, 0.081)],
    512:  [(-0.055,-0.040),(-0.054,-0.050),(-0.032,-0.023)],
}
EXACT_4x = {974: [(-0.089, 0.447),(-0.073, 0.070),(-0.068, 2.660)]}
C3_128 = [(-0.042,-0.002)]
C3_512 = [(-0.072,-0.033),(-0.054,-0.042),(-0.041,-0.023)]
E1V_128 = [(-0.038,-0.038),(-0.038,-0.030)]

# ================= fig1: benefit vs budget =================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
for ax, idx, ttl in [(axes[0], 0, "第 1 遍收益（E1 Δval）"), (axes[1], 1, "第 3 遍收益（E3 Δval）")]:
    for series, ms, color, label in [
        (H2_4x, "o", "#1f77b4", "H2 @4x（W1 胜者）"),
        (H2_1x, "s", "#2ca02c", "H2 @1x"),
    ]:
        xs = sorted(series)
        ys = [agg(series[m])[idx] for m in xs]
        ns = [agg(series[m])[2] for m in xs]
        ax.plot(xs, ys, marker=ms, color=color, label=label, lw=1.8, ms=6)
        for m, y, n, tot in zip(xs, ys, ns, [agg(series[m])[3] for m in xs]):
            if n < tot:
                ax.annotate("有发散", (m, y), textcoords="offset points", xytext=(6, 8),
                            fontsize=8, color="#d62728")
    # C3
    ax.scatter([128], [agg(C3_128)[idx]], marker="D", s=70, color="#9467bd", zorder=5,
               label="C3（语义 KL 路由）@128k")
    ax.scatter([512], [agg(C3_512)[idx]], marker="D", s=70, color="#9467bd", zorder=5,
               label="C3 @512k（3 seed 反超）" if idx == 1 else None)
    # E1V
    ax.scatter([128], [agg(E1V_128)[idx]], marker="*", s=160, color="#ff7f0e", zorder=6,
               label="E1V（E1 向量聚类）" if idx == 0 else None)
    # exact974k
    e3e = agg(EXACT_4x[974])
    if idx == 0:
        ax.scatter([974], [e3e[0]], marker="X", s=110, color="#d62728", zorder=6,
                   label="exact974k（零碰撞）")
    else:
        ax.scatter([974], [0.447], marker="X", s=110, color="#d62728", zorder=6,
                   label="exact974k：3/3 seed 发散")
        for v in (0.447, 0.070, 2.660):
            ax.scatter([974], [min(v, 1.0)], marker="x", s=45, color="#d62728")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xticks([128, 256, 512, 974]); ax.set_xticklabels(["128k","256k","512k","974k"])
    ax.set_xlabel("物理行预算 M"); ax.set_ylabel("Δval vs nogram（负=收益）")
    ax.set_title(ttl); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left" if idx == 0 else "upper left")
axes[1].set_ylim(-0.06, 1.0)
fig.suptitle("表预算扫描：E1 收益随预算单调升，E3 在 512k 饱和；exact 零碰撞在 4x 下 3/3 seed 发散", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_budget_scan.svg"); fig.savefig(f"{OUT}/fig_budget_scan.png", dpi=160)
print("fig1 done")

# ================= fig2: v25 LR dose response (E3 dVal) =================
dose = {
 "f256（f≥256 筛选）": {0.5:-0.037, 1:-0.028, 2:-0.031, 4:-0.020, 8:1.944, 16:-0.004, 128:0.066},
 "k256k（top-k 256k/阶）": {1:-0.038, 2:-0.043, 4:-0.002, 8:-0.020, 16:1.050},
 "d1m（全频 dict）": {0.5:-0.035, 1:-0.011, 2:0.022, 4:0.070, 8:0.044, 16:0.055},
 "hash1m（不筛碰撞）": {1:0.181, 4:0.300, 16:0.501},
}
fig, ax = plt.subplots(figsize=(7.6, 4.4))
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
for (label, d), c in zip(dose.items(), colors):
    xs = sorted(d); ys = [d[x] for x in xs]
    ysp = [min(y, 0.62) for y in ys]
    ax.plot(xs, ysp, marker="o", lw=1.8, ms=6, color=c, label=label)
    for x, y in zip(xs, ys):
        if y > 0.62:
            ax.annotate(f"+{y:.2f} 发散", (x, 0.62), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=c, arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_xticks([0.5,1,2,4,8,16,128]); ax.set_xticklabels(["0.5x","1x","2x","4x","8x","16x","128x"])
ax.set_xlabel("表学习率倍率（RMSProp 无动量）"); ax.set_ylabel("E3 Δval vs nogram（负=收益）")
ax.set_title("v25 帧 LR 剂量响应（s42，1B×3 遍）：甜点在 0.5–2x；≥8x 发散；128x 压制收益")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_lr_dose.svg"); fig.savefig(f"{OUT}/fig_lr_dose.png", dpi=160)
print("fig2 done")

# ================= fig3: E1V mechanism =================
m = {"e1v": (0.6235, 0.4635, 0.7878), "h2": (0.3651, 0.2685, 0.4655), "random": (0.3666, 0.3057, 0.4248)}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
ax = axes[0]
names = ["E1V 聚类", "H2 随机 hash", "随机划分（零假设）"]
means = [m["e1v"][0], m["h2"][0], m["random"][0]]
p10 = [m["e1v"][1], m["h2"][1], m["random"][1]]
p90 = [m["e1v"][2], m["h2"][2], m["random"][2]]
bars = ax.bar(names, means, color=["#ff7f0e", "#1f77b4", "#999999"], width=0.55)
ax.errorbar(range(3), means, yerr=[np.array(means)-np.array(p10), np.array(p90)-np.array(means)],
            fmt="none", ecolor="black", capsize=5, lw=1.2)
for i, v in enumerate(means):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("簇内教师向量余弦相似度（均值，误差棒=p10~p90）")
ax.set_title("E1V 把「学出来方向一致」的上下文聚到一起\n（H2/随机 ≈0.37 是表向量的自然正偏基线）")
ax.set_ylim(0, 0.85); ax.grid(alpha=0.3, axis="y")
ax = axes[1]
cats = ["p50", "p99", "max"]
e1v_sizes = [4, 46, 496]; h2_sizes = [7, 14, 21]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, e1v_sizes, w, label="E1V", color="#ff7f0e")
ax.bar(x + w/2, h2_sizes, w, label="H2", color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("簇大小（n=2 阶）")
ax.set_title("E1V 簇大小高度不均（最大 496）\nH2 近似均匀——两种共享结构本质不同")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.suptitle("实验 A 机制验证：n=2 阶 368,957 个上下文在 E1 教师表中的共享结构", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_e1v_mechanism.svg"); fig.savefig(f"{OUT}/fig_e1v_mechanism.png", dpi=160)
print("fig3 done")
