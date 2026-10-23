# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/figs/v26_w1_20261019"

fig, ax = plt.subplots(figsize=(14, 9.2))
ax.axis("off")
W = 14.0

def box(cx, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(plt.Rectangle((cx - w/2, y), w, h, facecolor=fc, edgecolor="#333333", lw=1.0, zorder=3))
    ax.text(cx, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.4)

def link(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#666666", lw=1.2, zorder=2)

C_ROOT="#dbeafe"; C_LINE="#e0e7ff"; C_DONE="#dcfce7"; C_NEW="#fef3c7"; C_READY="#fee2e2"; C_NOTE="#f5f5f4"

# root (y from top)
box(7.0, 8.35, 9.2, 0.62, "v2 核心问题：OE 第 1 遍收益，如何在第 2/3 遍不被 forking 吃掉\n50M backbone × 1B ClimbMix × 3 遍重放 · n=2–5 联合 · f≥256 选择集", C_ROOT, 10.5, True)

cols = [
 (1.75, "线 1 · 表 LR 剂量（v25 帧）", C_DONE,
  "f256 / k256k / d1m / hash1m\n× 0.5x–128x\n\n甜点 0.5–2x\nk256k@1x 最佳 −0.038\nhash1m E3 +0.30\n→ 碰撞=forking 铁证\n≥8x 发散；128x 压制"),
 (4.55, "线 2 · W1 路由搜索（128k）", C_DONE,
  "随机 H1 / H2\n结构 S1\n语义 C1 / C2 / C3\n教师向量 E1V（实验 A）\n\n判词@128k：H2 唯一\n三 seed 稳定（E3≈−0.02）\nE1V 两 seed E3 −0.034"),
 (7.35, "线 3 · 预算扫描→974k", C_NEW,
  "H2：E1 −0.03→−0.08 单调\nE3 在 512k 饱和 ≈−0.03\nexact 零碰撞 3/3 seed 发散\n→ 碰撞对 = 免费正则\nC3@512k 3 seed 反超 H2\n→ 语义路由预算依赖"),
 (10.15, "线 4 · 机制与稳定性", C_DONE,
  "E1V 簇内教师向量\ncos 0.624 vs 随机 0.365\n→ 梯度相容共享坐实\nEOS 发散为随机 onset\n低 LR 部分抑制\n帧平移审计 → 同机同\nseed 参照口径"),
 (12.95, "线 5 · 实验 B（就绪）", C_READY,
  "e1vo：E1V 质心冻结为表\n+ OGD：wte 梯度正交化\n出表子空间\n代码已 commit 并冒烟\n（--table_init_npz /\n--ogd_wte_topk）\n待主进程发射"),
]
for cx, title, fc, body in cols:
    box(cx, 7.35, 2.5, 0.5, title, C_LINE, 9.3, True)
    box(cx, 4.05, 2.5, 3.2, body, fc, 8.3)
    link(cx, 7.35, 7.0, 8.35)

# legend for colors
box(1.75, 3.35, 3.1, 0.5, "绿=已定案  黄=新发现  红=就绪未跑", C_NOTE, 8.3)

# GPT v3 mapping note
box(7.0, 0.55, 13.2, 2.2,
    "对照 GPT《oe_overnight_execution_plan_v3》落位：\n"
    "首轮 8 点 A00–A07 = 线 2（已完成；判词与 v3 §15「H2 胜过所有 learned → 优化随机路线」一致）\n"
    "Q1 LR 剂量 = 线 1（v3 列 2x/8x，实测 0.5x–128x 全谱）；Q2 容量邻居 = 线 3（v3 写 65k/262k，实测 128k–974k）\n"
    "E1V / 实验 B 来自用户 v2 计划的实验 A/B，超出 v3 首轮范围（v3 §6「不把 centroid 直接复制成 embedding」只约束首轮 8 点）\n"
    "v3 要求但未做 → 下一队列：PERM 语义打乱对照（C3 反超后的必须证伪项）· MIX 半宽 concat · SHRINK 置信度门 · confirmation 封存确认集 · 负载诊断",
    C_NOTE, 8.8)

ax.set_xlim(0, W); ax.set_ylim(0, 9.2)
fig.suptitle("v2 实验全景：五条线的关系与 GPT v3 计划落位（2026-10-19 fork 汇总）", fontsize=13, fontweight="bold", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/fig_experiment_tree.svg"); fig.savefig(f"{OUT}/fig_experiment_tree.png", dpi=150)
print("fig4 fixed")
