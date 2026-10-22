#!/usr/bin/env python3
"""Forking v2 实验阵列思维导图（2026-10-22，加入蒸馏线与 A100 排期）：一个目标 → 三条问题线（+已结题机制线）→ 实验叶子。
叶子状态：done / running / todo / user（用户自做）。数字全部来自 docs/experiment-log.md 对应 §。
输出 docs/figs/v2_experiment_array_20261021/{mindmap.svg,mindmap.html}；PNG 由 Chrome headless 截图。"""
import html, os

OUT = os.path.join(os.path.dirname(__file__), "..", "figs", "v2_experiment_array_20261022")
os.makedirs(OUT, exist_ok=True)

ST = {  # fill, stroke, tag
    "done": ("#eef6ee", "#2e7d32", "已做"),
    "run": ("#fff6e0", "#c77700", "在跑"),
    "todo": ("#ffffff", "#9e9e9e", "未做"),
    "user": ("#f1ecf8", "#6a4c93", "自做"),
}

BRANCHES = [
    ("0｜机制 why（已发表，结题）", "#7a7a7a", [
        ("done", "Good-Turing 缺失质量 + 概率挤出", "论文 arXiv:2610.00394；loss-mask 2% 上下文：gap 增长 1.018→0.007 nats"),
    ]),
    ("1｜生产级会不会 fork：scaling", "#1f5f8b", [
        ("done", "n / R / 数据包络 / 模型（§74–78）", "n≤5 幂律 R²≥0.84；「1B 时 forking 消失」= 只训 0.3 遍的伪影"),
        ("done", "val 泄漏审计（§137a）", "shard 61/64/65/66 含 2.5–7.6% val 原文 → 159M 的 −340 mK 是伪影；159M/200M 去片重跑"),
        ("done", "D/N 包络 5/10/20（§137b）", "E3 gap ∝ (D/N)^−1；E3 ΔVal 50M +17/+8/+34，104M −13/+19/+28 mK"),
        ("done", "R 下探到 R/N=0.12（§137c）", "R/N≤0.25 有害、2.5–16 平台、gap 在 R/N≈4 过零；V4.1 折算 R/N=0.1（总参）或 3–4（激活）"),
        ("done", "生产形态：1 遍里 10%/30% 子集重复 3 次", "子集上 train−val：OE +0.105/+0.119，nogram +0.025/+0.031；总 val 仍 +46–83 mK → 被平均稀释"),
        ("run", "D/N=20 等比梯 35–200M（去泄漏）", "200M/159M _nl 四+二臂在 360-2；104M R/N=0.5/2 补点"),
        ("todo", "D/N=40/80 + seed 复本 + 子集 3/4 遍", "外推 D/N 曲线需要右端实测点（A100 队列 Q1）"),
        ("todo", "外推到 1B×20B / V4.1 两版", "三道墙：语料 7.8B、f30 索引 R/N=8 到 ≈200M、exact 表单卡到 ≈200M"),
    ]),
    ("2｜消除危害，保住收益", "#a33b20", [
        ("done", "低频 mask / 每阶只留高频 + 低表 LR（§82–89）", "128× 无收益；1–4× 时 f256@4× E3 −0.022、gap 0.02–0.04"),
        ("done", "表 LR 旋钮 tls + E5（§128–136）", "gap ∝ tls^1.1；R/N=16 tls0.25 到 E5 健康；R/N=32 低 tls 也会在 E3–E5 发散"),
        ("user", "可训练低频 gate f(n)（俞善斌）", "合作者在训练；基线应为「mask + 低 tls」而非裸 mask"),
    ]),
    ("3｜更好的 OE：分配与容量", "#2e6b3a", [
        ("done", "无碰撞词典 / keephash / W1 路由 / W2 聚类", "收益来自覆盖率；128k 时没有路由胜过随机双 hash"),
        ("done", "Step-2：dense/MoE × OE（50M）", "收益可加，MoE 上的 OE 收益不随 epoch 衰减"),
        ("run", "multi-head H=1/2/4/8（Engram 式，§137e）", "同参数：1.6M 上下文进 400k 行 / 400k 进 400k；H4×tls4 对照 Engram 5× LR"),
        ("todo", "FLAME 104M/159M + mHC；固定 wall-clock", "最贵的一条，等 Q1/Q2 读数后再定"),
    ]),
    ("4｜蒸馏分组（Qwen3.5-4B 教师）", "#6a4c93", [
        ("run", "10M 席位表构建：10B token，2–5gram", "5B 里程碑强审通过（07:24）；n2/n3 各 2.25M、n4 2.04M、n5 0.89M 语义槽 + fallback"),
        ("todo", "学生 harness（D0）", "64 维 value + W_V + 标量 gate、depth 0/8/16/24、Qwen 词表 u32 数据、导出→assign 转换"),
        ("todo", "四对照 + nogram（D1）", "强多头 hash / 频率匹配随机分组 / 残差分组+随机 value / 残差分组+残差 value；E1 收益、E3 forking"),
        ("todo", "A100 版：只在 depth 0 注入（D1a）", "4 深度可训练表 2.56B 参数 ≈41GB 装不下 A100，depth-0 版 640M 可跑"),
    ]),
]

LEAF_W, LEAF_H, GAP = 860, 66, 12
BR_W, BR_H = 330, 64
X_ROOT, X_BR, X_LEAF = 40, 420, 860
Y0 = 120

def esc(s):
    return html.escape(s, quote=False)

rows = sum(len(b[2]) for b in BRANCHES)
H = Y0 + rows * (LEAF_H + GAP) + (len(BRANCHES) - 1) * 26 + 120
W = X_LEAF + LEAF_W + 40
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="#fbfaf6"/>',
       f'<text x="{X_ROOT}" y="52" font-size="28" font-weight="800" fill="#111">Forking v2 实验阵列（2026-10-22）</text>',
       f'<text x="{X_ROOT}" y="86" font-size="17" fill="#555">验收：同预算、E3 末（第 3 遍末）的 Δval = nogram − OE（&gt;0 才算收益）；gap = val − train 只用来诊断。'
       f'所有训练固定 epoch 数。</text>']

y = Y0
br_pos, leaf_pos = [], []
for bi, (bt, bc, leaves) in enumerate(BRANCHES):
    ys = []
    for st, title, note in leaves:
        ys.append(y)
        leaf_pos.append((bi, st, title, note, y))
        y += LEAF_H + GAP
    br_pos.append((bt, bc, (ys[0] + ys[-1] + LEAF_H) / 2))
    y += 26

root_y = (br_pos[0][2] + br_pos[-1][2]) / 2
RW, RH = 340, 132

def curve(x1, y1, x2, y2, c, w=2.4):
    mx = (x1 + x2) / 2
    return f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" fill="none" stroke="{c}" stroke-width="{w}"/>'

for bt, bc, by in br_pos:
    svg.append(curve(X_ROOT + RW, root_y, X_BR, by, bc, 3))
for bi, st, title, note, ly in leaf_pos:
    bt, bc, by = br_pos[bi]
    svg.append(curve(X_BR + BR_W, by, X_LEAF, ly + LEAF_H / 2, bc, 1.6))

svg.append(f'<rect x="{X_ROOT}" y="{root_y - RH/2}" width="{RW}" height="{RH}" rx="16" fill="#fff" stroke="#222" stroke-width="2.5"/>')
for i, (t, sz, wt, col) in enumerate([("第二阶段目标", 18, 600, "#666"),
                                       ("消除 forking 的危害", 24, 800, "#111"),
                                       ("只享受 OE 的收益", 24, 800, "#111")]):
    svg.append(f'<text x="{X_ROOT + RW/2}" y="{root_y - RH/2 + 36 + i*36}" text-anchor="middle" '
               f'font-size="{sz}" font-weight="{wt}" fill="{col}">{esc(t)}</text>')

for bt, bc, by in br_pos:
    svg.append(f'<rect x="{X_BR}" y="{by - BR_H/2}" width="{BR_W}" height="{BR_H}" rx="14" fill="{bc}"/>')
    svg.append(f'<text x="{X_BR + BR_W/2}" y="{by + 7}" text-anchor="middle" font-size="19" font-weight="800" fill="#fff">{esc(bt)}</text>')

for bi, st, title, note, ly in leaf_pos:
    fill, stroke, tag = ST[st]
    dash = ' stroke-dasharray="7,5"' if st == "todo" else ""
    svg.append(f'<rect x="{X_LEAF}" y="{ly}" width="{LEAF_W}" height="{LEAF_H}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
    svg.append(f'<rect x="{X_LEAF + 12}" y="{ly + 12}" width="50" height="24" rx="12" fill="{stroke}"/>')
    svg.append(f'<text x="{X_LEAF + 37}" y="{ly + 29}" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">{tag}</text>')
    svg.append(f'<text x="{X_LEAF + 74}" y="{ly + 30}" font-size="17" font-weight="700" fill="#151515">{esc(title)}</text>')
    svg.append(f'<text x="{X_LEAF + 74}" y="{ly + 54}" font-size="14" fill="#444">{esc(note)}</text>')

ly = H - 52
x = X_ROOT
svg.append(f'<text x="{x}" y="{ly + 17}" font-size="15" fill="#444">图例：</text>')
x += 56
for k in ("done", "run", "todo", "user"):
    fill, stroke, tag = ST[k]
    dash = ' stroke-dasharray="7,5"' if k == "todo" else ""
    svg.append(f'<rect x="{x}" y="{ly}" width="70" height="26" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
    svg.append(f'<text x="{x + 35}" y="{ly + 18}" text-anchor="middle" font-size="14" font-weight="700" fill="{stroke}">{tag}</text>')
    x += 86
svg.append(f'<text x="{x + 10}" y="{ly + 18}" font-size="14" fill="#666">mK = 0.001 nat；R/N = 每阶表行数 ÷ (参数量/1000)；tls = 表学习率 ÷ 主干学习率；§ = docs/experiment-log.md 小节</text>')
svg.append("</svg>")

s = "".join(svg)
with open(os.path.join(OUT, "mindmap.svg"), "w") as f:
    f.write(s)
with open(os.path.join(OUT, "mindmap.html"), "w") as f:
    f.write(f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Forking v2 实验阵列</title>'
            f'<style>html,body{{margin:0;background:#fbfaf6}}</style></head><body>{s}</body></html>')
print(OUT, W, H)
