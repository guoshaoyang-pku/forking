#!/usr/bin/env python3
"""Forking v2 实验阵列思维导图（2026-10-21 夜）：一个目标 → 三条问题线（+已结题机制线）→ 实验叶子。
叶子状态：done / running / todo / user（用户自做）。数字全部来自 docs/experiment-log.md 对应 §。
输出 docs/figs/v2_experiment_array_20261021/{mindmap.svg,mindmap.html}；PNG 由 Chrome headless 截图。"""
import html, os

OUT = os.path.join(os.path.dirname(__file__), "..", "figs", "v2_experiment_array_20261021")
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
    ("1｜生产级会不会 fork：scaling law", "#1f5f8b", [
        ("done", "n 扫描：2–8 单独 + 2–5 联合（§76）", "n≤5 gap–频率幂律 R²≥0.84，α=0.21/0.35/0.31/0.25；联合 2–5 gap 8.23（与碰撞负载混淆）"),
        ("done", "表大小 R 扫描：n=2…2–7 × 10 点（§74/76）", "R≤1k gap≈0，≥100k 进高原；n≥5 联合在 300k 见峰 → 碰撞集中比总容量更推高 gap"),
        ("done", "数据包络 + 固定 epoch 协议（§76/78）", "「1B 时 forking 消失」= 固定步数只训 0.3 遍的伪影；≥4 阶联合在 37M 内几乎不降"),
        ("done", "模型大小，数据不固定（线 A，§76）", "log gap = 2.01 + 0.108·logN − 0.170·logD；单独 trigram 17→159M：gap 10.8→13.6"),
        ("run", "D/N=20 等比梯 17–200M（§116–135）", "dGap ∝ N^0.12 几乎不动；Δval 变薄 +52→+38→+33 mK；存活率 17M 0/2·35M 3/4·≥50M 全活；200M 在跑"),
        ("todo", "生产形态：大池中一小段数据重复 3–4 遍", "最像 DS 的情形（15–30T 里 100B 重复 3–4 遍）；需新增「fresh+replay 混合」数据器"),
        ("todo", "等比梯上的 gap–频率幂律", "每个 token 的统计（per-token dump）已落盘，尚未统计；回答「α 随 N 怎么变」"),
        ("todo", "外推 1B×20B / 100B / 750B-A10B", "三道墙：语料 7.8B → dense ≤390M；f30 索引 R/N=8 只撑到 ≈200M；exact 表显存单卡只到 ≈200M"),
    ]),
    ("2｜消除 forking 危害，保住收益", "#a33b20", [
        ("done", "硬 mask 低频（论文小 setting）", "DS 风格复现：屏蔽 f≤100 的读取，gap 3.32→0.82，val 改善 1.60"),
        ("done", "n=2–5 每阶只留高频 @1B×E3（§82–89）", "表 LR 128× 时：gap 0.93→0.10，但收益≈0；降到 1–4× 后 E1 −0.07～−0.10，f256@4× E3 仍 −0.022、gap 0.02–0.04"),
        ("done", "按频率截断 = exact top-R 的 R 扫描（§120–122）", "R/N 8→16：E3 train −0.050，val 只 −0.0004 → 多出来的低频行几乎全变成记忆"),
        ("done", "表 LR 旋钮 tls（§128–131）", "gap ∝ tls^1.1；R/N≤16 时 Δval 不变；R/N=32+tls0.25 E3 +42 mK 最好；代价：E1 收益少 30–57%"),
        ("done", "发散：adam+clip（§98–99）+ 冻表（§79–80）", "clip 救活 16× 发散；冻随机表 gap 0.064、计数初始化冻表 0.022（对照 2.70）"),
        ("run", "E5：低 tls 的收益能不能撑过第 3 遍", "50M R/N=16/32 × tls0.25 × s43 的 E5 臂 + nogram 配对（360-2 GPU3/4/5，今晚刚起）"),
        ("todo", "可训练低频 gate f(n)（俞善斌）", "每阶一个 sigmoid(频率) 门，用不重复、不与训练/验证重合的数据训练；本仓库还没有代码"),
        ("todo", "OE 读取 dropout / 低频上下文合并", "论文 Q9 提到的两条修法，尚无干净实验"),
    ]),
    ("3｜端到端 OE 收益 / 更好的 OE", "#2e6b3a", [
        ("done", "1B 第一遍收益是真的（§89）", "E1 train 收益 = val 收益（±0.002）；−0.07～−0.10，落在 OE/SCONE 文献区间"),
        ("done", "无碰撞词典 + keephash 三方对照（§73/77）", "覆盖率主导（−0.75～−5.67）；kept 集合内去碰撞反而 +0.25～+0.64 → 收益来自「只留高频」"),
        ("done", "路由 W1 + 残差聚类多层 OE W2（§90–107）", "128k 时没有路由胜过随机双 hash；随机初始化三层 = exact974k；质心初始化在中层有害"),
        ("done", "Step-2：dense/MoE × OE（50M，§113–114）", "两者收益可加（偏差 0.002）；MoE 上的 OE 收益不随 epoch 衰减；moe_oe 比 dense 好 0.216"),
        ("todo", "固定训练时长（wall-clock）比较", "目前只做了 matched active FLOPs；查表开销没有计入"),
        ("todo", "FLAME 104M/159M 成阵 + mHC on/off", "第二步扩展；FLAME MoE 一臂 16–22 H200·h，最贵的一条线"),
        ("user", "n-gram 蒸馏 / VQ-VAE 预训练 OE 模块", "郭绍阳在做；接口：作为表初始化或合并方案进入第 2 条线比较"),
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
       f'<text x="{X_ROOT}" y="52" font-size="28" font-weight="800" fill="#111">Forking v2 实验阵列（2026-10-21）</text>',
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
