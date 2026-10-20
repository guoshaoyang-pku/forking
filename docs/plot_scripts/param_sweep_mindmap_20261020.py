#!/usr/bin/env python3
"""v2 扫参逻辑思维导图 v2（2026-10-20 晚：W1 R-scaling + D/N=20 不变量梯 + 判决线）→ docs/figs/param_sweep_mindmap_20261020/"""
import html, os

OUT = os.path.join(os.path.dirname(__file__), "..", "figs", "param_sweep_mindmap_20261020")
os.makedirs(OUT, exist_ok=True)

W, H = 2460, 1180
CARD_W, CARD_H = 600, 300

def esc(s):
    return html.escape(s, quote=False)

def box(x, y, w, h, fill, stroke, dash=False, rx=14):
    d = f' stroke-dasharray="7,5"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>'

def title_bar(x, y, w, text, color):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="44" rx="14" fill="{color}"/>'
            f'<rect x="{x}" y="{y+22}" width="{w}" height="22" fill="{color}"/>'
            f'<text x="{x+w/2}" y="{y+30}" text-anchor="middle" font-size="20" font-weight="700" fill="#fff">{esc(text)}</text>')

def bullets(x, y, lines, lh=40, size=16):
    out = []
    for i, (txt, bold, col) in enumerate(lines):
        w = ' font-weight="700"' if bold else ""
        c = f' fill="{col}"' if col else ' fill="#222"'
        out.append(f'<text x="{x}" y="{y + i*lh}" font-size="{size}"{w}{c}>{esc(txt)}</text>')
    return "\n".join(out)

def edge(x1, y1, x2, y2, color):
    mx = (x1 + x2) / 2
    return f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" fill="none" stroke="{color}" stroke-width="2.5"/>'

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="#fafaf7"/>']

RX, RY, RW, RH = 40, 500, 230, 160
svg.append(box(RX, RY, RW, RH, "#fff", "#333"))
svg.append(f'<text x="{RX+RW/2}" y="{RY+42}" text-anchor="middle" font-size="22" font-weight="800" fill="#111">v2 扫参逻辑</text>')
svg.append(f'<text x="{RX+RW/2}" y="{RY+78}" text-anchor="middle" font-size="18" fill="#333">谁动 · 谁固定</text>')
svg.append(f'<text x="{RX+RW/2}" y="{RY+112}" text-anchor="middle" font-size="18" fill="#c0392b">不变量是什么？</text>')

cards = [
    # ── 左列：历史/已闭环 ──
    (360, 60, "线 A｜机制 scaling（已发表）", "#5b7a99", [
        ("扫：模型 17→159M × 数据 10–50M（双变量幂律）", False, None),
        ("D/N = 0.06–2.9 刻意不固定", True, "#b03a2e"),
        ("产出：log gap = 2.01 + 0.108·logN − 0.170·logD", False, None),
        ("这是 forking gap 的 scaling，不是 OE 收益的", False, "#7d3c98"),
    ]),
    (360, 410, "线 B｜配方×模型梯（§110–112，已闭环）", "#1e8449", [
        ("扫：模型 35/63/104/159M；固定 1B×E3", False, None),
        ("D/N 28.6→6.3 不固定 → 159M 欠训 confound", True, "#b03a2e"),
        ("产出：E3 U 型 −0.061/−0.048/−0.025/−0.120", False, None),
        ("教训：欠训区表收益虚高 → 必须恒定 D/N", False, "#b03a2e"),
    ]),
    (360, 760, "判决线｜优化器与路由（§98–99,115，已闭环）", "#8e6e3a", [
        ("adam+clip 救活 @16× 2/2 发散臂（clip 是关键）", True, "#1e8449"),
        ("exact974k 零碰撞复活：E3 −0.035 零发散", False, None),
        ("PERM≈C3（差 0.002）→ 占用结构承载收益，", True, "#1e8449"),
        ("语义标签免费 → 聚类重点放流量/负载", True, "#1e8449"),
    ]),
    # ── 中列：当前在扫 ──
    (1090, 60, "W1｜R-scaling（§115–116，在扫）", "#b03a2e", [
        ("扫：R/阶 64k→974k（H2 路由，adam+clip@1×）", False, None),
        ("固定：50M × 1B unique × E3（D/N=20）", False, None),
        ("E1 收益：−0.027/−0.036/−0.046/−0.038（↑至256k）", False, None),
        ("E3 收益：−0.010/−0.021/−0.032/−0.031（饱和512k）", True, "#b03a2e"),
        ("RMSProp 时代 4 档另作对照，不混画", False, "#7d3c98"),
    ]),
    (1090, 410, "线 B′｜D/N=20 不变量梯（§116，在扫）", "#b03a2e", [
        ("扫：N=35/50/104/159M，D=20N，R=8N/阶 exact", True, "#b03a2e"),
        ("不动骨干格点（6L/640 … 12L/1024），R 跟踪 N", False, None),
        ("35m@tls0.5 + 104m 对在跑（360-2），159m 待铺", False, None),
        ("检验：E1 收益 / E2-E3 衰减率 / 发散模态", False, None),
        ("≈常数 → (D/N,R/N) 即不变量，外推有据", True, "#1e8449"),
    ]),
    (1090, 760, "线 C｜Step-2 结构四格（§113–114，已完成）", "#7d3c98", [
        ("扫：dense/MoE × OE on/off（50M，matched active）", False, None),
        ("OE×MoE 可加（偏差 0.002）；OE-on-MoE 零衰减", True, "#1e8449"),
        ("moe_oe E3 2.4960 全局最优（−0.216 vs dense）", False, None),
        ("moe_oe_exact 在跑；104M/159M 成阵 + mHC 待铺", False, None),
    ]),
    # ── 右列：目标与正交 ──
    (1800, 60, "目标外推 setting（未跑）", "#c0392b", [
        ("恒定 D/N=20：1B 模型×20B → 100B×2T", True, "#b03a2e"),
        ("MoE 750B-A10B × 15–30T（DSv4.1-Flash 档）", False, None),
        ("100B 语料重复 3–4 遍 → forking 区", False, None),
        ("Engram 对标：R-scaling 幂律 39× 未饱和", False, "#7d3c98"),
    ]),
    (1800, 410, "正交探针（梯子后第二轮）", "#c0392b", [
        ("钉 D/N=20 变 R/N ∈ {2,4,8,16}", False, None),
        ("钉 R/N=8 变 D/N ∈ {10,20,40}", False, None),
        ("假设：R/N 驱动收益幅度，", True, "#1e8449"),
        ("D/N 驱动 forking 衰减率（可证伪）", True, "#1e8449"),
        ("R/N 口径须钉死：active 还是 total（MoE 差 75×）", False, "#b03a2e"),
    ]),
    (1800, 760, "大档与结构扩展（第三轮）", "#c0392b", [
        ("159m × 3.18B（梯子 L 档，数据待搬 74 片）", False, None),
        ("1B 模型 × 20B tokens（目标 setting 第一点）", False, None),
        ("mHC on/off @104M；多层注入 offload residual", False, None),
        ("train-then-merge 聚类表（占用加权，非语义）", False, None),
    ]),
]

for cx, cy, title, bar, lines in cards:
    svg.append(edge(RX + RW, RY + RH/2, cx, cy + 22, bar))
for cx, cy, title, bar, lines in cards:
    svg.append(box(cx, cy, CARD_W, CARD_H, "#fff", bar, dash=(bar == "#c0392b")))
    svg.append(title_bar(cx, cy, CARD_W, title, bar))
    svg.append(bullets(cx + 24, cy + 80, lines))

svg.append(f'<text x="{W/2}" y="{H-58}" text-anchor="middle" font-size="16" fill="#555">图例：实线框=已闭环/在扫（红=在扫），虚线框=未跑；绿字=已证实结论，红字=警示/confound，紫字=口径注意。列：历史 → 在扫 → 目标。</text>')
svg.append(f'<text x="{W/2}" y="{H-28}" text-anchor="middle" font-size="16" fill="#555">口径：D/N 按 unique tokens 计；R = 每阶表行数（exact 零碰撞 = distinct+1，hash = bucket 数）；N = backbone 参数。源：experiment-log §76/§98–116、Engram arXiv:2601.07372</text>')
svg.append('</svg>')

page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>v2 扫参思维导图 v2</title>
<style>body{{margin:0;background:#fafaf7;display:flex;justify-content:center}}</style></head>
<body>{''.join(svg)}</body></html>"""

with open(os.path.join(OUT, "mindmap.html"), "w") as f:
    f.write(page)
with open(os.path.join(OUT, "mindmap.svg"), "w") as f:
    f.write("".join(svg))
print(os.path.join(OUT, "mindmap.html"))
