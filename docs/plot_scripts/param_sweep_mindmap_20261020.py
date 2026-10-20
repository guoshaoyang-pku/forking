#!/usr/bin/env python3
"""v2 扫参逻辑思维导图（谁在动、谁固定、D/N 是否恒定）→ docs/figs/param_sweep_mindmap_20261020/"""
import html, os

OUT = os.path.join(os.path.dirname(__file__), "..", "figs", "param_sweep_mindmap_20261020")
os.makedirs(OUT, exist_ok=True)

W, H = 1800, 1060
CARD_W, CARD_H = 560, 310

def esc(s):
    return html.escape(s, quote=False)

def box(x, y, w, h, fill, stroke, dash=False, rx=14):
    d = f' stroke-dasharray="7,5"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>'

def title_bar(x, y, w, text, color):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="44" rx="14" fill="{color}"/>'
            f'<rect x="{x}" y="{y+22}" width="{w}" height="22" fill="{color}"/>'
            f'<text x="{x+w/2}" y="{y+30}" text-anchor="middle" font-size="20" font-weight="700" fill="#fff">{esc(text)}</text>')

def bullets(x, y, lines, lh=42, size=17):
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

# root
RX, RY, RW, RH = 40, 445, 210, 150
svg.append(box(RX, RY, RW, RH, "#fff", "#333"))
svg.append(f'<text x="{RX+RW/2}" y="{RY+42}" text-anchor="middle" font-size="22" font-weight="800" fill="#111">v2 扫参逻辑</text>')
svg.append(f'<text x="{RX+RW/2}" y="{RY+78}" text-anchor="middle" font-size="18" fill="#333">谁动 · 谁固定</text>')
svg.append(f'<text x="{RX+RW/2}" y="{RY+110}" text-anchor="middle" font-size="18" fill="#c0392b">D/N 是否恒定？</text>')

cards = [
    # (cx, cy, title, bar_color, edge_color, lines[(text,bold,color)])
    (360, 90, "线 A｜机制 scaling（v2.3/v2.4，已发表）", "#5b7a99", "#5b7a99", [
        ("扫：模型 5 档 17→159M × 数据 10–50M（嵌套前缀池）", False, None),
        ("固定：295M tokens(2000 步) · 表 1M/阶 · LR 6e-4", False, None),
        ("D/N = 0.06–2.9，刻意不固定（拟合双变量幂律）", True, "#b03a2e"),
        ("产出：log gap = 2.01 + 0.108·logN − 0.170·logD", False, None),
        ("产出：tri 随模型 +0.76 ↑；n2-5 联合 −0.53 ↓", False, None),
    ]),
    (360, 570, "线 B｜配方×模型梯（§110–112，当前主线）", "#1e8449", "#1e8449", [
        ("扫：模型 35 / 63 / 104 / 159M（数据不动）", False, None),
        ("固定：1B unique × E3 · exact974k+adam+clip@1×", False, None),
        ("D/N = 28.6 / 15.9 / 9.6 / 6.3（不固定）", True, "#b03a2e"),
        ("产出：E3 收益 U 型 −0.061/−0.048/−0.025/−0.120", False, None),
        ("警示：159M 欠训 confound，勿把 50M→104M 线性外推", False, "#b03a2e"),
    ]),
    (1030, 90, "线 C｜Step-2 结构四格（§113）", "#7d3c98", "#7d3c98", [
        ("扫：dense / MoE × OE on/off（解耦 active 与 total）", False, None),
        ("固定：50M 尺度 · 1B × E3 · matched active FLOPs", False, None),
        ("产出：OE-on-MoE 零衰减 −0.038/−0.042/−0.037", True, "#1e8449"),
        ("产出：moe_oe E3 = 2.4960 全局最优（vs dense −0.216）", False, None),
        ("下一步：104M/159M 成阵 + mHC on/off", False, None),
    ]),
    (1030, 570, "目标外推 setting（未跑）", "#c0392b", "#c0392b", [
        ("恒定 D/N = 20（total tokens / 总参数）", True, "#b03a2e"),
        ("第一目标：1B 模型 × 20B tokens；MoE 大者同比例", False, None),
        ("生产端：100B dense / 750B-A10B MoE × 15–30T", False, None),
        ("其中 100B 语料重复到第 3–4 遍 → forking 区", False, None),
        ("当前唯一接近 D/N≈20 的点：63M × 1B unique", False, "#7d3c98"),
    ]),
]

for cx, cy, title, bar, ecolor, lines in cards:
    svg.append(edge(RX + RW, RY + RH/2, cx, cy + 22, ecolor))
for cx, cy, title, bar, ecolor, lines in cards:
    svg.append(box(cx, cy, CARD_W, CARD_H, "#fff", bar, dash=(bar == "#c0392b")))
    svg.append(title_bar(cx, cy, CARD_W, title, bar))
    svg.append(bullets(cx + 24, cy + 82, lines))

# footer note
svg.append(f'<text x="{W/2}" y="{H-30}" text-anchor="middle" font-size="16" fill="#555">口径：D/N 按 unique tokens 计（E=3 时 total = 3× unique）；模型参数为 backbone 参数（63M = 8L/768d 标准 setting）。源：experiment-log §76/§110–113、docs/appendices/v2_scaling_20261017</text>')
svg.append('</svg>')

page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>v2 扫参思维导图</title>
<style>body{{margin:0;background:#fafaf7;display:flex;justify-content:center}}</style></head>
<body>{''.join(svg)}</body></html>"""

with open(os.path.join(OUT, "mindmap.html"), "w") as f:
    f.write(page)
with open(os.path.join(OUT, "mindmap.svg"), "w") as f:
    f.write("".join(svg))
print(os.path.join(OUT, "mindmap.html"))
