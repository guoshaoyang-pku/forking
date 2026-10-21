#!/usr/bin/env python3
"""v2 实验思维导图（2026-10-21 21:4x）——E=3 固定 epoch、E3 末尾验收口径下的全部轴与依赖。
输出 → docs/figs/param_sweep_mindmap_20261021/（mindmap.svg / mindmap.html）
数值来源：docs/experiment-log.md §110–§132(n)、data/runs_fixed/*_e3_fd_fixed、code/cluster/run_v26_dn20.sh
"""
import html, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "param_sweep_mindmap_20261021")
os.makedirs(OUT, exist_ok=True)

W, H = 1660, 1900
CARD_X, CARD_W, CARD_H = 840, 720, 320
CARD_Y = [60, 410, 760, 1110, 1460]

FONT = "PingFang SC, Hiragino Sans GB, Microsoft YaHei, sans-serif"
GREEN, RED, BLUE, OLIVE, PURPLE, GREY = "#1e8449", "#b03a2e", "#5b7a99", "#8e6e3a", "#7d3c98", "#555"


def esc(s):
    return html.escape(s, quote=False)


def box(x, y, w, h, stroke, fill="#fff", dash=False, rx=14):
    d = ' stroke-dasharray="7,5"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"{d}/>')


def title_bar(x, y, w, text, color):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="46" rx="14" fill="{color}"/>'
            f'<rect x="{x}" y="{y+24}" width="{w}" height="22" fill="{color}"/>'
            f'<text x="{x+w/2}" y="{y+31}" text-anchor="middle" font-size="20" '
            f'font-weight="700" fill="#fff">{esc(text)}</text>')


def lines(x, y, rows, lh=32, size=16.5):
    out = []
    for i, (txt, bold, col) in enumerate(rows):
        wt = ' font-weight="700"' if bold else ""
        c = f' fill="{col}"' if col else ' fill="#222"'
        out.append(f'<text x="{x}" y="{y + i*lh}" font-size="{size}"{wt}{c}>{esc(txt)}</text>')
    return "\n".join(out)


def edge(x1, y1, x2, y2, color, width=2.5):
    mx = (x1 + x2) / 2
    return (f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" fill="none" '
            f'stroke="{color}" stroke-width="{width}"/>')


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{FONT}">',
       f'<rect width="{W}" height="{H}" fill="#fafaf7"/>']

# ── 根：主线 ──
ROOT = (40, 600, 300, 440)
svg.append(box(ROOT[0], ROOT[1], ROOT[2], ROOT[3], "#111"))
svg.append(f'<text x="{ROOT[0]+ROOT[2]/2}" y="{ROOT[1]+46}" text-anchor="middle" font-size="23" '
           f'font-weight="800" fill="#111">v2 主线</text>')
svg.append(lines(ROOT[0]+22, ROOT[1]+92, [
    ("目标：OE 收益最大化", True, None),
    ("× forking 最小化", True, None),
    ("", False, None),
    ("外推目标 setting：", False, RED),
    ("恒定 D/N = 20。", False, None),
    ("1B×20B → 100B×2T", False, None),
    ("MoE 750B-A10B ×15–30T", False, None),
    ("", False, None),
    ("验收：E3 末尾 dVal < 0", True, GREEN),
    ("（同 seed 配对 nogram）", False, GREY),
]))

# ── 口径锁：全部训练共用 ──
LK = (400, 60, 380, 1720)
svg.append(box(LK[0], LK[1], LK[2], LK[3], "#111"))
svg.append(f'<text x="{LK[0]+LK[2]/2}" y="{LK[1]+40}" text-anchor="middle" font-size="19" '
           f'font-weight="800" fill="#111">口径锁（不变量）</text>')
svg.append(f'<line x1="{LK[0]+16}" y1="{LK[1]+58}" x2="{LK[0]+LK[2]-16}" y2="{LK[1]+58}" '
           f'stroke="#ccc" stroke-width="1.5"/>')
svg.append(lines(LK[0]+20, LK[1]+92, [
    ("① 固定 epoch 数", True, None),
    ("--train_epochs E。本梯 E=3", False, None),
    ("launcher 默认 E=3，可覆写", False, None),
    ("NGLAB_EPOCHS（不改代码）", False, None),
    ("", False, None),
    ("② 验收位置 = E3 末尾", True, None),
    ("末 60 行窗（每 10 步一记）", False, None),
    ("同 seed 配对 nogram，同机", False, None),
    ("Δval = arm − nogram", False, None),
    ("负 = 赢（验收通过）", False, GREEN),
    ("", False, None),
    ("③ 步数由池自动定", True, None),
    ("steps = E × 池 / 总 batch", False, None),
    ("17M: 6915 = 3 × 2305", False, None),
    ("", False, None),
    ("④ D/N = 20", True, None),
    ("（unique token 口径）", False, None),
    ("", False, None),
    ("⑤ 表：exact 零碰撞", True, None),
    ("R = 每阶行数 = distinct+1", False, None),
    ("ρ = R / (N/1000)", False, None),
    ("", False, None),
    ("⑥ 跨机门槛 = 语义量", True, None),
    ("min_count + R + counts", False, None),
    ("（不比 md5：两机来源不同）", False, PURPLE),
    ("", False, None),
    ("⑦ 诊断 ≠ 验收", True, None),
    ("gap 只看方向与机制", False, None),
    ("dVal 才是验收标准", False, GREEN),
    ("", False, None),
    ("⑧ run_id 约定", True, None),
    ("诊断臂带 _fd（fast-diag 管线）", False, None),
    ("改口径即新起 run_id", False, None),
], lh=42, size=16))

# ── 五个分支 ──
CARDS = [
    (BLUE, False, "输入｜已闭环的三条线", [
        ("线 A 机制 scaling（已发表）：扫 17→159M × 10–50M", False, None),
        ("  → log gap = 2.01 + 0.108·logN − 0.170·logD", False, None),
        ("线 B 配方×模型梯（§110–112）：钉 1B×E3 扫模型", False, None),
        ("  → 159M 欠训 confound ⇒ 必须恒定 D/N", False, RED),
        ("判决线（§98–99/115）：adam+clip 救活发散臂", False, None),
        ("  exact 零碰撞复活 E3 −0.035、零发散；PERM≈C3", False, None),
        ("线 C 结构四格（§113–114）：OE×MoE 可加", False, None),
        ("  moe_oe E3 2.4960 全局最优", False, None),
    ]),
    (RED, False, "在扫｜iso-ratio 梯 + 剂量窗", [
        ("扫：N=17/35/50/104/159M，D=20N，ρ=8/16/32", True, None),
        ("固定：E=3、tls=1（除剂量窗）、同 seed 配对 nogram", False, None),
        ("R 跟踪 N（r136k/r272k/r544k/r832k/r1272k）", False, None),
        ("在跑（360-2 第三批 7 臂）：ρ=8 tls0.125", False, None),
        ("  ρ=16 tls0.375/0.625；tls0.5 的 s44/s45 复本", False, None),
        ("剂量窗：剂量 = Σ命中 × E × 6e-4·tls", False, None),
        ("  ρ=8/16/32 @tls1 → 5.16 / 3.01 / 1.72", False, None),
        ("  ⇒ R/N 与 tls 是同一剂量的两个旋钮", True, PURPLE),
    ]),
    (GREEN, False, "判决｜本轮结果（E3 末尾读）", [
        ("104M 最干净：E3 val 平稳，2 seed 终局 −0.028/−0.038", True, GREEN),
        ("50M 可用：ρ=8 两 seed 终局 dGap +0.0670/+0.0674", True, GREEN),
        ("17M 剔除：11 臂 done，E3 窗 4 seed 仅 1/4 通过", True, RED),
        ("  （−0.064 / +0.866 / +0.214 / +0.392）", False, None),
        ("  ρ=8 无解：tls0.125 在 E1 即输（+0.064）", False, None),
        ("35M 待审：台账 ρ=8 记 +49 mK 的那一臂", True, RED),
        ("  = 被记「发散、final val 3.42」的同一臂", False, None),
        ("159M 剔除：每 epoch 末段 val 骤降-回爬", False, RED),
    ]),
    (OLIVE, False, "外推｜能不能（分岔点）", [
        ("干净梯只剩 50M–104M（×2），35M 未审 ⇒ 现不能外推", True, RED),
        ("分岔点：dGap 50M→104M 已持平（0.0670→0.0689）", False, None),
        ("  读作饱和 0.07 → 净收益 ≈30 mK 不消失", False, None),
        ("  读作 N^0.10 → 净收益在 N≈4.3B 归零", False, None),
        ("  两读法都拟合现有 3 点 ⇒ 这才是分岔点", True, PURPLE),
        ("表 ∝ N^1.31：100B 按 ρ=8 约定 → 12 TB / 61×", False, None),
        ("  按固定 1M 词表 → ρ=0.01（低 800×）→ 趋零", False, None),
        ("⇒ 模型规模不决定答案，「表开多大」决定", True, RED),
    ]),
    (PURPLE, True, "下一步｜按依赖排序", [
        ("S0 窗口口径审计（末 600 步 vs 末 60 行 vs 全 E3）", False, None),
        ("  → 修 35M，恢复 ×3 干净梯（外推前提）", False, None),
        ("S1 现有语料爬 200/300/400M（D/N=20、ρ=8/16、3 seed）", False, None),
        ("  ~13 H200-h/臂，8 卡 ~10 h → 跨度 ×2 拉到 ×11", False, None),
        ("S2 拿 20B unique → 1B×20B 锚点（本地 8B，差 2.5×）", False, None),
        ("S3 干预线（mask_low / freeze / count-init）@400M", False, None),
        ("  目标：把「被 gap 吃掉」的 64–80% 压下来", False, None),
        ("  注意：ρ 与 tls 做不到这件事", True, RED),
    ]),
]

svg.append(edge(ROOT[0] + ROOT[2], ROOT[1] + ROOT[3] / 2, LK[0], LK[1] + LK[3] / 2, "#111", 3))
svg.append(f'<text x="{ROOT[0]+ROOT[2]+8}" y="{ROOT[1]+ROOT[3]/2-10}" font-size="15" '
           f'fill="#111" font-weight="700">先锁口径</text>')

for (color, dash, title, rows), cy in zip(CARDS, CARD_Y):
    svg.append(box(CARD_X, cy, CARD_W, CARD_H, color, dash=dash))
    svg.append(title_bar(CARD_X, cy, CARD_W, title, color))
    svg.append(lines(CARD_X + 26, cy + 82, rows, lh=32, size=16.5))
    svg.append(edge(LK[0] + LK[2], LK[1] + LK[3] / 2, CARD_X, cy + CARD_H / 2, color, 2))

# 分支间的依赖标签
CHAIN = ["提供口径与 setting", "产出判决", "决定能否外推", "据此排下一步"]
for i, lab in enumerate(CHAIN):
    y = (CARD_Y[i] + CARD_H + CARD_Y[i + 1]) / 2
    svg.append(f'<text x="{CARD_X+18}" y="{y+5}" font-size="15" font-weight="700" '
               f'fill="#666">↓ {esc(lab)}</text>')

svg.append(f'<text x="{W/2}" y="{H-70}" text-anchor="middle" font-size="15.5" fill="{GREY}">'
           f'图例：绿=已证实/通过，红=失效/卡点，紫=口径注意，虚线框=未跑或待审。左→右 = 主线 · 口径锁 · 分支，分支间 ↓ 为依赖。</text>')
svg.append(f'<text x="{W/2}" y="{H-40}" text-anchor="middle" font-size="15.5" fill="{GREY}">'
           f'源：docs/experiment-log.md §110–§132(n)、data/runs_fixed/*_e3_fd_fixed、code/cluster/run_v26_dn20.sh（E=3 默认）。'
           f'所有训练固定 epoch 数，验收一律读 E3 末尾。</text>')
svg.append('</svg>')

body = "".join(svg)
page = (f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
        f'<title>v2 实验思维导图（E=3 固定 epoch）</title>'
        f'<style>body{{margin:0;background:#fafaf7;display:flex;justify-content:center}}</style>'
        f'</head><body>{body}</body></html>')

with open(os.path.join(OUT, "mindmap.svg"), "w") as f:
    f.write(body)
with open(os.path.join(OUT, "mindmap.html"), "w") as f:
    f.write(page)
print(os.path.join(OUT, "mindmap.html"))
