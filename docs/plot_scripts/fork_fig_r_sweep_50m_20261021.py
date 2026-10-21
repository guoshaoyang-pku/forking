#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-panel figure for the completed 50M exact R/N sweep (§122(a) + §122(0)).

Seven points (R/N = 0, 2, 2.6, 5, 8, 16, 32) read at each epoch's last logged
step, sourced directly from `docs/experiment-log.md` §122(a). Render a vector
SVG, then use Chrome headless to capture the SVG at its native canvas size as PNG.
"""
from __future__ import annotations

import html
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figs" / "v26_r_sweep_50m_20261021"
OUT.mkdir(parents=True, exist_ok=True)
SVG_PATH = OUT / "fig_r_sweep_50m.svg"
PNG_PATH = OUT / "fig_r_sweep_50m.png"

# Read the committed experiment-log table rather than duplicating result
# numbers in the drawing code. This deliberately limits the source to §122(a).
LOG_PATH = ROOT / "docs" / "experiment-log.md"
LOG_SECTION = LOG_PATH.read_text(encoding="utf-8").split("### §122", 1)[1].split("\n### §", 1)[0]
ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*"
    r"([0-9.]+)/([0-9.]+)\s*\([^)]*\)\s*\|\s*"
    r"([0-9.]+)/([0-9.]+)\s*\([^)]*\)\s*\|\s*"
    r"([0-9.]+)/([0-9.]+)\s*\([^)]*\)\s*\|"
)
RUN_TO_RN = {
    "nogram": 0.0,
    "r100k": 2.0,
    "r128k": 2.6,
    "r256k": 5.0,
    "r400k": 8.0,
    "r800k": 16.0,
    "r1600k": 32.0,
}


def read_epoch_boundary_data() -> dict[float, tuple[float, float, float, float, float, float]]:
    result: dict[float, tuple[float, float, float, float, float, float]] = {}
    for line in LOG_SECTION.splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        arm, ratio, e1_train, e1_val, e2_train, e2_val, e3_train, e3_val = match.groups()
        arm_key = "nogram" if arm.strip() == "nogram" else arm.strip()
        if arm_key not in RUN_TO_RN:
            continue
        r_n = RUN_TO_RN[arm_key]
        if abs(float(ratio) - r_n) > 1e-9:
            raise ValueError(f"Unexpected R/N for §122 arm {arm_key}: {ratio}")
        # Store columns as (E1 val, E1 train, E2 val, E2 train, E3 val, E3 train).
        result[r_n] = tuple(map(float, (e1_val, e1_train, e2_val, e2_train, e3_val, e3_train)))
    if set(result) != set(RUN_TO_RN.values()):
        raise ValueError(f"§122(a) did not provide all seven sweep points: {sorted(result)}")
    return result


DATA = read_epoch_boundary_data()
XS = sorted(DATA)

# The positive R/N values use log2 spacing. The zero/no-table point is shown
# separately to the left; this preserves the requested tick order and spacing.
def xcoord(ratio: float) -> float:
    return 0.0 if ratio == 0 else math.log2(ratio)

X_TICKS = [(xcoord(r), label, sublabel) for r, label, sublabel in [
    (0.0, "0", "(nogram)"),
    (2.0, "2", None),
    (2.6, "2.6", None),
    (5.0, "5", None),
    (8.0, "8", None),
    (16.0, "16", None),
    (32.0, "32", None),
]]
X_MIN, X_MAX = -0.42, 5.30

# Repo palette from recent scaling plots.
C_E1 = "#1f77b4"
C_E2 = "#2ca02c"
C_E3 = "#d62728"
C_TRAIN = "#d62728"
C_VAL = "#1f77b4"
C_GAP = "#7d3c98"
C_NOGRAM = "#999999"
C_TEXT = "#222222"
C_MUTED = "#60666d"
C_GRID = "#dfe3e8"
C_VGRID = "#f0f2f4"

W, H = 1800, 1090
# Compress the plot columns horizontally while retaining text size so the figure
# remains readable when displayed at a typical blog width (~1200 CSS pixels).
SCALE_X = 0.625
MARGIN_X = 45 * SCALE_X
COL_LEFTS = tuple(x * SCALE_X for x in (45, 945, 1845))
COL_W = 890 * SCALE_X
PLOT_TOP, PLOT_BOTTOM = 258, 690
PLOT_X_PAD_LEFT, PLOT_X_PAD_RIGHT = 104 * SCALE_X, 18 * SCALE_X
FONT_STACK = "PingFang SC, PingFang HK, Hiragino Sans GB, Heiti TC, Microsoft YaHei, sans-serif"

svg: list[str] = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT_STACK}" role="img" aria-labelledby="fig-title fig-desc">',
    '<title id="fig-title">50M nanoGPT 50M R/N sweep: epoch loss, memorization split, and E3 gap</title>',
    '<desc id="fig-desc">Three panels show validation loss by epoch, E3 train and validation loss, and E3 validation-minus-training gap over R/N values 0, 2, 2.6, 5, 8, 16, and 32.</desc>',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    f'<style>text {{ font-family: {html.escape(FONT_STACK, quote=True)}; }} .axis {{ stroke: #555b61; stroke-width: 1.5; }} .tick {{ fill: #596068; }}</style>',
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def add_text(
    x: float,
    y: float,
    value: str,
    size: float = 20,
    fill: str = C_TEXT,
    anchor: str = "start",
    weight: int | str = 400,
    transform: str | None = None,
    italic: bool = False,
) -> None:
    transform_attr = f' transform="{transform}"' if transform else ""
    italic_attr = ' font-style="italic"' if italic else ""
    svg.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{transform_attr}{italic_attr}>{esc(value)}</text>'
    )


def panel_x_bounds(index: int) -> tuple[float, float]:
    left = COL_LEFTS[index] + PLOT_X_PAD_LEFT
    right = COL_LEFTS[index] + COL_W - PLOT_X_PAD_RIGHT
    return left, right


def map_x(value: float, index: int) -> float:
    left, right = panel_x_bounds(index)
    return left + ((value - X_MIN) / (X_MAX - X_MIN)) * (right - left)


def map_y(value: float, ymin: float, ymax: float) -> float:
    return PLOT_TOP + (ymax - value) / (ymax - ymin) * (PLOT_BOTTOM - PLOT_TOP)


def add_panel_header(index: int, heading: str, subtitle: str) -> None:
    x = COL_LEFTS[index] + 12
    add_text(x, 145, heading, size=25, fill="#151719", weight=700)
    add_text(x, 178, subtitle, size=18, fill=C_MUTED)


def add_legend_item(
    x: float,
    y: float,
    label: str,
    color: str,
    shape: str = "circle",
    dash: str | None = None,
) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(
        f'<line x1="{x:.2f}" y1="{y - 6:.2f}" x2="{x + 36:.2f}" y2="{y - 6:.2f}" '
        f'stroke="{color}" stroke-width="3.2"{dash_attr}/>'
    )
    cx = x + 18
    cy = y - 6
    if shape == "square":
        svg.append(f'<rect x="{cx - 5:.2f}" y="{cy - 5:.2f}" width="10" height="10" fill="{color}"/>')
    elif shape == "diamond":
        svg.append(f'<path d="M {cx:.2f} {cy - 6:.2f} L {cx + 6:.2f} {cy:.2f} L {cx:.2f} {cy + 6:.2f} L {cx - 6:.2f} {cy:.2f} Z" fill="{color}"/>')
    elif not dash:
        svg.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" fill="{color}"/>')
    add_text(x + 47, y, label, size=17, fill="#343a40")


def add_plot_frame(
    index: int,
    ymin: float,
    ymax: float,
    yticks: list[float],
    ylabel: str,
    yformatter,
    *,
    highlight_from: float | None = None,
    zero_line: bool = False,
) -> tuple[float, float]:
    left, right = panel_x_bounds(index)
    clip_id = f"plot-{index}"
    svg.append(
        f'<defs><clipPath id="{clip_id}"><rect x="{left:.2f}" y="{PLOT_TOP}" '
        f'width="{right - left:.2f}" height="{PLOT_BOTTOM - PLOT_TOP}"/></clipPath></defs>'
    )
    if highlight_from is not None:
        hx = map_x(xcoord(highlight_from), index)
        svg.append(
            f'<rect x="{hx:.2f}" y="{PLOT_TOP}" width="{right - hx:.2f}" '
            f'height="{PLOT_BOTTOM - PLOT_TOP}" fill="{"#fff0f0" if index == 1 else "#f5effa"}"/>'
        )
    # Light vertical guides align all six requested R/N ticks.
    for tick, _, _ in X_TICKS:
        x = map_x(tick, index)
        svg.append(f'<line x1="{x:.2f}" y1="{PLOT_TOP}" x2="{x:.2f}" y2="{PLOT_BOTTOM}" stroke="{C_VGRID}" stroke-width="1"/>')
    for tick in yticks:
        y = map_y(tick, ymin, ymax)
        stroke = "#8f969d" if zero_line and abs(tick) < 1e-12 else C_GRID
        width = 1.5 if zero_line and abs(tick) < 1e-12 else 1.0
        dash = ' stroke-dasharray="7 6"' if zero_line and abs(tick) < 1e-12 else ""
        svg.append(f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" stroke="{stroke}" stroke-width="{width}"{dash}/>')
        add_text(left - 13, y + 6, yformatter(tick), size=18, fill="#596068", anchor="end")
    svg.append(f'<line class="axis" x1="{left:.2f}" y1="{PLOT_TOP}" x2="{left:.2f}" y2="{PLOT_BOTTOM}"/>')
    svg.append(f'<line class="axis" x1="{left:.2f}" y1="{PLOT_BOTTOM}" x2="{right:.2f}" y2="{PLOT_BOTTOM}"/>')
    for tick, label, sublabel in X_TICKS:
        x = map_x(tick, index)
        svg.append(f'<line x1="{x:.2f}" y1="{PLOT_BOTTOM}" x2="{x:.2f}" y2="{PLOT_BOTTOM + 7}" stroke="#555b61" stroke-width="1.3"/>')
        add_text(x, PLOT_BOTTOM + 31, label, size=19, fill="#343a40", anchor="middle")
        if sublabel:
            add_text(x, PLOT_BOTTOM + 54, sublabel, size=16, fill="#596068", anchor="middle")
    add_text((left + right) / 2, 782, "R/N = 每阶表行数 / backbone 参数", size=20, fill="#343a40", anchor="middle")
    ymid = (PLOT_TOP + PLOT_BOTTOM) / 2
    ylabel_x = COL_LEFTS[index] + 2
    add_text(ylabel_x, ymid, ylabel, size=19, fill="#343a40", anchor="middle", transform=f"rotate(-90 {ylabel_x} {ymid})")
    return left, right


def add_series(
    index: int,
    values: list[float],
    ymin: float,
    ymax: float,
    color: str,
    marker: str = "circle",
    width: float = 4.0,
    dash: str | None = None,
) -> None:
    coords = [(map_x(xcoord(r), index), map_y(v, ymin, ymax)) for r, v in zip(XS, values)]
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(
        f'<g clip-path="url(#plot-{index})"><polyline points="{points}" fill="none" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
    )
    for x, y in coords:
        if marker == "square":
            svg.append(f'<rect x="{x - 7:.2f}" y="{y - 7:.2f}" width="14" height="14" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')
        elif marker == "diamond":
            svg.append(f'<path d="M {x:.2f} {y - 8:.2f} L {x + 8:.2f} {y:.2f} L {x:.2f} {y + 8:.2f} L {x - 8:.2f} {y:.2f} Z" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')
        else:
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
    svg.append('</g>')


# Global title and study coordinates.
add_text(W / 2, 52, "50M nanoGPT × 1B unique tokens × 3 epochs", size=33, fill="#17191b", anchor="middle", weight=700)
add_text(W / 2, 84, "ClimbMix  ·  exact 零碰撞 n-gram 表（orders 2–5）  ·  Adam + grad clip  ·  table LR 1×  ·  seed 42", size=19, fill="#555d65", anchor="middle")
svg.append(f'<line x1="45" y1="108" x2="{W - 45}" y2="108" stroke="#e5e7e9" stroke-width="1.2"/>')

# Panel headings.
add_panel_header(0, "A｜验证损失随 R/N 变化", "Val loss by epoch · lower is better")
add_panel_header(1, "B｜E3 训练—验证分歧", "Train–validation split · memorization at epoch boundary")
add_panel_header(2, "C｜E3 分叉 gap vs R/N", "gap = val loss − train loss")

# Subtle separators keep the three plots distinct without adding chart furniture.
for x in (922 * SCALE_X, 1822 * SCALE_X):
    svg.append(f'<line x1="{x}" y1="128" x2="{x}" y2="955" stroke="#e9ebed" stroke-width="1.2"/>')

# Compact legends sit above each chart, not over the data.
for x, label, color in [
    (COL_LEFTS[0] + 40, "E1 验证", C_E1),
    (COL_LEFTS[0] + 220, "E2 验证", C_E2),
    (COL_LEFTS[0] + 400, "E3 验证", C_E3),
]:
    add_legend_item(x, 222, label, color)
add_legend_item(COL_LEFTS[1] + 35, 211, "E3 训练", C_TRAIN, shape="square")
add_legend_item(COL_LEFTS[1] + 245, 211, "E3 验证", C_VAL)
add_legend_item(COL_LEFTS[1] + 35, 238, "无表训练", C_NOGRAM, dash="3 6")
add_legend_item(COL_LEFTS[1] + 245, 238, "无表验证", C_NOGRAM, dash="8 6")
add_legend_item(COL_LEFTS[2] + 315, 222, "E3 gap（val − train）", C_GAP, shape="diamond")

# ------------------------------ Panel A ------------------------------
A_MIN, A_MAX = 2.57, 2.87
add_plot_frame(
    0, A_MIN, A_MAX, [2.60, 2.65, 2.70, 2.75, 2.80, 2.85],
    "验证损失（epoch 边界）", lambda v: f"{v:.2f}",
)
for offset, color in ((0, C_E1), (2, C_E2), (4, C_E3)):
    add_series(0, [DATA[r][offset] for r in XS], A_MIN, A_MAX, color)

# ------------------------------ Panel B ------------------------------
B_MIN, B_MAX = 2.525, 2.720
b_left, b_right = add_plot_frame(
    1, B_MIN, B_MAX, [2.54, 2.58, 2.62, 2.66, 2.70],
    "E3 损失（epoch 边界）", lambda v: f"{v:.2f}", highlight_from=8.0,
)
# Faint E3 nogram references (train dotted, val dashed).
for ref, dash in ((DATA[0.0][5], "3 6"), (DATA[0.0][4], "8 6")):
    y = map_y(ref, B_MIN, B_MAX)
    svg.append(f'<line x1="{b_left:.2f}" y1="{y:.2f}" x2="{b_right:.2f}" y2="{y:.2f}" stroke="{C_NOGRAM}" stroke-width="2" stroke-dasharray="{dash}" opacity="0.8" clip-path="url(#plot-1)"/>')
add_series(1, [DATA[r][5] for r in XS], B_MIN, B_MAX, C_TRAIN, marker="square")
add_series(1, [DATA[r][4] for r in XS], B_MIN, B_MAX, C_VAL)

# Direct labels identify the small/large E3 changes from R/N=8 to 16.
x_mid = (map_x(xcoord(8.0), 1) + map_x(xcoord(16.0), 1)) / 2
val_mid = (map_y(DATA[8.0][4], B_MIN, B_MAX) + map_y(DATA[16.0][4], B_MIN, B_MAX)) / 2
train_mid = (map_y(DATA[8.0][5], B_MIN, B_MAX) + map_y(DATA[16.0][5], B_MIN, B_MAX)) / 2
add_text(x_mid, val_mid - 24, "Δval −0.0004", size=17, fill=C_VAL, anchor="middle", weight=700)
add_text(x_mid, train_mid + 34, "Δtrain −0.0500", size=17, fill=C_TRAIN, anchor="middle", weight=700)

# ------------------------------ Panel C ------------------------------
C_MIN, C_MAX = -0.05, 0.16
# Compute label values from decimal source strings so the exactly halfway
# R/N=8 gap (2.6618 - 2.6373 = 0.0245) rounds to the displayed +0.025.
gap_exact = [Decimal(str(DATA[r][4])) - Decimal(str(DATA[r][5])) for r in XS]
gaps = [float(gap) for gap in gap_exact]
add_plot_frame(
    2, C_MIN, C_MAX, [-0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14],
    "E3 gap（验证 − 训练）", lambda v: "0" if abs(v) < 1e-12 else f"{'+' if v > 0 else '−'}{abs(v):.2f}",
    highlight_from=8.0, zero_line=True,
)
add_series(2, gaps, C_MIN, C_MAX, C_GAP, marker="diamond", width=4.2)

# Point labels; alternate the two close R/N=2 and 2.6 values vertically to avoid overlap.
label_offsets = {
    0.0: (0, -19),
    2.0: (-3, -29),
    2.6: (4, 38),
    5.0: (0, -25),
    8.0: (-3, -25),
    16.0: (-8, -23),
    32.0: (-8, -23),
}
for ratio, gap_exact_value in zip(XS, gap_exact):
    dx, dy = label_offsets[ratio]
    x = map_x(xcoord(ratio), 2) + dx
    y = map_y(float(gap_exact_value), C_MIN, C_MAX) + dy
    label = gap_exact_value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    add_text(x, y, f"{label:+.3f}".replace("-", "−"), size=18, fill=C_GAP, anchor="middle", weight=600)

# Panel takeaways, separated from axes and manually wrapped to stay within each column.
# Use compact lines rather than one long sentence so the SVG remains legible in a 3-panel embed.
for index, lines, color in [
    (0, [
        "E1/E2 验证损失随 R 持续下降",
        "E3 收益在 R/N=8–16 达峰；32 仍 −0.016",
    ], "#31363b"),
    (1, [
        "8→16：Δtrain −0.0500，Δval −0.0004",
        "16→32：训练再降 0.0373，验证收益缩至 −0.016",
        "额外容量主要增加训练拟合；验证仍小幅受益",
    ], "#9d332d"),
    (2, [
        "E3 Δgap 单调上升：+0.019 → +0.172",
        "R/N=32 仍优于 nogram：Δval = −0.016",
    ], C_GAP),
]:
    y0 = 827 if len(lines) == 3 else 836
    for line_idx, line in enumerate(lines):
        add_text(COL_LEFTS[index] + COL_W / 2, y0 + line_idx * 25, line, size=16.5, fill=color, anchor="middle", weight=600)

# Figure provenance, endpoint semantics, and metric definitions.
svg.append(f'<line x1="{MARGIN_X}" y1="968" x2="{W - MARGIN_X}" y2="968" stroke="#e5e7e9" stroke-width="1.2"/>')
add_text(
    W / 2, 1000,
    "数据来源：docs/experiment-log.md §122(a) 七个端点（seed 42）；基线 v26dn_50m_nogram_r400k_d1b_tls1_s42_e3_fd_fixed + exact arms v26dn_50m_exact_r{100k,128k,256k,400k,800k,1600k}_d1b_tls1_s42_e3_fd_fixed。",
    size=15.5, fill="#555d65", anchor="middle",
)
add_text(
    W / 2, 1028,
    "读数 = 每个 epoch 的最后一个 logged step（6780 / 13560 / 20343）；R/N = 每阶表行数 ÷ (backbone 参数 / 1000)；gap = val − train。Lower loss is better.",
    size=15.5, fill="#555d65", anchor="middle",
)
add_text(
    W / 2, 1054,
    "跨机拼接：r100k / r400k / r800k 在 360-2，r128k / r256k / r1600k 在 ophis（同 launcher、同 npz 口径；跨机 val 帧差 ≤0.008）。",
    size=15.5, fill="#555d65", anchor="middle",
)
svg.append('</svg>')
SVG_PATH.write_text("\n".join(svg) + "\n", encoding="utf-8")


def find_chrome() -> str | None:
    candidates = [
        os.environ.get("CHROME_BIN"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def render_png_with_chrome() -> None:
    chrome = find_chrome()
    if chrome is None:
        raise SystemExit(
            f"SVG written to {SVG_PATH}, but Chrome/Chromium was not found; "
            "PNG was not rendered. Set CHROME_BIN and rerun."
        )
    with tempfile.TemporaryDirectory(prefix="r-sweep-50m-render-") as tmp:
        temp = Path(tmp)
        page = temp / "render.html"
        profile = temp / "chrome-profile"
        image_uri = html.escape(SVG_PATH.as_uri(), quote=True)
        page.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            f"html,body{{margin:0;padding:0;width:{W}px;height:{H}px;overflow:hidden;background:#fff}}"
            f"img{{display:block;width:{W}px;height:{H}px}}"
            f"</style></head><body><img src=\"{image_uri}\" width=\"{W}\" height=\"{H}\" alt=\"\"></body></html>",
            encoding="utf-8",
        )
        command = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--force-device-scale-factor=1",
            f"--window-size={W},{H}",
            f"--user-data-dir={profile}",
            f"--screenshot={PNG_PATH}",
            page.as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                "Chrome headless failed to render the PNG.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
    if not PNG_PATH.is_file():
        raise RuntimeError(f"Chrome returned without creating {PNG_PATH}")
    header = PNG_PATH.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Chrome output is not a valid PNG: {PNG_PATH}")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (W, H):
        raise RuntimeError(
            f"Unexpected Chrome screenshot dimensions {width}x{height}; expected {W}x{H}."
        )


render_png_with_chrome()
print(f"SVG: {SVG_PATH}")
print(f"PNG: {PNG_PATH}")
