#!/usr/bin/env python3
"""Generate a dependency-free SVG for the §60 boundary results."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/appendices/multilayer_memory_20260920_3602/boundary_results.csv"
OUT = ROOT / "docs/figs/multilayer_memory_20260920_3602"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"v": "#D55E00", "y": "#0072B2", "engram": "#009E73"}
LABELS = {"v": "v injection", "y": "y injection", "engram": "Engram-like gated proxy"}
ROWS = list(csv.DictReader(SRC.open()))
STEPS = [6740, 13480, 20220]
ROUTES = ["v", "y", "engram"]
W, H = 1180, 430

def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def txt(x, y, s, size=12, anchor="middle", weight="normal", fill="#243447"):
    return f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>'

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">', '<rect width="100%" height="100%" fill="white"/>', txt(W/2, 26, "Matched 20× replay · multi-layer memory · seed 42", 20, weight="bold"), txt(W/2, 47, "360-2 · 0-based injection blocks [1,4] · gap = fixed val − online train · three boundary evaluations", 11, fill="#52606d")]
for pi, (metric, title) in enumerate([("train_loss", "online train loss"), ("val_loss", "fixed validation loss"), ("gap", "gap = val − train")]):
    x0, x1, y0, y1 = 30 + pi * 385, 380 + pi * 385, 85, 335
    vals = [float(r[metric]) for r in ROWS]
    lo, hi = min(vals), max(vals)
    if metric == "gap": lo, hi = min(lo, -0.05), max(hi, 0.05)
    pad = (hi - lo) * 0.14 or 0.2
    lo, hi = lo - pad, hi + pad
    def X(i): return x0 + 35 + i * (x1 - x0 - 55) / 2
    def Y(v): return y1 - (v - lo) / (hi - lo) * (y1 - y0)
    parts.append(txt((x0+x1)/2, 70, title, 15, weight="bold"))
    for j in range(5):
        v = lo + j * (hi - lo) / 4; yy = Y(v)
        parts += [f'<line x1="{x0+35:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#e2e8f0"/>', txt(x0+28, yy+4, f"{v:.2f}", 9, "end", fill="#52606d")]
    for i, (lab, step) in enumerate(zip(["e1", "e2", "e3"], STEPS)):
        xx = X(i); parts += [f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#f0f3f6"/>', txt(xx, y1+20, f"{lab} ({step})", 10)]
    parts += [f'<line x1="{x0+35}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#52606d"/>', f'<line x1="{x0+35}" y1="{y0}" x2="{x0+35}" y2="{y1}" stroke="#52606d"/>']
    if metric == "gap" and lo < 0 < hi:
        yy = Y(0); parts.append(f'<line x1="{x0+35}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#555" stroke-dasharray="4,4"/>')
    for route in ROUTES:
        rr = sorted([r for r in ROWS if r["route"] == route], key=lambda r: int(r["step"]))
        pts = " ".join(f"{X(i):.1f},{Y(float(r[metric])):.1f}" for i, r in enumerate(rr))
        parts.append(f'<polyline fill="none" stroke="{COLORS[route]}" stroke-width="2.5" points="{pts}"/>')
        for i, r in enumerate(rr): parts.append(f'<circle cx="{X(i):.1f}" cy="{Y(float(r[metric])):.1f}" r="4" fill="{COLORS[route]}"/>')
for i, route in enumerate(ROUTES):
    xx = 280 + i * 220; parts += [f'<line x1="{xx}" y1="390" x2="{xx+30}" y2="390" stroke="{COLORS[route]}" stroke-width="2.5"/>', txt(xx+40, 394, LABELS[route], 11, "start")]
parts.append("</svg>")
(OUT / "fig_multilayer_memory_boundary.svg").write_text("\n".join(parts))
print(OUT / "fig_multilayer_memory_boundary.svg")
