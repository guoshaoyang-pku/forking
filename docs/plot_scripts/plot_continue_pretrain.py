#!/usr/bin/env python3
"""Generate evidence-traceable SVG plots for continued-pretraining runs."""
from __future__ import annotations
import csv, json, html
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/appendices/continue_pretrain_20260919"
OUT = ROOT / "docs/figs/continue_pretrain_20260919"
RUNS = {
    "ID": {"trainable": "id_trainable_train_log.jsonl", "frozen": "id_frozen_train_log.jsonl"},
    "Magicoder OOD": {"trainable": "magicoder_trainable_train_log.jsonl", "frozen": "magicoder_frozen_train_log.jsonl"},
    "OpenR1 Math OOD": {"trainable": "openr1_trainable_train_log.jsonl", "frozen": "openr1_frozen_train_log.jsonl"},
}
COLORS = {"trainable": "#b42318", "frozen": "#1769aa"}
LABELS = {"trainable": "table trainable", "frozen": "table frozen"}
def load(path):
    rows = sorted([json.loads(x) for x in path.read_text().splitlines() if x.strip()], key=lambda x: x["step"])
    if len(rows) != 5 or [x["epoch"] for x in rows] != [1, 2, 3, 4, 5]:
        raise ValueError(f"expected five epoch boundaries in {path}")
    return rows
def esc(x): return html.escape(str(x))
def line(points, color, width=2, dash=False):
    da = ' stroke-dasharray="6,5"' if dash else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{da} points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in points)}"/>'
def circle(x, y, color): return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
def txt(x, y, s, size=13, anchor="middle", weight="normal", fill="#243447"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(s)}</text>'
def panel(svg, x, y, w, h, title, series, kind):
    left, right, top, bottom = x + 58, x + w - 18, y + 42, y + h - 46
    vals = []
    for d in series.values():
        vals.extend(d["train"] if kind == "loss" else d["gap"])
        if kind == "loss": vals.extend(d["val"] + d["fixed"])
    lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.10 or 1; lo -= pad; hi += pad
    if kind == "gap": lo, hi = min(lo, -0.15), max(hi, 0.15)
    def px(i): return left + (i - 1) * (right - left) / 4
    def py(v): return bottom - (v - lo) * (bottom - top) / (hi - lo)
    svg.append(txt(x + w/2, y + 20, title, 15, weight="bold"))
    for i in range(6):
        v = lo + (hi - lo) * i / 5; yy = py(v)
        svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#d9e2ec"/>')
        svg.append(txt(left - 8, yy + 4, f"{v:.2f}", 10, "end", fill="#52606d"))
    for i in range(1, 6):
        xx = px(i); svg.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" stroke="#eef2f6"/>'); svg.append(txt(xx, bottom + 18, str(i), 10))
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#52606d"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#52606d"/>')
    if kind == "gap" and lo < 0 < hi:
        yy = py(0); svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#68737d" stroke-dasharray="3,3"/>')
    for arm, d in series.items():
        c = COLORS[arm]
        if kind == "gap":
            arrs = [("gap", d["gap"], False)]
        else:
            arrs = [("train", d["train"], False), ("val", d["val"], False), ("fixed", d["fixed"], True)]
        for key, arr, dashed in arrs:
            pts = [(px(i + 1), py(v)) for i, v in enumerate(arr)]; svg.append(line(pts, c, 2.4 if not dashed else 1.4, dashed))
            if not dashed:
                for xx, yy in pts: svg.append(circle(xx, yy, c))
    svg.append(txt((left + right)/2, bottom + 37, "epoch", 11)); svg.append(txt(x + 14, (top + bottom)/2, "gap" if kind == "gap" else "loss (nats)", 11, fill="#52606d"))
def main():
    OUT.mkdir(parents=True, exist_ok=True); W, H = 1500, 850
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">', '<rect width="100%" height="100%" fill="white"/>']
    svg.append(txt(W/2, 31, "Continued pretraining after a low-gap 20x checkpoint", 23, weight="bold"))
    svg.append(txt(W/2, 57, "Solid: observed train/val. Dashed: validation if the epoch-1 gap stayed fixed. Seed 42; 1665 updates; gap = fixed val - online train.", 12))
    effects = []
    for j, (domain, arms) in enumerate(RUNS.items()):
        series = {}
        for arm, fn in arms.items():
            rows = load(SOURCE / fn); train = [r["train_loss"] for r in rows]; val = [r["val_loss"] for r in rows]; gap = [r["gap"] for r in rows]; fixed = [v + gap[0] for v in train]
            series[arm] = {"train": train, "val": val, "gap": gap, "fixed": fixed}
            effects.append({"domain": domain, "arm": arm, "initial_gap": gap[0], "final_gap": gap[-1], "gap_increase": gap[-1] - gap[0], "train_delta": train[-1] - train[0], "val_delta": val[-1] - val[0], "val_delta_if_gap_fixed": fixed[-1] - fixed[0], "val_improvement_erased_by_gap": (val[-1] - val[0]) - (fixed[-1] - fixed[0])})
        x = 20 + j * 495; panel(svg, x, 78, 465, 330, domain, series, "loss"); panel(svg, x, 455, 465, 330, domain, series, "gap")
    svg += [line([(1120,92),(1160,92)], COLORS["trainable"], 2.4), txt(1170,97,"table trainable",12,"start"), line([(1120,120),(1160,120)], COLORS["frozen"], 2.4), txt(1170,125,"table frozen",12,"start"), line([(1120,148),(1160,148)], "#333", 2.2), txt(1170,153,"observed train / val",12,"start"), line([(1120,176),(1160,176)], "#333", 1.4, True), txt(1170,181,"val if epoch-1 gap fixed",12,"start")]
    svg.append("</svg>"); (OUT / "continue_pretrain_train_val_gap.svg").write_text("\n".join(svg))
    with (OUT / "val_effects.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(effects[0])); w.writeheader(); w.writerows(effects)
    print(OUT / "continue_pretrain_train_val_gap.svg"); print(OUT / "val_effects.csv")
if __name__ == "__main__": main()
