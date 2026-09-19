#!/usr/bin/env python3
"""Plot sparse epoch-boundary continued-pretraining loss/gap observations."""
from __future__ import annotations
import csv, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "appendices" / "continue_pretrain_20260919"
OUT = ROOT / "docs" / "figs" / "continue_pretrain_20260919"
RUNS = {
    "ID": {"trainable": "id_trainable_train_log.jsonl", "frozen": "id_frozen_train_log.jsonl", "file": "fig_continue_pretrain_id.svg"},
    "Magicoder OOD": {"trainable": "magicoder_trainable_train_log.jsonl", "frozen": "magicoder_frozen_train_log.jsonl", "file": "fig_continue_pretrain_magicoder.svg"},
    "OpenR1 Math OOD": {"trainable": "openr1_trainable_train_log.jsonl", "frozen": "openr1_frozen_train_log.jsonl", "file": "fig_continue_pretrain_openr1.svg"},
}
COLORS = {"trainable": "#c4493d", "frozen": "#2d6f9f"}
LABELS = {"trainable": "table trainable", "frozen": "table frozen"}

def load(path):
    rows = sorted([json.loads(line) for line in path.read_text().splitlines() if line.strip()], key=lambda row: row["step"])
    if len(rows) != 5 or [row["epoch"] for row in rows] != [1, 2, 3, 4, 5]:
        raise ValueError(f"expected five epoch boundaries in {path}")
    return rows

def esc(value): return html.escape(str(value))
def text(x, y, value, size=13, anchor="middle", weight="normal", fill="#243447"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(value)}</text>')
def polyline(points, color, width=2.2, dash=None, opacity=1.0):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dash_attr} points="{pts}"/>'
def circle(x, y, color, opacity=0.95):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" opacity="{opacity}"/>'
def nice_ticks(lo, hi, count=5):
    step = (hi - lo) / count
    return [lo + i * step for i in range(count + 1)]

def draw_panel(svg, x, y, width, height, title, series, kind):
    left, right = x + 76, x + width - 30
    top, bottom = y + 42, y + height - 54
    values = []
    for arm in series.values():
        values.extend(arm["train"] if kind == "loss" else arm["gap"])
        if kind == "loss": values.extend(arm["val"]); values.extend(arm["fixed"])
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.12 or 0.2
    lo, hi = lo - pad, hi + pad
    if kind == "gap": lo, hi = min(lo, -0.10), max(hi, 0.10)
    ticks = nice_ticks(lo, hi)
    def px(epoch): return left + (epoch - 1) * (right - left) / 4
    def py(value): return bottom - (value - lo) * (bottom - top) / (hi - lo)
    svg.append(text(x + width / 2, y + 21, title, 16, weight="bold"))
    for value in ticks:
        yy = py(value)
        svg.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="#dfe6ed"/>')
        svg.append(text(left - 10, yy + 4, f"{value:.2f}", 10, "end", fill="#52606d"))
    for epoch in range(1, 6):
        xx = px(epoch)
        svg.append(f'<line x1="{xx:.1f}" y1="{top:.1f}" x2="{xx:.1f}" y2="{bottom:.1f}" stroke="#eef2f6"/>')
        svg.append(text(xx, bottom + 19, f"epoch {epoch}", 10))
    svg.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" stroke="#52606d"/>')
    svg.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" stroke="#52606d"/>')
    if kind == "gap" and lo < 0 < hi:
        yy = py(0); svg.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="#68737d" stroke-dasharray="3,3"/>')
    for arm, data in series.items():
        color = COLORS[arm]
        if kind == "loss":
            fixed_points = [(px(epoch), py(value)) for epoch, value in enumerate(data["fixed"], 1)]
            svg.append(polyline(fixed_points, color, 1.2, "2,4", 0.50))
            for key, opacity, width_ in (("train", 1.0, 2.4), ("val", 0.60, 2.0)):
                points = [(px(epoch), py(value)) for epoch, value in enumerate(data[key], 1)]
                svg.append(polyline(points, color, width_, opacity=opacity))
                for xx, yy in points: svg.append(circle(xx, yy, color, opacity))
        else:
            points = [(px(epoch), py(value)) for epoch, value in enumerate(data["gap"], 1)]
            svg.append(polyline(points, color, 2.4, "7,5"))
            for xx, yy in points: svg.append(circle(xx, yy, color))
    svg.append(text((left + right) / 2, bottom + 42, "replay epoch", 11, fill="#52606d"))
    svg.append(text(x + 18, (top + bottom) / 2, "gap = val − train" if kind == "gap" else "loss (nats)", 11, fill="#52606d"))

def make_figure(domain, config):
    series, effects = {}, []
    for arm in ("trainable", "frozen"):
        rows = load(SOURCE / config[arm])
        train = [row["train_loss"] for row in rows]; val = [row["val_loss"] for row in rows]; gap = [row["gap"] for row in rows]
        fixed = [train_value + gap[0] for train_value in train]
        series[arm] = {"train": train, "val": val, "gap": gap, "fixed": fixed}
        effects.append({"domain": domain, "arm": arm, "initial_gap": gap[0], "final_gap": gap[-1], "gap_increase": gap[-1] - gap[0], "train_delta": train[-1] - train[0], "val_delta": val[-1] - val[0], "val_delta_if_gap_fixed": fixed[-1] - fixed[0], "val_improvement_erased_by_gap": (val[-1] - val[0]) - (fixed[-1] - fixed[0])})
    width, height = 1120, 840
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', text(width / 2, 31, f"Continued pretraining · {domain}", 23, weight="bold"), text(width / 2, 56, "Epoch-boundary observations (steps 333, 666, 999, 1332, 1665) · seed 42 · gap = fixed val − online train", 12), text(width / 2, 75, "Solid: observed train/val · dotted: val if the epoch-1 gap stayed fixed · lower panel: gap", 12, fill="#52606d")]
    draw_panel(svg, 24, 92, 1072, 320, "train / validation loss", series, "loss")
    draw_panel(svg, 24, 465, 1072, 300, "validation gap", series, "gap")
    legend_y = 790
    svg.append(polyline([(90, legend_y), (125, legend_y)], COLORS["trainable"], 2.4)); svg.append(text(135, legend_y + 4, LABELS["trainable"], 11, "start"))
    svg.append(polyline([(285, legend_y), (320, legend_y)], COLORS["frozen"], 2.4)); svg.append(text(330, legend_y + 4, LABELS["frozen"], 11, "start"))
    svg.append(polyline([(490, legend_y), (525, legend_y)], "#374151", 2.2)); svg.append(text(535, legend_y + 4, "train: solid · val: translucent · gap: dashed", 11, "start"))
    svg.append(polyline([(825, legend_y), (860, legend_y)], "#374151", 1.2, "2,4", 0.50)); svg.append(text(870, legend_y + 4, "fixed initial gap", 11, "start"))
    svg.append('</svg>')
    output = OUT / config["file"]; output.write_text("\n".join(svg)); return output, effects

def main():
    OUT.mkdir(parents=True, exist_ok=True); all_effects = []
    for domain, config in RUNS.items():
        output, effects = make_figure(domain, config); all_effects.extend(effects); print(output)
    with (OUT / "val_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_effects[0])); writer.writeheader(); writer.writerows(all_effects)
    print(OUT / "val_effects.csv")

if __name__ == "__main__": main()
