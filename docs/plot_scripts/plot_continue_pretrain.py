#!/usr/bin/env python3
"""Plot dense freq=10 continued-pretraining loss/gap observations."""
from __future__ import annotations
import csv, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "appendices" / "continue_pretrain_20260920_freq10"
OUT = ROOT / "docs" / "figs" / "continue_pretrain_20260920_freq10"
RUNS = {
    "ID": {"trainable": "sft_id_tri_e5_freq10_trainable_s42", "frozen": "sft_id_tri_e5_freq10_frozen_s42", "file": "fig_continue_pretrain_id_freq10.svg"},
    "Magicoder OOD": {"trainable": "sft_ood_magicoder_e5_freq10_trainable_s42", "frozen": "sft_ood_magicoder_e5_freq10_frozen_s42", "file": "fig_continue_pretrain_magicoder_freq10.svg"},
    "OpenR1 Math OOD": {"trainable": "sft_ood_openr1_10shards_e5_freq10_trainable_s42", "frozen": "sft_ood_openr1_10shards_e5_freq10_frozen_s42", "file": "fig_continue_pretrain_openr1_freq10.svg"},
}
COLORS = {"trainable": "#c4493d", "frozen": "#2d6f9f"}
LABELS = {"trainable": "table trainable", "frozen": "table frozen"}

def load(run_dir):
    path = SOURCE / run_dir / "train_log.jsonl"
    rows = sorted([json.loads(line) for line in path.read_text().splitlines() if line.strip()], key=lambda row: row["step"])
    expected = list(range(10, 1661, 10)) + [1665]
    if [row["step"] for row in rows] != expected:
        raise ValueError(f"expected dense freq=10 steps in {path}")
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
    max_step = max(max(data["steps"]) for data in series.values())
    def px(step): return left + (step / max_step) * (right - left)
    def py(value): return bottom - (value - lo) * (bottom - top) / (hi - lo)
    svg.append(text(x + width / 2, y + 21, title, 16, weight="bold"))
    for value in ticks:
        yy = py(value)
        svg.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="#dfe6ed"/>')
        svg.append(text(left - 10, yy + 4, f"{value:.2f}", 10, "end", fill="#52606d"))
    for epoch in range(1, 6):
        xx = px(333 * epoch)
        svg.append(f'<line x1="{xx:.1f}" y1="{top:.1f}" x2="{xx:.1f}" y2="{bottom:.1f}" stroke="#eef2f6"/>')
        svg.append(text(xx, bottom + 19, f"epoch {epoch}", 10))
    svg.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" stroke="#52606d"/>')
    svg.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" stroke="#52606d"/>')
    if kind == "gap" and lo < 0 < hi:
        yy = py(0); svg.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="#68737d" stroke-dasharray="3,3"/>')
    for arm, data in series.items():
        color = COLORS[arm]
        if kind == "loss":
            fixed_points = [(px(step), py(value)) for step, value in zip(data["steps"], data["fixed"])]
            svg.append(polyline(fixed_points, color, 1.2, "2,4", 0.50))
            for key, opacity, width_ in (("train", 1.0, 2.4), ("val", 0.60, 2.0)):
                points = [(px(step), py(value)) for step, value in zip(data["steps"], data[key])]
                svg.append(polyline(points, color, width_, opacity=opacity))
                for xx, yy in points: svg.append(circle(xx, yy, color, opacity))
        else:
            points = [(px(step), py(value)) for step, value in zip(data["steps"], data["gap"])]
            svg.append(polyline(points, color, 2.4, "7,5"))
            for xx, yy in points: svg.append(circle(xx, yy, color))
    svg.append(text((left + right) / 2, bottom + 42, "optimizer step", 11, fill="#52606d"))
    svg.append(text(x + 18, (top + bottom) / 2, "gap = val − train" if kind == "gap" else "loss (nats)", 11, fill="#52606d"))

def make_figure(domain, config):
    series, effects = {}, []
    for arm in ("trainable", "frozen"):
        rows = load(config[arm])
        steps = [row["step"] for row in rows]; train = [row["train_loss"] for row in rows]; val = [row["val_loss"] for row in rows]; gap = [row["gap"] for row in rows]
        fixed = [train_value + gap[0] for train_value in train]
        series[arm] = {"steps": steps, "train": train, "val": val, "gap": gap, "fixed": fixed}
        effects.append({"domain": domain, "arm": arm, "initial_gap": gap[0], "final_gap": gap[-1], "gap_increase": gap[-1] - gap[0], "train_delta": train[-1] - train[0], "val_delta": val[-1] - val[0], "val_delta_if_gap_fixed": fixed[-1] - fixed[0], "val_improvement_erased_by_gap": (val[-1] - val[0]) - (fixed[-1] - fixed[0])})
    width, height = 1120, 840
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', text(width / 2, 31, f"Continued pretraining · {domain}", 23, weight="bold"), text(width / 2, 56, "Dense observations every 10 steps + epoch-5 boundary at 1665 · seed 42 · gap = fixed val − online train", 12), text(width / 2, 75, "Solid: observed train/val · dotted: val if the epoch-1 gap stayed fixed · lower panel: gap", 12, fill="#52606d")]
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
