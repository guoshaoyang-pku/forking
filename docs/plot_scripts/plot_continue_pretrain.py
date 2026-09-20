#!/usr/bin/env python3
"""Plot dense continued-pretraining curves, split by table trainability."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "appendices" / "continue_pretrain_20260920_freq10"
OUT = ROOT / "docs" / "figs" / "continue_pretrain_20260920_freq10"
DATASETS = (
    ("ID", "sft_id_tri_e5_freq10_{arm}_s42", "#7654a8"),
    ("Magicoder OOD", "sft_ood_magicoder_e5_freq10_{arm}_s42", "#d97732"),
    ("OpenR1-10shard OOD", "sft_ood_openr1_10shards_e5_freq10_{arm}_s42", "#187c75"),
)
EXPECTED_STEPS = list(range(10, 1661, 10)) + [1665]


def load(run_dir: str) -> dict[str, list]:
    path = SOURCE / run_dir / "train_log.jsonl"
    rows = sorted((json.loads(line) for line in path.read_text().splitlines() if line.strip()), key=lambda row: row["step"])
    steps = [row["step"] for row in rows]
    if steps != EXPECTED_STEPS:
        raise ValueError(f"expected dense steps in {path}, got {steps[:3]}...{steps[-3:]}")
    return {key: [row[key] for row in rows] for key in ("step", "train_loss", "val_loss", "gap")}


def esc(value: object) -> str:
    return html.escape(str(value))


def txt(x: float, y: float, value: object, size: int = 13, anchor: str = "middle", weight: str = "normal", fill: str = "#243447") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(value)}</text>')


def line(points: list[tuple[float, float]], color: str, width: float = 2.2, dash: str | None = None, opacity: float = 1.0) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dash_attr} points="{pts}"/>'


def dot(x: float, y: float, color: str, radius: float = 2.8) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}"/>'


def ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    return [lo + i * (hi - lo) / count for i in range(count + 1)]


def draw_panel(svg: list[str], x: float, y: float, width: float, height: float, title: str, data: dict[str, list], color: str, kind: str, x_labels: bool) -> None:
    left, right = x + 62, x + width - 22
    top, bottom = y + 39, y + height - (38 if x_labels else 20)
    values = data["train_loss"] + data["val_loss"] if kind == "loss" else data["gap"]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.12 or 0.2
    lo, hi = lo - pad, hi + pad
    if kind == "gap":
        lo, hi = min(lo, -0.08), max(hi, 0.08)
    def px(step: int) -> float:
        return left + step / 1665 * (right - left)
    def py(value: float) -> float:
        return bottom - (value - lo) * (bottom - top) / (hi - lo)
    svg.append(txt(x + width / 2, y + 20, title, 15, weight="bold"))
    for value in ticks(lo, hi):
        yy = py(value)
        svg.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="#dfe6ed"/>')
        svg.append(txt(left - 8, yy + 4, f"{value:.2f}", 9, "end", fill="#52606d"))
    for epoch in range(1, 6):
        xx = px(333 * epoch)
        svg.append(f'<line x1="{xx:.1f}" y1="{top:.1f}" x2="{xx:.1f}" y2="{bottom:.1f}" stroke="#eef2f6"/>')
        if x_labels:
            svg.append(txt(xx, bottom + 16, f"e{epoch}", 9, fill="#52606d"))
    svg.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" stroke="#52606d"/>')
    svg.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" stroke="#52606d"/>')
    if kind == "gap" and lo < 0 < hi:
        yy = py(0)
        svg.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="#68737d" stroke-dasharray="3,3"/>')
    sample_rows = (0, 32, 65, 98, 131, 166)
    if kind == "loss":
        for key, dash, opacity, width_ in (("train_loss", None, 1.0, 2.3), ("val_loss", "5,3", 0.62, 2.0)):
            points = [(px(s), py(v)) for s, v in zip(data["step"], data[key])]
            svg.append(line(points, color, width_, dash, opacity))
            for row in sample_rows:
                svg.append(dot(*points[row], color, 2.5))
    else:
        points = [(px(s), py(v)) for s, v in zip(data["step"], data["gap"])]
        svg.append(line(points, color, 2.5, "7,4"))
        for row in sample_rows:
            svg.append(dot(*points[row], color, 2.8))
        svg.append(txt(right - 2, max(top + 13, py(data["gap"][-1]) - 7), f"final {data['gap'][-1]:+.3f}", 10, "end", "bold", color))
    if x_labels:
        svg.append(txt((left + right) / 2, bottom + 31, "optimizer step", 10, fill="#52606d"))
    svg.append(txt(x + 13, (top + bottom) / 2, "gap (nats)" if kind == "gap" else "loss", 10, fill="#52606d"))


def make_figure(arm: str) -> tuple[Path, list[dict]]:
    width, height = 1420, 890
    margin_x, panel_w, panel_h, gap_x = 30, 445, 300, 14
    xs = [margin_x + i * (panel_w + gap_x) for i in range(3)]
    loaded: list[tuple[str, str, str, dict[str, list]]] = []
    effects = []
    for label, template, color in DATASETS:
        run_id = template.format(arm=arm)
        data = load(run_id)
        loaded.append((label, run_id, color, data))
        effects.append({"dataset": label, "arm": arm, "run_id": run_id, "seed": 42, "step": data["step"][-1],
                        "initial_gap": data["gap"][0], "final_gap": data["gap"][-1],
                        "gap_increase_from_step10": data["gap"][-1] - data["gap"][0],
                        "gap_at_epoch1": data["gap"][32], "gap_increase_from_epoch1": data["gap"][-1] - data["gap"][32],
                        "train_delta_from_epoch1": data["train_loss"][-1] - data["train_loss"][32],
                        "val_delta_from_epoch1": data["val_loss"][-1] - data["val_loss"][32]})
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           txt(width / 2, 31, f"Continued pretraining · table {arm}", 24, weight="bold"),
           txt(width / 2, 56, "ID and OOD datasets · dense observations every 10 steps + epoch-5 boundary at 1665 · seed 42", 12),
           txt(width / 2, 76, "Top: train (solid) and fixed validation (dashed). Bottom: online gap = fixed validation − current-batch train; each dataset has its own y-scale.", 12, fill="#52606d")]
    for i, (label, run_id, color, data) in enumerate(loaded):
        draw_panel(svg, xs[i], 95, panel_w, panel_h, label, data, color, "loss", False)
        draw_panel(svg, xs[i], 430, panel_w, panel_h, f"{label} · gap", data, color, "gap", True)
        svg.append(txt(xs[i] + panel_w / 2, 790, f"run: {run_id}", 9, fill="#52606d"))
    legend_y = 826
    for start, dash, label, opacity in ((55, None, "train", 1.0), (170, "5,3", "fixed val", 0.62), (300, "7,4", "gap", 1.0)):
        svg.append(line([(start, legend_y), (start + 39, legend_y)], "#374151", 2.4, dash, opacity))
        svg.append(txt(start + 48, legend_y + 4, label, 11, "start"))
    svg.append(txt(505, legend_y + 4, "all curves use recorded JSONL values; final gap labels are nats", 11, "start", fill="#52606d"))
    svg.append('</svg>')
    output = OUT / f"fig_continue_pretrain_{arm}_freq10.svg"
    output.write_text("\n".join(svg))
    return output, effects


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    effects = []
    for arm in ("trainable", "frozen"):
        output, rows = make_figure(arm)
        effects.extend(rows)
        print(output)
    with (OUT / "val_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(effects)
    print(OUT / "val_effects.csv")


if __name__ == "__main__":
    main()
