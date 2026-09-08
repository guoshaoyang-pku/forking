"""§3-end figure: multi-epoch staircase + gap vs epoch number (128× standard).

Left panel: dense online gap over 2000 steps (~6 epochs of shard 1, 337
batches/epoch) for the input main arm and the no-gram control; epoch
boundaries marked.  Points are raw online records (val every 10 steps);
thin lines are 3-point visual connectors.

Right panel: gap at each epoch boundary of the 20-epoch L4 long replay
(trigram-only / both-tables / no-gram), with a per-epoch increment strip
beneath it: increments peak at e2–e3 then settle to a near-plateau, still
positive at e20 — approximately stable step heights, i.e. each additional
epoch adds a roughly constant amount of gap.

Sources (all _fixed runs, seed 42, table LR 128×, warmup_constant(100)):
- docs/appendices/s1_scaling_three_axis/nglab1x_input_v5_128x_freq10_fd_train_log.jsonl
  (run nglab1x_input_v5_128x_freq10_fd_fixed, 2000 steps, val every 10)
- docs/appendices/s1_scaling_three_axis/nglab1x_nogram_v5_128x_freq10_fd_train_log.jsonl
- docs/appendices/s1_scaling_three_axis/s1_epoch_long_replay_points.csv
  (runs s1v5_128_ep_tri_1xL4_20ep / s1v5_128_ep1xL4_20ep_both /
   s1v5_128_ep1xL4_20ep_nogram, 6740 steps = 20 epochs)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
APPX = ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
FIGS = ROOT / "docs" / "figs" / "main"

EPOCH_STEPS = 337  # shard 1 = 337 device batches = 1 epoch (L4 convention)


def load_log(name):
    rows = []
    for line in (APPX / name).open():
        rows.append(json.loads(line))
    return rows


def smooth3(values):
    out = []
    for i in range(len(values)):
        lo = max(0, i - 1)
        hi = min(len(values), i + 2)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def main():
    input_rows = load_log("nglab1x_input_v5_128x_freq10_fd_train_log.jsonl")
    nogram_rows = load_log("nglab1x_nogram_v5_128x_freq10_fd_train_log.jsonl")

    boundary = {}
    for row in csv.DictReader((APPX / "s1_epoch_long_replay_points.csv").open()):
        if "20ep" in row["run_id"]:
            boundary.setdefault(row["arm"], {})[int(row["epoch"])] = float(row["gap"])

    figure = plt.figure(figsize=(13.2, 5.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.10,
                               wspace=0.24)
    ax_curve = figure.add_subplot(grid[0, 0])
    ax_gap = figure.add_subplot(grid[0, 1])
    ax_inc = figure.add_subplot(grid[1, 1], sharex=ax_gap)

    # ---- left: dense staircase over ~6 epochs ----
    for rows, label, color in (
        (input_rows, "input (bigram+trigram)", "#2d6f9f"),
        (nogram_rows, "no-gram control", "#686d73"),
    ):
        steps = [r["step"] for r in rows]
        gaps = [r["gap"] for r in rows]
        ax_curve.scatter(steps, gaps, color=color, s=9, alpha=0.45, zorder=3)
        ax_curve.plot(steps, smooth3(gaps), color=color, linewidth=1.0,
                      label=label, zorder=4)
    for e in range(1, 7):
        ax_curve.axvline(e * EPOCH_STEPS, color="#b45309", linewidth=0.7,
                         linestyle="--", alpha=0.6, zorder=1)
    ax_curve.axhline(0, color="#686d73", linewidth=0.6, linestyle=":")
    ax_curve.set_xlabel("optimizer step (337 batches = 1 epoch of shard 1)")
    ax_curve.set_ylabel("online gap = fixed val − online train")
    ax_curve.set_title("multi-epoch replay: gap steps up at every epoch boundary")
    ax_curve.legend(fontsize=8, frameon=False, loc="upper left")
    ax_curve.grid(alpha=0.22)
    ax_curve.set_ylim(bottom=-0.4)
    ymax = ax_curve.get_ylim()[1]
    for e in range(1, 7):
        ax_curve.text(e * EPOCH_STEPS + 10, ymax * 0.99, f"e{e + 1}",
                      fontsize=7.5, color="#b45309", va="top")

    # ---- right top: gap at epoch boundary vs epoch number ----
    arms = (
        ("trigram-only", "trigram-only", "#2f8f83"),
        ("both-tables", "bigram+trigram", "#67439b"),
        ("nogram", "no-gram control", "#686d73"),
    )
    series = {}
    for arm, label, color in arms:
        pts = sorted(boundary.get(arm, {}).items())
        if len(pts) < 20:
            raise SystemExit(f"missing 20ep boundary data for {arm}: {len(pts)}")
        series[label] = (color, pts)
        x = [e for e, _ in pts]
        y = [g for _, g in pts]
        ax_gap.scatter(x, y, color=color, s=20, zorder=3)
        ax_gap.plot(x, y, color=color, linewidth=0.9, alpha=0.85, label=label)

    tri_x = [e for e, _ in series["trigram-only"][1]]
    tri_y = [g for _, g in series["trigram-only"][1]]
    ref = [(e, g) for e, g in zip(tri_x, tri_y) if e >= 2]
    slope = (ref[-1][1] - ref[0][1]) / (ref[-1][0] - ref[0][0])
    ax_gap.plot([ref[0][0], ref[-1][0]],
                [ref[0][1], ref[0][1] + slope * (ref[-1][0] - ref[0][0])],
                color="#9ca3af", linestyle="--", linewidth=0.8,
                label=f"secant ≈ {slope:.2f}/epoch (e2–e20, trigram-only)")
    ax_gap.axhline(0, color="#686d73", linewidth=0.6, linestyle=":")
    ax_gap.set_ylabel("gap at epoch boundary")
    ax_gap.set_title("gap vs epoch number · 20-epoch replay")
    ax_gap.set_xticks(range(1, 21, 2))
    ax_gap.legend(fontsize=8, frameon=False, loc="upper left")
    ax_gap.grid(alpha=0.22)
    plt.setp(ax_gap.get_xticklabels(), visible=False)

    # ---- right bottom: per-epoch increments ----
    for label, (color, pts) in series.items():
        if label == "no-gram control":
            continue
        x = [e for e, _ in pts]
        y = [g for _, g in pts]
        inc = [y[i] - y[i - 1] for i in range(1, len(y))]
        ax_inc.plot(x[1:], inc, color=color, linewidth=0.9, marker="o",
                    markersize=3.2, label=label)
    ax_inc.axhline(0, color="#686d73", linewidth=0.6, linestyle=":")
    ax_inc.set_xlabel("epoch boundary (e−1 → e)")
    ax_inc.set_ylabel("Δgap")
    ax_inc.set_xticks(range(2, 21, 2))
    ax_inc.grid(alpha=0.22)
    ax_inc.legend(fontsize=7.5, frameon=False, loc="upper right")

    figure.suptitle(
        "why multi-epoch training is necessary for forking · 128× · seed 42 · fixed replay of shard 1",
        fontsize=11,
    )
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_v5_s1_epoch_staircase.png"
    figure.savefig(out, dpi=180)
    print("saved", out)


if __name__ == "__main__":
    main()
