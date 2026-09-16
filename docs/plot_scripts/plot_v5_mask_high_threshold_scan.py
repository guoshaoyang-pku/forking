#!/usr/bin/env python3
"""mask_high threshold scan figure (128×, epoch-2 boundary).

Plots final gap at step 1000 vs mask_high frequency threshold (high→low),
all 14 points from the causalv5m2 inclusive-semantics refresh (F>=t).
Points are raw final gaps; thin 3-point connector only as visual aid.

Mask semantics: the frequency mask is context-level.  For t>0, `high`
masks only contexts seen in the train shard with F>=t, so novel contexts
(F=0) remain active and the completed scan stops at t=1.  The lower
boundary t=0 is defined as a full context mask, including novel contexts,
and must be regenerated before it can be plotted.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = ROOT / "data" / "runs_fixed"
OUT = ROOT / "docs" / "figs" / "main"
OUT.mkdir(parents=True, exist_ok=True)

# (threshold, run_id) — causalv5m2_* are the F>=t inclusive-semantics refresh;
# t=200 is part of the same causalv5m2 batch (ophis GPU3).
POINTS = [
    (12800, "causalv5m2_mask_high_t12800_e1_fixed"),
    (6400, "causalv5m2_mask_high_t6400_e1_fixed"),
    (3200, "causalv5m2_mask_high_t3200_e1_fixed"),
    (1600, "causalv5m2_mask_high_t1600_e1_fixed"),
    (800, "causalv5m2_mask_high_t800_e1_fixed"),
    (400, "causalv5m2_mask_high_t400_e1_fixed"),
    (200, "causalv5m2_mask_high_t200_e1_fixed"),
    (100, "causalv5m2_mask_high_t100_e1_fixed"),
    (50, "causalv5m2_mask_high_t50_e1_fixed"),
    (25, "causalv5m2_mask_high_t25_e1_fixed"),
    (10, "causalv5m2_mask_high_t10_e1_fixed"),
    (5, "causalv5m2_mask_high_t5_e1_fixed"),
    (2, "causalv5m2_mask_high_t2_e1_fixed"),
    (1, "causalv5m2_mask_high_t1_e1_fixed"),
]


def read_final_gap(run_id):
    path = RUNS_FIXED / run_id / "summary.json"
    if not path.exists():
        return None, "missing"
    data = json.loads(path.read_text())
    summary = data.get("summary", data)
    gap = summary.get("final_gap")
    steps = summary.get("steps")
    if gap is None:
        # fall back to train_log last row
        log_path = RUNS_FIXED / run_id / "train_log.jsonl"
        if log_path.exists():
            rows = [json.loads(line) for line in log_path.open() if line.strip()]
            if rows:
                row = max(rows, key=lambda r: r.get("step", -1))
                gap = row.get("gap", row["val_loss"] - row["train_loss"])
                steps = row.get("step")
    return (float(gap), steps)


def main():
    x, y, labels = [], [], []
    for threshold, run_id in POINTS:
        gap, steps = read_final_gap(run_id)
        if gap is None:
            print(f"[skip] {run_id}: {steps}")
            continue
        x.append(threshold)
        y.append(gap)
        labels.append(run_id.replace("causalv5m2_mask_high_", "mask_high ").replace(
            "_e1_fixed", "").replace("causalv5c_", "mask_high "))
        print(f"{run_id}: gap={gap:.4f} @step {steps}")

    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    labels = np.asarray(labels)[order]

    figure, axis = plt.subplots(figsize=(9.6, 5.6))
    axis.scatter(x, y, s=42, color="#0f766e", zorder=3, alpha=0.9)
    # thin 3-point visual connector
    for start in range(len(x) - 2):
        window_x = x[start:start + 3]
        window_y = y[start:start + 3]
        axis.plot(window_x, window_y, color="#0f766e",
                  linewidth=0.7, alpha=0.45, zorder=2)
    # Label only the landmarks so the dense high-threshold plateau remains
    # readable; every point is still retained as a raw marker.
    label_thresholds = {1, 10, 100, 200, 3200, 12800}
    for xi, yi in zip(x, y):
        if int(xi) not in label_thresholds:
            continue
        axis.annotate(f"$t={int(xi)}$", (xi, yi), textcoords="offset points",
                      xytext=(4, 7), fontsize=8, color="#374151")

    axis.set_xscale("log")
    axis.set_xlabel("mask_high frequency threshold $t$ (mask contexts with $F \\geq t$)")
    axis.set_ylabel("final gap at step 1000")
    axis.axhline(0, color="#686d73", linewidth=0.8, linestyle=":")
    axis.grid(alpha=0.25, which="both")
    axis.set_title(
        "mask_high threshold scan · 128× · epoch-2 boundary (1000 steps)\n"
        "lower $t$ masks more seen contexts; novel ($F{=}0$) contexts are never masked;\n"
        "all points use the inclusive F$\\geq t$ semantics (causalv5m2 refresh)"
    )
    figure.tight_layout()
    out = OUT / "fig_v5_128x_mask_high_threshold_scan.png"
    figure.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print("saved:", out)


if __name__ == "__main__":
    main()
