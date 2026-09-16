#!/usr/bin/env python3
"""Paper Figure 1: forking under fixed replay, four injection arms.

Two side-by-side panels (per the manuscript's Figure-1 spec):
  left  -- online train loss (solid) and fixed val loss (dashed) per arm;
  right -- gap = fixed val loss - online train loss.

Source runs (minimal setting, table LR 128x, seed 42, 2000 steps):
  nglab1x_{input,y,v,nogram}_v5_128x_freq10_fixed
Set NGLAB_RUNS_FIXED to the mirror containing the four runs when the
repo data/ directory is not mounted.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
OUT_DIR = ROOT / "docs" / "figs" / "main"
OUT_NAME = "fig_paper1_forking_curves"
EPOCH_LEN = 337
WINDOW = 3  # 3-point (30-step) moving average of the 10-step logs

ARMS = ["input", "y", "v", "nogram"]
RUN = "nglab1x_{arm}_v5_128x_freq10_fixed"


def load(arm):
    rows = []
    path = MIRROR / RUN.format(arm=arm) / "train_log.jsonl"
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def moving_average(values, window=WINDOW):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main():
    S.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax_loss, ax_gap = axes

    for arm in ARMS:
        rows = load(arm)
        steps = [r["step"] for r in rows]
        train = [r["train_loss"] for r in rows]
        val = [r["val_loss"] for r in rows]
        gap = [r["gap"] for r in rows]
        color = S.ARM_COLORS[arm]

        ax_loss.plot(steps, moving_average(train), color=color, lw=1.4,
                     label=arm, zorder=3)
        ax_loss.plot(steps, moving_average(val), color=color, lw=1.2,
                     ls="--", alpha=0.85, zorder=3)
        ax_gap.plot(steps, moving_average(gap), color=color, lw=1.4,
                    label=arm, zorder=3)

    for ax in axes:
        S.add_epoch_lines(ax, epoch_len=EPOCH_LEN, xmax=2000)
        ax.set_xlim(0, 2000)
        ax.set_xlabel("optimizer step")
    ax_loss.set_ylabel("cross-entropy (nats)")
    ax_loss.set_title("Training (solid) and validation (dashed) loss")
    ax_gap.set_ylabel("gap = val − train (nats)")
    ax_gap.set_title("Validation minus training gap")
    ax_gap.axhline(0, color="#acb4bf", lw=0.7, zorder=1)
    ax_loss.legend(title="injection arm", loc="upper right")

    fig.tight_layout()
    png, svg = S.save(fig, OUT_DIR, OUT_NAME)
    print(f"wrote {png}\n     {svg}")


if __name__ == "__main__":
    main()
