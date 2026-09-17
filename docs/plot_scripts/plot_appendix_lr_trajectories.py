#!/usr/bin/env python3
"""Plot the recorded table-learning-rate trajectories for Appendix C.

The input is the frozen evidence packet committed in
``docs/appendices/lr_beta_ablation/results/appendix_data.json``.  No values
are entered by hand: each line is read from the recorded 2000-step trajectory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v5_style import apply_style, save

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/appendices/lr_beta_ablation/results/appendix_data.json"
OUT = ROOT / "docs/figs/appendices"

COLORS = {
    1.0: "#2d6f9f",
    2.0: "#c58a0b",
    4.0: "#c4493d",
}


def load():
    raw = json.loads(DATA.read_text())
    groups = {}
    # The frozen packet has complete, same-β₂ trajectories for ×2 and ×4.
    # The nominal ×1 replacement is only a partial snapshot, so it is not
    # silently promoted to a long curve here.
    for shard, prefix in (("1x", "b2_099_1x_lr"), ("2-epoch", "b2_099_2ep_lr")):
        groups[shard] = {}
        for scale in (2.0, 4.0):
            key = f"{prefix}{int(scale)}"
            if key not in raw:
                raise RuntimeError(f"missing recorded trajectory: {key}")
            rec = raw[key]
            if rec.get("steps") not in (None, 2000) or rec.get("traj", {}).get("step", [None])[-1] != 2000:
                raise RuntimeError(f"expected 2000-step trajectory: {key}")
            if rec.get("config", {}).get("table_betas") != [0.0, 0.99]:
                raise RuntimeError(f"table beta2 is not fixed at 0.99: {key}")
            groups[shard][scale] = rec
    return groups


def main():
    groups = load()
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.2), sharex=False)
    fields = (("train_loss", "train loss"), ("val_loss", "fixed validation loss"), ("gap", "gap = val − train"))
    for row, (shard, records) in enumerate(groups.items()):
        for col, (field, ylabel) in enumerate(fields):
            ax = axes[row, col]
            for scale, rec in records.items():
                traj = rec["traj"]
                ax.plot(traj["step"], traj[field], color=COLORS[scale], lw=1.35, label=f"table LR ×{scale:g}")
                ax.plot(traj["step"][-1], traj[field][-1], "o", color=COLORS[scale], ms=3.8)
            for bound in next(iter(records.values()))["traj"].get("epoch_bounds", []):
                ax.axvline(bound, color="#999999", ls="--", lw=0.75, alpha=0.55)
            if field == "gap":
                ax.axhline(0.0, color="#888888", lw=0.7)
            ax.set_title(f"{shard} shard · {ylabel}")
            ax.set_xlabel("training step")
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Table learning-rate trajectories (β₂=0.99, backbone LR=0.004)", fontsize=12)
    fig.text(0.5, 0.01, "Recorded runs: seed 42 · input injection · 2000 steps · gap = fixed val − online train", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    paths = save(fig, OUT, "fig_table_lr_trajectories")
    print("\n".join(str(p.relative_to(ROOT)) for p in paths))


if __name__ == "__main__":
    main()
