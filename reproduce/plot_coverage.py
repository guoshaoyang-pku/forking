#!/usr/bin/env python3
"""Redraw eight coverage fits from packaged observations, not from a paper image."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from verify_coverage import verify, read_csv, DEFAULT_PACKET, STEPS, BINS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    result = verify(DEFAULT_PACKET)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_csv(DEFAULT_PACKET, "online_points.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#152c42", "text.color": "#152c42", "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, constrained_layout=True)
    for i, branch in enumerate(("bigram", "trigram")):
        for j, step in enumerate(STEPS):
            groups = defaultdict(list)
            for row in rows:
                if row["branch"] == branch and row["arm"] == "both" and int(row["step"]) == step:
                    groups[int(row["F_lo"]), int(row["F_hi"])].append(row)
            x = [float(groups[b][0]["f"]) for b in BINS]
            y = [statistics.mean(float(r["gap"]) for r in groups[b]) for b in BINS]
            sd = [statistics.stdev(float(r["gap"]) for r in groups[b]) for b in BINS]
            fit = next(f for f in result["fits"] if f["branch"] == branch and f["step"] == step)
            ax = axes[i, j]
            color = "#196693" if i == 0 else "#c65b2e"
            ax.errorbar(x, y, yerr=sd, fmt="o", ms=3.7, capsize=2, lw=.8, color=color)
            limits = [min(x), max(x)]
            ax.plot(limits, [fit["a"] * v + fit["b"] for v in limits], color="#152c42", lw=1.2)
            ax.set_title(f"{branch.capitalize()} · step {step}", loc="left", fontsize=11)
            ax.text(.04, .96, f"R² = {fit['r2']:.3f}\nRMSE = {fit['rmse']:.3f}", transform=ax.transAxes, va="top", fontsize=9)
            ax.grid(alpha=.16)
            if j == 0:
                ax.set_ylabel("Online gap (nats)")
            if i == 1:
                ax.set_xlabel("Novel-continuation fraction")
    fig.suptitle("Coverage and replay: mean ± sample SD over seeds 42, 43, 44", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved {args.output}; inputs independently verified before plotting")


if __name__ == "__main__":
    main()
