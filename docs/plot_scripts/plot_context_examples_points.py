"""Show epoch-1 to epoch-5 probability changes with explicit corpus membership.

Reuses the verified full-softmax exports and valid-position counts from the
context-example audit. Lines connect endpoints only, not measured trajectories.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_context_examples_logp import (
    ROOT, SOURCE, CONTEXTS, ARMS, read_inputs, digest, write_csv,
)

OUT = ROOT / "docs/figs/context_examples_points_20260925"
COLORS = {"both": "#0072B2", "train-only": "#0072B2",
          "val-only": "#D55E00", "neither": "#8B9299", "others": "#343A40"}
WORDS = ["dog", "cat", "fish", "horse", "bird", "others"]


def main():
    values, classes, records, count_records, max_error = read_inputs()
    counts = {(r["context"], r["token"]): r for r in count_records}
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": .8, "xtick.major.size": 0,
                         "ytick.major.size": 3.})
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.0))
    fig.subplots_adjust(left=.09, right=.99, top=.84, bottom=.085,
                        wspace=.18, hspace=.83)
    categories = [("both", "Train-seen"), ("val-only", "Val-only (novel)"),
                  ("neither", "Neither"), ("others", "Others (top-20)")]
    cat_handles = [Line2D([], [], color=COLORS[key], marker="D" if key == "others" else "o",
                         linestyle="none", markersize=5, label=label)
                   for key, label in categories]
    fig.legend(handles=cat_handles, loc="upper center", bbox_to_anchor=(.54, .963),
               frameon=False, ncol=4, fontsize=9, columnspacing=1.15, handletextpad=.45)
    epoch_handles = [Line2D([], [], marker="o", color="#343A40", markerfacecolor="white",
                           linestyle="none", markersize=5, label="Epoch 1"),
                     Line2D([], [], marker="o", color="#343A40", linestyle="none",
                           markersize=5, label="Epoch 5")]
    fig.legend(handles=epoch_handles, loc="upper center", bbox_to_anchor=(.54, .925),
               frameon=False, ncol=2, fontsize=9, columnspacing=1.8, handletextpad=.45)
    for col, label in enumerate(["No-gram", "N-gram (bigram + trigram)"]):
        pos = axes[0, col].get_position()
        fig.text(pos.x0 + pos.width / 2, .99, label, ha="center", va="top", fontsize=11.5)
    for row, context in enumerate(CONTEXTS):
        pos = axes[row, 0].get_position()
        fig.text(.54, pos.y1 + .017, f'Prediction after "{context} ..."',
                 ha="center", va="bottom", fontsize=10.5)
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            for j, word in enumerate(WORDS):
                category = classes[context][j] if j < 5 else "others"
                color = COLORS[category]
                marker = "D" if j == 5 else "o"
                y = [values[context, arm, e][j] for e in [1, 5]]
                x = [j - .17, j + .17]
                ax.plot(x, y, color=color, linewidth=1.6, zorder=3)
                ax.plot(x[0], y[0], marker=marker, markersize=5.5, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=1.2, zorder=4)
                ax.plot(x[1], y[1], marker=marker, markersize=5.5, markerfacecolor=color,
                        markeredgecolor=color, markeredgewidth=1.2, zorder=5)
                if j < 5:
                    c = counts[context, word]
                    texts = [str(c["train_count"]), str(c["heldout_count"])]
                else:
                    texts = ["—", "—"]
                for y_pos, label in zip([-.23, -.34], texts):
                    ax.text(j, y_pos, label, transform=ax.get_xaxis_transform(),
                            ha="center", va="center", fontsize=8, color="#333333", clip_on=False)
            ax.set_xticks(np.arange(6), WORDS, fontsize=8.5)
            for tick, category in zip(ax.get_xticklabels(), classes[context] + ["others"]):
                tick.set_color(COLORS[category])
            for y_pos, label in [(-.23, "Train"), (-.34, "Val")]:
                ax.text(-.025, y_pos, label, transform=ax.transAxes, ha="right", va="center",
                        fontsize=8, color="#444444", clip_on=False)
            ax.set_xlim(-.65, 5.65)
            ax.set_ylim(-16, 0)
            ax.set_yticks(np.arange(-16, 1, 4))
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", color="#D8DDE3", linewidth=.6, alpha=.75)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(r"$\log p$ (nats)", fontsize=10)
            ax.text(.02, .97, f"({chr(97 + 2 * row + col)})", transform=ax.transAxes,
                    va="top", fontsize=9, color="#333333")
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"fig_context_examples_points.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=.04)
    svg = OUT / "fig_context_examples_points.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    write_csv(OUT / "plotted_values.csv", records)
    write_csv(OUT / "token_classes.csv", count_records)
    manifest = dict(source="docs/appendices/context_examples_20260925/logits.json",
                    source_sha256=digest(SOURCE / "logits.json"),
                    counts_sha256=digest(SOURCE / "context_counts.json"),
                    script=str(Path(__file__).relative_to(ROOT)),
                    input_reader="docs/plot_scripts/plot_context_examples_logp.py",
                    metric="natural log probability",
                    max_abs_logp_difference_from_export=max_error,
                    epochs=[1, 5], steps=[337, 1685], seed=42, shape=[3, 2], shared_ylim=[-16, 0],
                    encoding="open epoch1 and filled epoch5 points joined; higher means more probable",
                    color="blue train-seen (both/train-only); orange val-only; grey neither; charcoal others",
                    counts="exact context-token occurrences: Train shard1; Val shards2,3,4; valid within-chunk positions",
                    token_classes="corpus membership, not separate model predictions for train and val",
                    trajectory="two endpoints only; connecting line does not imply monotone intermediate changes",
                    others="log of mean top20 probability excluding named tokens; reselected per distribution")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
