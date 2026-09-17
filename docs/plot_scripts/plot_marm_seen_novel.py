#!/usr/bin/env python3
"""§56 seen-vs-novel continuation split: does "seen continuations crowd out
other continuations" hold?

Sources (data/runs_scaling/):
  - marm_seen_novel.json       from code/tools/context_freq_split.py
    (per-position npz x train-shard n-gram counts)
  - marm_gap_by_ctxfreq.json   from code/tools/gap_by_context_freq.py
    (training-time freq-bin accumulator, per epoch boundary)

Cells on the x axis of panels (a)/(b)/(d)/(e): context-frequency bin of the
trigram context (a,b,c) in the train stream x whether the true continuation
y completed that context in train (seen 4-gram) or not (novel). "novel ctx"
= the context itself never occurred in train.

Outputs docs/figs/theory/fig_marm_seen_novel.{png,svg} (+ .caption.txt).
No hand-entered numbers.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT_JSON = ROOT / "data/runs_scaling/marm_seen_novel.json"
GAP_JSON = ROOT / "data/runs_scaling/marm_gap_by_ctxfreq.json"
FIGS = ROOT / "docs/figs/theory"

EPOCHS = ["e1", "e2", "e3"]
EPOCH_LABELS = ["epoch 1", "epoch 2", "epoch 3"]
EP_COLORS = {"both": ["#fdd0a2", "#fd8d3c", "#d94801"],
             "nogram": ["#d9d9d9", "#969696", "#525252"]}
CELLS = [("novel ctx", "ctxnovel|novel"),
         ("f=1\nnovel c", "ctx1|novel"), ("f=1\nseen c", "ctx1|seen"),
         ("f=2-3\nnovel c", "ctx2-3|novel"), ("f=2-3\nseen c", "ctx2-3|seen"),
         ("f=4-31\nnovel c", "ctx4-31|novel"), ("f=4-31\nseen c", "ctx4-31|seen"),
         ("f=32-511\nnovel c", "ctx32-511|novel"),
         ("f=32-511\nseen c", "ctx32-511|seen"),
         ("f=512+\nnovel c", "ctx512+|novel"), ("f=512+\nseen c", "ctx512+|seen")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.2,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
})


def cells_panel(ax, split, arm, side, title):
    x = np.arange(len(CELLS))
    for ei, ep in enumerate(EPOCHS):
        vals = [split["arms"][arm][ep][side]["groups"][k]["true_prob_mean"]
                if k in split["arms"][arm][ep][side]["groups"] else np.nan
                for _, k in CELLS]
        ax.plot(x, vals, "-o", color=EP_COLORS[arm][ei],
                label=EPOCH_LABELS[ei], markersize=2.6, linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([c for c, _ in CELLS], fontsize=5.8)
    ax.set_title(title, pad=3)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("trigram context in train × true continuation seen in train")
    ax.legend(loc="upper left")


def gap_bar_panel(ax, gap, title):
    d = gap["arms"]["both"]["trigram"]["e3_step1000"]
    keys = list(d.keys())
    vals = [d[k]["gap_mean"] for k in keys]
    fracs = [d[k]["frac_val"] for k in keys]
    x = np.arange(len(keys))
    bars = ax.bar(x, vals, color=["#d94801" if v > 2 else "#fdae6b" if v > 1
                                  else "#fdd0a2" if v > 0.25 else "#d9d9d9"
                                  for v in vals])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=60, fontsize=5.5)
    ax.set_ylabel("per-token gap (val − train), nats")
    ax.set_title(title, pad=3)
    for x0, f in zip(x, fracs):
        if f >= 0.05:
            ax.text(x0, 0.15, f"{f:.0%}", ha="center", fontsize=5.2,
                    color="#555555", rotation=90)


def seen_mass_panel(ax, split, side, title):
    arms = ["nogram", "bigram", "trigram", "both"]
    colors = ["#999999", "#0072b2", "#009e73", "#d55e00"]
    x = np.arange(len(arms))
    w = 0.25
    for ei, ep in enumerate(EPOCHS):
        vals = [split["arms"][a][ep][side]["seen_mass_topk"] for a in arms]
        ax.bar(x + (ei - 1) * w, vals, w, color=EP_COLORS["both"][ei],
               label=EPOCH_LABELS[ei])
    ax.set_xticks(x)
    ax.set_xticklabels(["native", "+bigram", "+trigram", "+both"], fontsize=7)
    ax.set_ylabel("prob mass on train-seen 4-grams")
    ax.set_ylim(0, 0.75)
    ax.set_title(title, pad=3)
    ax.legend(loc="upper left", ncol=1)


def main():
    split = json.loads(SPLIT_JSON.read_text())
    gap = json.loads(GAP_JSON.read_text())
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    cells_panel(axes[0, 0], split, "both", "val",
                "+both · val batches")
    cells_panel(axes[0, 1], split, "nogram", "val",
                "native · val batches (control)")
    gap_bar_panel(axes[0, 2], gap,
                  "+both · gap by context frequency (trigram ctx, e3)")
    cells_panel(axes[1, 0], split, "both", "train",
                "+both · train batches")
    cells_panel(axes[1, 1], split, "nogram", "train",
                "native · train batches (control)")
    seen_mass_panel(axes[1, 2], split, "train",
                    "mass on train-seen continuations (train batches)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean p(true token)")
    fig.subplots_adjust(hspace=0.42, wspace=0.24)
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"fig_marm_seen_novel.{ext}", bbox_inches="tight")
    plt.close(fig)

    meta = split["meta"]
    (FIGS / "fig_marm_seen_novel.caption.txt").write_text(
        "Runs: s1v5_128_marm_{nogram,bigram,trigram,both}{,_e1ckpt,_e2ckpt}"
        "_fixed · seed 42 · epoch boundaries = steps 337/674/1000 · "
        "table LR 128x · eval: 4 fixed val batches (shards "
        "2,3,4,5,6,7,8,9,10,6542) + 4 train-stream-head batches "
        "(72x2048 tokens each) · n-gram counts from train shard_00001.bin "
        f"({meta['n_tokens']:,} tokens) · y = mean p(true token) per cell; "
        "cells split by trigram-context frequency in train and whether the "
        "(context, continuation) pair occurred in train · mass panel uses "
        "top-32 renormalized probabilities\n")
    print(f"wrote {FIGS}/fig_marm_seen_novel.{{png,svg}}")


if __name__ == "__main__":
    main()
