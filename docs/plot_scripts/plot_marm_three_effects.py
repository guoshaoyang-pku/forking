#!/usr/bin/env python3
"""Three-effect decomposition figure: what trained-in n-gram continuations do
to other continuations.

Panels (mean p(true token) on a LOG y axis — per-position NLL loss =
-ln p, so equal vertical distance = equal NLL difference in nats; native
= dashed grey control, module arm = solid orange, epoch 1 -> 3):
  (A) train memorisation: seen-context -> seen-continuation cells, train
      batches, by context frequency (dose-response of effect A); at f=1,
      e3: +both 0.41 vs native 0.22.
  (B) crowding-out: val batches, SEEN contexts, NOVEL continuations,
      across ALL five context-frequency bins — the full "novel
      continuation" column of the 2x2 cell table: native climbs with
      replay at every f (f=1: 0.04 -> 0.16) while +both is suppressed
      (0.04 -> 0.05); the native/both ratio widens with f (~3.5x at f=1,
      ~10x at f=512+, e3).
  (C) generalisation erosion: val batches, NOVEL contexts (novel cont.
      by construction): native climbs (0.08 -> 0.24), +both stalls
      (0.07 -> 0.10).
  (D) the safe case: val batches, seen contexts, seen continuations:
      +both ends ABOVE native (0.62 vs 0.50 at f=1).

Sources: data/runs_scaling/marm_seen_novel.json from
code/tools/context_freq_split.py. Output:
docs/figs/theory/fig_marm_three_effects.{png,svg} + .caption.txt sidecar.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT_JSON = ROOT / "data/runs_scaling/marm_seen_novel.json"
FIGS = ROOT / "docs/figs/theory"

EPOCHS = ["e1", "e2", "e3"]
EPOCH_LABELS = ["epoch 1", "epoch 2", "epoch 3"]
EP_COLORS_BOTH = ["#fdd0a2", "#fd8d3c", "#a63603"]
EP_COLORS_NOG = ["#d9d9d9", "#969696", "#404040"]
FBINS = ["ctx1", "ctx2-3", "ctx4-31", "ctx32-511", "ctx512+"]
FBIN_LABELS = ["f=1", "f=2-3", "f=4-31", "f=32-511", "f=512+"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "axes.axisbelow": True,
})


def g(split, arm, ep, side, key):
    return split["arms"][arm][ep][side]["groups"][key]["true_prob_mean"]


def fbin_panel(ax, split, side, suffix, title, ylabel=None):
    x = np.arange(len(FBINS))
    for ei, ep in enumerate(EPOCHS):
        ax.plot(x, [g(split, "both", ep, side, b + suffix) for b in FBINS],
                "-o", color=EP_COLORS_BOTH[ei], markersize=3.2,
                linewidth=1.5, label=f"+both · {EPOCH_LABELS[ei]}")
    for ei, ep in enumerate(EPOCHS):
        ax.plot(x, [g(split, "nogram", ep, side, b + suffix) for b in FBINS],
                "--s", color=EP_COLORS_NOG[ei], markersize=2.8,
                linewidth=1.1, alpha=0.55 + 0.15 * ei,
                label=f"native · {EPOCH_LABELS[ei]}")
    ax.set_xticks(x)
    ax.set_xticklabels(FBIN_LABELS)
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 1.0)
    ax.set_xlabel("context frequency in train")
    if ylabel:
        ax.set_ylabel("mean p(true token)")
        ax.legend(loc="upper right", fontsize=6.2, ncol=2,
                  columnspacing=0.9, handlelength=1.6)
    ax.set_title(title, pad=3)


def epoch_panel(ax, split, side, key, title, ylabel=None):
    x = np.arange(1, 4)
    for arm, colors, marker, ls in (("both", EP_COLORS_BOTH, "o", "-"),
                                    ("nogram", EP_COLORS_NOG, "s", "--")):
        vals = [g(split, arm, ep, side, key) for ep in EPOCHS]
        ax.plot(x, vals, marker, color=colors[2], markersize=4,
                linewidth=1.6, linestyle=ls,
                label="+both" if arm == "both" else "native")
        for ei, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (x[ei], v), textcoords="offset points",
                        xytext=(6, -2), fontsize=6,
                        color=colors[2], alpha=0.75 + 0.1 * ei)
    ax.set_xticks(x)
    ax.set_xticklabels(EPOCH_LABELS)
    ax.set_xlim(0.8, 3.6)
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 1.0)
    ax.set_xlabel("epoch boundary")
    if ylabel:
        ax.set_ylabel("mean p(true token)")
    ax.set_title(title, pad=3)
    ax.legend(loc="upper left")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-json", default=str(SPLIT_JSON),
                    help="marm_seen_novel.json from context_freq_split.py")
    args = ap.parse_args()
    split = json.loads(Path(args.split_json).read_text())
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1))

    fbin_panel(axes[0], split, "train", "|seen",
               "A · memorisation (train, seen cont.)", ylabel=True)
    fbin_panel(axes[1], split, "val", "|novel",
               "B · crowding-out (val, seen ctx, novel cont.)")
    epoch_panel(axes[2], split, "val", "ctxnovel|novel",
                "C · erosion (val, novel ctx, novel cont.)")
    fbin_panel(axes[3], split, "val", "|seen",
               "D · safe case (val, seen cont.)")

    fig.subplots_adjust(wspace=0.24)
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"fig_marm_three_effects.{ext}", bbox_inches="tight")
    plt.close(fig)
    (FIGS / "fig_marm_three_effects.caption.txt").write_text(
        "Runs: s1v5_128_marm_{both,nogram}{,_ckpt,_e1ckpt,_e2ckpt}_fixed · "
        "seed 42 · epoch boundaries = steps 337/674/1000 · table LR 128x · "
        "eval: 4 train-stream-head batches + 4 fixed val batches (shards "
        "2,3,4,5,6,7,8,9,10,6542; 72x2048 tokens each) · y = mean p(true "
        "token), log scale (per-position NLL loss = -ln p: equal vertical "
        "distance = equal NLL difference in nats) · cells defined by "
        "train-shard trigram context "
        "count f and whether the (context, continuation) 4-gram occurred in "
        "train · panel B shows the full novel-continuation column across "
        "all 5 context-frequency bins (the earlier f=1-only epoch view is "
        "the leftmost tick) · panel C = novel contexts (novel continuation "
        "by construction) · source: marm_seen_novel.json "
        "(code/tools/context_freq_split.py)\n")
    print(f"wrote {FIGS}/fig_marm_three_effects.{{png,svg}}")


if __name__ == "__main__":
    main()
