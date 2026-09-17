#!/usr/bin/env python3
"""Hypothesis-verification figure: val damage concentrates at novel/low-freq
contexts.

Panel test (per-position paired excess NLL vs the native run, same eval
positions, 95% CI = mean +/- 1.96*sd/sqrt(n)):
  (a-c) excess NLL by trigram-context train frequency, +bigram/+trigram/+both,
        epochs 1-3;
  (d)   share of the +both e3 total excess attributable to each bin.

Sources: data/runs_scaling/marm_nll_excess.json from
code/tools/nll_excess_by_ctx.py (npz per-position probs x train-shard
trigram counts). Output: docs/figs/theory/fig_marm_gap_by_ctxfreq.{png,svg}
+ .caption.txt sidecar. No hand-entered numbers.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXCESS_JSON = ROOT / "data/runs_scaling/marm_nll_excess.json"
FIGS = ROOT / "docs/figs/theory"

ARMS = ["bigram", "trigram", "both"]
ARM_TITLES = {"bigram": "+bigram", "trigram": "+trigram", "both": "+both"}
EPOCHS = ["e1", "e2", "e3"]
EP_COLORS = ["#fdd0a2", "#fd8d3c", "#a63603"]
EP_LABELS = ["epoch 1", "epoch 2", "epoch 3"]
BINS = ["novel", "1", "2-3", "4-31", "32-511", "512+"]
BIN_NAMES = ["novel", "f=1", "f=2-3", "f=4-31", "f=32-511", "f=512+"]

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
    "axes.axisbelow": True,
})


def main():
    d = json.loads(EXCESS_JSON.read_text())
    fracs = d["arms"]["both"]["e3"]["bins"]
    bin_labels = [f"{n}\n({fracs[b]['frac_val']:.0%})"
                  for n, b in zip(BIN_NAMES, BINS)]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))

    for ax, arm in zip([axes[0, 0], axes[0, 1], axes[1, 0]], ARMS):
        x = np.arange(len(BINS))
        for ei, ep in enumerate(EPOCHS):
            rec = d["arms"][arm][ep]
            m = np.array([rec["bins"][b]["excess_mean"] for b in BINS])
            lo = np.array([rec["bins"][b]["ci95_lo"] for b in BINS])
            hi = np.array([rec["bins"][b]["ci95_hi"] for b in BINS])
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="-o",
                        color=EP_COLORS[ei], label=EP_LABELS[ei],
                        markersize=3, linewidth=1.4, capsize=2)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=6.5)
        ax.set_title(f"{ARM_TITLES[arm]} vs native · val batches", pad=3)
        ax.set_ylabel("paired excess NLL (nats)")
        ax.legend(loc="upper right")

    ax = axes[1, 1]
    shares = [fracs[b]["contrib_share"] for b in BINS]
    x = np.arange(len(BINS))
    ax.bar(x, shares, color=["#a63603", "#e6550d", "#fd8d3c", "#fdae6b",
                             "#fdd0a2", "#d9d9d9"])
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=6.5)
    ax.set_ylabel("share of total excess NLL")
    ax.set_title("+both · e3 · where the val damage comes from", pad=3)
    for x0, s in zip(x, shares):
        ax.text(x0, s + 0.012, f"{s:.0%}", ha="center", fontsize=6.5)

    fig.subplots_adjust(hspace=0.34, wspace=0.24)
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"fig_marm_gap_by_ctxfreq.{ext}", bbox_inches="tight")
    plt.close(fig)
    (FIGS / "fig_marm_gap_by_ctxfreq.caption.txt").write_text(
        "Runs: s1v5_128_marm_{bigram,trigram,both}{,_ckpt,_e1ckpt,_e2ckpt}"
        "_fixed vs s1v5_128_marm_nogram_* · seed 42 · epoch boundaries = "
        "steps 337/674/1000 · table LR 128x · eval: 4 fixed val batches "
        "(shards 2,3,4,5,6,7,8,9,10,6542; 72x2048 tokens each) · y = paired "
        "per-position excess NLL vs the native run at the same positions "
        "(-log p(true)); error bars = 95% CI (1.96*sd/sqrt(n)) · context "
        "bins = train-stream trigram (a,b,c) count; bin labels show the "
        "share of val tokens · source: marm_nll_excess.json "
        "(code/tools/nll_excess_by_ctx.py)\n")
    print(f"wrote {FIGS}/fig_marm_gap_by_ctxfreq.{{png,svg}}")


if __name__ == "__main__":
    main()
