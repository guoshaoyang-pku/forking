#!/usr/bin/env python3
"""§56 module-arm logits rank distribution: native / +bigram / +trigram / both.

Sources (authoritative, data/runs_scaling/, seed 42, table LR 128x):
  - s1v5_128_marm_{arm}_ckpt_fixed            (epoch-3 boundary, step 1000)
  - s1v5_128_marm_{arm}_{e1,e2}ckpt_fixed     (epoch-1/2 boundaries, steps 337/674)
Rank statistics per arm come from code/tools/eval_logits_rank.py, which loads
each run's final_model.pt and re-runs the model on the training stream head
and on the same fixed val batches used during training. Because shard-1 fixed
replay + seed 42 make the first 337/674 steps of the 337/674/1000-step runs
bit-identical, the e1/e2/e3 boundaries are the same trajectory sampled at the
end of epoch 1/2/3.

Outputs (provenance goes to sidecar .caption.txt files, NOT into the images):
  - fig_marm_logits_rank.{png,svg}        raw mean softmax probability per
    rank (normalized over the full 8192 vocab), linear y, top-10 ranks
  - fig_marm_logits_rank_share.{png,svg}  per-position top-10 conditional
    share (each position's top-10 probs renormalized to sum 1; from
    code/tools/topk_share_stats.py) — head SHAPE independent of confidence
  - fig_marm_true_rank_cdf.{png,svg}      CDF of true-token rank (step-1000)

No numbers are hand-entered; everything is read from the eval JSONs.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "data/runs_scaling/s1v5_128_marm_logits_rank.json"
DEFAULT_EPOCH_JSON = ROOT / "data/runs_scaling/s1v5_128_marm_logits_rank_epochs.json"
DEFAULT_SHARE_JSON = ROOT / "data/runs_scaling/marm_topk_share.json"
FIGS = ROOT / "docs/figs/theory"

ARMS = ["nogram", "bigram", "trigram", "both"]
LABELS = {
    "nogram": "native (no n-gram)",
    "bigram": "+bigram",
    "trigram": "+trigram",
    "both": "+both",
}
COLORS = {
    "nogram": "#999999",
    "bigram": "#0072b2",
    "trigram": "#009e73",
    "both": "#d55e00",
}
ZORDER = {"nogram": 2, "bigram": 3, "trigram": 3, "both": 4}

EPOCHS = ["e1", "e2", "e3"]
EPOCH_LABELS = {"e1": "epoch 1", "e2": "epoch 2", "e3": "epoch 3"}
EPOCH_ALPHA = {"e1": 0.4, "e2": 0.65, "e3": 1.0}
RANK_MAX = 10  # plot only the top-RANK_MAX ranks, linear axes

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "lines.linewidth": 1.6,
})


def epoch_caption(meta: dict, note: str) -> str:
    runs = ", ".join(EPOCH_DATA[f"{a}_{e}"]["run_id"]
                     for a in ARMS for e in EPOCHS)
    return (f"Runs: {runs} · seed 42 · epoch boundaries = steps "
            f"337/674/1000 (epoch_batches 337) · table LR 128x · "
            f"eval: {meta['n_train_batches']} train batches (stream head) and "
            f"{meta['n_val_batches']} fixed val batches "
            f"({meta['device_batch_size']}x{meta['sequence_len']} tokens each) · "
            f"{note}")


def epoch_rank_panel(ax, arm: str, side: str, values, ppl_of, stat_name="ppl"):
    for ei, ep in enumerate(EPOCHS):
        p = values(arm, ep, side)
        lbl = f"{EPOCH_LABELS[ep]}"
        if ppl_of is not None:
            lbl += f" — {stat_name} {ppl_of(arm, ep, side):.1f}"
        ax.plot(np.arange(1, len(p) + 1), p, color=COLORS[arm],
                alpha=EPOCH_ALPHA[ep], linewidth=1.2 + 0.4 * ei,
                label=lbl, zorder=ZORDER[arm])
    ax.set_xlim(1, len(values(arm, EPOCHS[0], side)))
    ax.set_xticks(range(1, len(values(arm, EPOCHS[0], side)) + 1))
    ax.set_xlabel("logit rank")
    ax.set_title(f"{LABELS[arm]} · {'train' if side == 'train' else 'val'} batches",
                 fontsize=8.5, pad=3)
    ax.legend(loc="upper right", fontsize=6.2)


def true_rank_cdf_panel(ax, side: str):
    vocab = len(ARMS_DATA["nogram"][side]["true_rank_hist"])
    # hist[i] = #positions whose true token has zero-based rank i.
    # Plot one-based rank so rank 1 is the top prediction and the log axis
    # has no invalid zero coordinate.
    ranks = np.arange(1, vocab + 1)
    for arm in ARMS:
        hist = np.asarray(ARMS_DATA[arm][side]["true_rank_hist"], dtype=np.float64)
        cdf = np.cumsum(hist) / hist.sum()
        ax.plot(ranks, cdf, color=COLORS[arm], label=LABELS[arm],
                zorder=ZORDER[arm], linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlim(1, vocab)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("rank of true next token (1 = top)")
    ax.set_ylabel("fraction of positions")
    ax.set_title(f"{'train' if side == 'train' else 'validation'} batches")
    ax.legend(loc="lower right")


def write_caption(name: str, text: str):
    path = FIGS / f"{name}.caption.txt"
    path.write_text(text + "\n")
    print(f"caption -> {path}")


def main():
    global ARMS_DATA, EPOCH_DATA
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DEFAULT_JSON),
                    help="step-1000 eval JSON (drives the true-rank CDF)")
    ap.add_argument("--epoch-json", default=str(DEFAULT_EPOCH_JSON),
                    help="epoch-boundary eval JSON (drives the main figure)")
    ap.add_argument("--share-json", default=str(DEFAULT_SHARE_JSON),
                    help="top-k share JSON from topk_share_stats.py")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    ARMS_DATA = data["arms"]
    meta = data["meta"]
    missing = [a for a in ARMS if a not in ARMS_DATA]
    if missing:
        raise SystemExit(f"eval JSON missing arms: {missing}")

    with open(args.epoch_json) as f:
        epoch_data = json.load(f)
    EPOCH_DATA = epoch_data["arms"]
    epoch_meta = epoch_data["meta"]
    missing = [f"{a}_{e}" for a in ARMS for e in EPOCHS if f"{a}_{e}" not in EPOCH_DATA]
    if missing:
        raise SystemExit(f"epoch eval JSON missing arms: {missing}")
    FIGS.mkdir(parents=True, exist_ok=True)

    share = None
    if Path(args.share_json).exists():
        with open(args.share_json) as f:
            share = json.load(f)["arms"]
    else:
        print(f"[warn] share JSON not found at {args.share_json}; "
              f"skipping the normalized-share figure")

    # figure 1: raw mean softmax probability per rank (normalized over the
    # full vocab), linear y, top-RANK_MAX ranks.
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2), sharex=True,
                             sharey="row")

    def raw_vals(arm, ep, side):
        return np.asarray(EPOCH_DATA[f"{arm}_{ep}"][side]["p_at_rank"])[:RANK_MAX]

    def raw_ppl(arm, ep, side):
        return float(np.exp(EPOCH_DATA[f"{arm}_{ep}"][side]["nll_mean"]))

    for col, arm in enumerate(ARMS):
        epoch_rank_panel(axes[0, col], arm, "train", raw_vals, raw_ppl, "raw")
        epoch_rank_panel(axes[1, col], arm, "val", raw_vals, raw_ppl, "raw")
    for row in range(2):
        axes[row, 0].set_ylabel("mean predictive probability")
    fig.suptitle("Output distribution by logit rank across module ablations "
                 "(epoch boundaries 1/2/3)", fontsize=10, y=0.99)
    fig.subplots_adjust(hspace=0.32, wspace=0.18)
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"fig_marm_logits_rank.{ext}", bbox_inches="tight")
    plt.close(fig)
    write_caption("fig_marm_logits_rank",
                  epoch_caption(epoch_meta,
                                "y = mean softmax probability at each logit "
                                "rank (normalized over the full 8192 vocab); "
                                "linear axes, top-10 ranks"))

    # figure 2: per-position top-10 conditional share — head shape with each
    # position's confidence (total head mass) normalized away.
    if share is not None:
        fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2), sharex=True,
                                 sharey="row")

        def share_vals(arm, ep, side):
            return np.asarray(share[f"{arm}_{ep}"][side]["share_at_rank"])

        def head_H(arm, ep, side):
            return share[f"{arm}_{ep}"][side]["head_entropy_mean"]

        for col, arm in enumerate(ARMS):
            epoch_rank_panel(axes[0, col], arm, "train", share_vals,
                             head_H, stat_name="H")
            epoch_rank_panel(axes[1, col], arm, "val", share_vals,
                             head_H, stat_name="H")
        for row in range(2):
            axes[row, 0].set_ylabel("share of top-10 mass")
        fig.suptitle("Top-10 conditional distribution (per-position "
                     "normalized) — head shape", fontsize=10, y=0.99)
        fig.subplots_adjust(hspace=0.32, wspace=0.18)
        for ext in ("png", "svg"):
            fig.savefig(FIGS / f"fig_marm_logits_rank_share.{ext}",
                        bbox_inches="tight")
        plt.close(fig)
        write_caption("fig_marm_logits_rank_share",
                      epoch_caption(epoch_meta,
                                    "y = per-position top-10 conditional "
                                    "share, i.e. probs[r]/sum_top10(probs) "
                                    "averaged over positions (confidence "
                                    "normalized away); legend shows mean "
                                    "within-head entropy instead of ppl"))

    # companion: true-token rank CDFs (step-1000)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.7), sharey=True)
    true_rank_cdf_panel(axes[0], "train")
    true_rank_cdf_panel(axes[1], "val")
    fig.suptitle("Rank of the true next token under each module arm",
                 fontsize=10, y=1.02)
    fig.subplots_adjust(wspace=0.2)
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"fig_marm_true_rank_cdf.{ext}", bbox_inches="tight")
    plt.close(fig)
    write_caption("fig_marm_true_rank_cdf",
                  f"Runs: " + ", ".join(ARMS_DATA[a]["run_id"] for a in ARMS) +
                  " · seed 42 · step 1000 · table LR 128x · eval: "
                  f"{meta['n_train_batches']} train batches (stream head, "
                  f"seen ~3x) and {meta['n_val_batches']} fixed val batches "
                  f"({meta['device_batch_size']}x{meta['sequence_len']} "
                  "tokens each)")

    # console summary for backfill
    for side in ("train", "val"):
        print(f"[{side}] epoch boundaries:")
        for arm in ARMS:
            for ep in EPOCHS:
                d = EPOCH_DATA[f"{arm}_{ep}"][side]
                print(f"  {arm:8s} {EPOCH_LABELS[ep]} ppl={np.exp(d['nll_mean']):7.2f} "
                      f"H={d['entropy_mean']:6.3f} margin={d['margin_mean']:6.4f} "
                      f"top1={d['top1_mean']:6.4f} top10={d['top10_mass']:6.4f} "
                      f"p(true rank0)={d['true_rank_hist'][0] / d['n_positions']:6.4f}")
    print(f"wrote fig_marm_logits_rank.{{png,svg}}, "
          f"fig_marm_logits_rank_share.{{png,svg}}, "
          f"fig_marm_true_rank_cdf.{{png,svg}}")


if __name__ == "__main__":
    main()
