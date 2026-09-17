"""Core-logic figure for Section 4.1: where the output mass goes.

For one module arm versus the native (no n-gram) model, split eval positions
by the n-gram context of the module (bigram key = (t-1,t), trigram key =
(t-2,t-1,t)) into three situations that share the same seen-context set:

  row A  train · seen context · seen continuation   (memorisation)
  row B  val   · seen context · seen continuation   (benefit / safe case)
  row C  val   · seen context · novel continuation  (crowding-out)

Left column: mean probability at logit rank 1..10 at epoch 3, native vs arm,
each bar stacked into mass on candidates that co-occurred with the context
in the train shard ("train-seen candidates", dark) and all other candidates
(light). Horizontal lines: mean p(true token). Right column: epoch 1->3
trajectories of p(true) (solid) and top-32 mass on train-seen candidates
(dashed) for native vs arm.

Main-text figure (--novel): 1 x 4, novel-continuation group (row C) only,
one rank panel per epoch 1/2/3 plus a CE/ppl trajectory panel. The
seen-continuation groups (rows A/B) go to the appendix figure (--appendix).
Bar segments are labelled by in-panel arrows instead of a patch legend.

Data: data/runs_scaling/marm_subspace_mass.json produced by
code/tools/logits_subspace_stats.py from the module_arms_epoch_*_ckpt
per-position npz (eval_logits_rank.py --save_topk 32).

Usage:
  python docs/plot_scripts/plot_marm_subspace_mass.py --arm both --novel
  python docs/plot_scripts/plot_marm_subspace_mass.py --arm both --appendix
  python docs/plot_scripts/plot_marm_subspace_mass.py --arm bigram --key bigram
  python docs/plot_scripts/plot_marm_subspace_mass.py --arm trigram --key trigram
  python docs/plot_scripts/plot_marm_subspace_mass.py --arm both --key bigram
  python docs/plot_scripts/plot_marm_subspace_mass.py --summary
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MARM_SUBSPACE_JSON",
                           ROOT / "data" / "runs_scaling" / "marm_subspace_mass.json"))
FIGS = ROOT / "docs" / "figs" / "theory"

LABELS = {"nogram": "native", "bigram": "+bigram", "trigram": "+trigram",
          "both": "+both"}
COLORS = {"nogram": "#8c8c8c", "bigram": "#0072b2", "trigram": "#009e73",
          "both": "#d55e00"}
EPOCHS = ["e1", "e2", "e3"]
EP_LABELS = {"e1": "epoch 1", "e2": "epoch 2", "e3": "epoch 3"}
EP_X = [1, 2, 3]
NR = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.6,
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

ROWS = [
    ("train", "ctxseen|seen", "A · train · seen context · seen continuation",
     "memorisation"),
    ("val", "ctxseen|seen", "B · val · seen context · seen continuation",
     "benefit"),
    ("val", "ctxseen|novel", "C · val · seen context · novel continuation",
     "crowding-out"),
]


def lighten(hex_color, amount=0.55):
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * amount)


def get(d, arm, ep, side, key, grp):
    return d["arms"][arm][ep][side][key].get(grp)


def share_of_side(d, arm, side, key, grp):
    g_all = d["arms"][arm]["e3"][side][key]["all"]["n"]
    g = d["arms"][arm]["e3"][side][key].get(grp)
    return (g["n"] / g_all) if g else 0.0


def rank_panel(ax, d, arm, key, side, grp, ep="e3", ylabel=True,
               annotate=False, legend=False):
    w = 0.38
    x = np.arange(1, NR + 1)
    ptrue = {}
    ymax = 0.0
    for j, a in enumerate(("nogram", arm)):
        g = get(d, a, ep, side, key, grp)
        if g is None:
            continue
        p = np.array(g["rank_prob"][:NR])
        ps = np.array(g["rank_seen_prob"][:NR])
        off = (j - 0.5) * w
        ax.bar(x + off, ps, w, color=COLORS[a], edgecolor="none", zorder=3)
        ax.bar(x + off, p - ps, w, bottom=ps, color=lighten(COLORS[a]),
               edgecolor="none", zorder=3)
        ptrue[a] = g["true_prob_mean"]
        ymax = max(ymax, p[0])
    for a, v in ptrue.items():
        ax.axhline(v, color=COLORS[a], linestyle="--", linewidth=1.1, zorder=4)
        ax.text(NR + 0.55, v, f"{v:.3f}", color=COLORS[a], fontsize=7.2,
                va="center", ha="left")
    ax.set_ylim(0, ymax * (1.42 if annotate else 1.24))
    ax.set_xlim(0.4, NR + 0.6)
    ax.set_xticks(x)
    ax.set_xlabel("logit rank")
    if ylabel:
        ax.set_ylabel("mean predictive probability")
    if annotate:
        g = get(d, arm, ep, side, key, grp)
        ps0 = g["rank_seen_prob"][0]
        p0 = g["rank_prob"][0]
        ax.annotate("train-seen\ncandidates", xy=(1, ps0 * 0.55),
                    xytext=(1.9, ymax * 1.16), fontsize=7.6,
                    color=COLORS[arm], ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=COLORS[arm],
                                    lw=0.9))
        ax.annotate("other\ncandidates",
                    xy=(1, ps0 + (p0 - ps0) * 0.62),
                    xytext=(5.6, ymax * 1.16), fontsize=7.6,
                    color="#666666", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
    if legend:
        handles = [
            Patch(color=COLORS["nogram"], label="native (no n-gram)"),
            Patch(color=COLORS[arm], label=f"{LABELS[arm]} model"),
            Line2D([], [], color="k", linestyle="--", linewidth=1.1,
                   label="mean p(true token)"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7.6)


def epoch_panel(ax, d, arm, key, side, grp, ylabel=True):
    for a in ("nogram", arm):
        pt, sm = [], []
        for ep in EPOCHS:
            g = get(d, a, ep, side, key, grp)
            pt.append(g["true_prob_mean"] if g else np.nan)
            sm.append(g["seen_mass"] if g else np.nan)
        ax.plot(EP_X, pt, "-o", color=COLORS[a], markersize=3.5, zorder=4)
        ax.plot(EP_X, sm, "--s", color=COLORS[a], markersize=3, zorder=3,
                alpha=0.9)
        ax.text(3.08, pt[-1], f"{pt[-1]:.3f}", color=COLORS[a], fontsize=7,
                va="center")
        ax.text(3.08, sm[-1], f"{sm[-1]:.2f}", color=COLORS[a], fontsize=7,
                va="center", alpha=0.9)
    ax.set_xlim(0.8, 3.45)
    ax.set_xticks(EP_X)
    ax.set_xticklabels(["epoch 1", "epoch 2", "epoch 3"])
    ax.set_ylim(0, 0.82)
    if ylabel:
        ax.set_ylabel("probability")


def ce_panel(ax, d, arm, key, side, grp, legend=True,
             legend_loc="upper left"):
    all_ce = []
    for a in ("nogram", arm):
        ce, ppl = [], []
        for ep in EPOCHS:
            g = get(d, a, ep, side, key, grp)
            ce.append(g["ce_mean"])
            ppl.append(g["ppl"])
        all_ce += ce
        ax.plot(EP_X, ce, "-o", color=COLORS[a], markersize=4,
                label=("native (no n-gram)" if a == "nogram"
                       else f"{LABELS[arm]} model"))
        for x, c, pp in zip(EP_X, ce, ppl):
            if a == "nogram":
                ax.annotate(f"ppl {pp:,.0f}", xy=(x, c),
                            xytext=(x - 0.07, c - 0.42), fontsize=7,
                            color=COLORS[a], ha="right")
            else:
                ax.annotate(f"ppl {pp:,.0f}", xy=(x, c),
                            xytext=(x + 0.07, c + 0.22), fontsize=7,
                            color=COLORS[a])
    span = max(all_ce) - min(all_ce)
    ax.set_xlim(0.8, 3.55)
    ax.set_ylim(min(all_ce) - 0.55 - 0.08 * span, max(all_ce) + 0.5)
    ax.set_xticks(EP_X)
    ax.set_xticklabels(["epoch 1", "epoch 2", "epoch 3"])
    ax.set_xlabel("epoch boundary")
    ax.set_ylabel("mean cross-entropy (nats)")
    if legend:
        ax.legend(loc=legend_loc, fontsize=7.8)


def novel_figure(d, arm="both", key="bigram"):
    """Main-text figure: novel-continuation group only, 3 epochs + CE/ppl."""
    side, grp = "val", "ctxseen|novel"
    share = share_of_side(d, arm, side, key, grp)
    n = get(d, arm, "e3", side, key, grp)["n"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.05],
                                          "wspace": 0.34})
    for j, ep in enumerate(EPOCHS):
        rank_panel(axes[j], d, arm, key, side, grp, ep=ep,
                   annotate=(j == 0), legend=(j == 2))
        axes[j].set_title(EP_LABELS[ep], loc="left", fontsize=9.5)
    ce_panel(axes[3], d, arm, key, side, grp)
    axes[3].set_title("CE / ppl of this group", loc="left", fontsize=9.5)
    fig.suptitle(
        f"Crowding-out: val positions with a train-seen bigram context whose "
        f"true continuation was never seen with it ({share:.0%} of val, "
        f"n={n:,}) · {LABELS[arm]} vs native", fontsize=10.5, y=1.02)
    name = "fig_marm_subspace_both_novel"
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    write_caption(name, d, arm, key)
    print(f"-> {FIGS / name}.png")


def appendix_figure(d, arm="both", key="bigram"):
    """Appendix: seen-continuation groups (train row A, val row B)."""
    rows = [r for r in ROWS if r[1] == "ctxseen|seen"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.05],
                                          "wspace": 0.34, "hspace": 0.62})
    for r, (side, grp, title, tag) in enumerate(rows):
        share = share_of_side(d, arm, side, key, grp)
        n = get(d, arm, "e3", side, key, grp)["n"]
        for j, ep in enumerate(EPOCHS):
            rank_panel(axes[r, j], d, arm, key, side, grp, ep=ep,
                       annotate=(r == 0 and j == 0),
                       legend=(r == 0 and j == 2))
            axes[r, j].set_title(f"{title} · {EP_LABELS[ep]}\n{tag} · "
                                 f"{share:.0%} of {side} (n={n:,})",
                                 loc="left", fontsize=8.8)
        ce_panel(axes[r, 3], d, arm, key, side, grp, legend=(r == 0),
                 legend_loc="lower left")
        axes[r, 3].set_title(f"{title}\nCE / ppl of this group",
                             loc="left", fontsize=8.8)
    fig.suptitle(f"Seen-continuation groups (appendix): {LABELS[arm]} vs "
                 f"native · context = bigram (t-1,t)", fontsize=10.5, y=1.0)
    name = "fig_marm_subspace_both_seen_appendix"
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    write_caption(name, d, arm, key)
    print(f"-> {FIGS / name}.png")


def arm_figure(d, arm, key):
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 8.6),
                             gridspec_kw={"width_ratios": [1.55, 1],
                                          "hspace": 0.55, "wspace": 0.32})
    for r, (side, grp, title, tag) in enumerate(ROWS):
        share = share_of_side(d, arm, side, key, grp)
        n = get(d, arm, "e3", side, key, grp)["n"]
        axes[r, 0].set_title(f"{title}\n{tag} · {share:.0%} of {side} "
                             f"positions (n={n:,}) · epoch 3",
                             loc="left", fontsize=9.2)
        rank_panel(axes[r, 0], d, arm, key, side, grp)
        axes[r, 1].set_title("\nepoch 1 → 3", loc="left", fontsize=9.2)
        epoch_panel(axes[r, 1], d, arm, key, side, grp)

    handles = [
        Patch(color=COLORS["nogram"], label="native · mass on train-seen candidates"),
        Patch(color=lighten(COLORS["nogram"]), label="native · other candidates"),
        Patch(color=COLORS[arm], label=f"{LABELS[arm]} · mass on train-seen candidates"),
        Patch(color=lighten(COLORS[arm]), label=f"{LABELS[arm]} · other candidates"),
        Line2D([], [], color="k", linestyle="--", linewidth=1.1,
               label="mean p(true token)  [left]"),
        Line2D([], [], color="k", marker="o", markersize=3.5,
               label="p(true token)  [right]"),
        Line2D([], [], color="k", linestyle="--", marker="s", markersize=3,
               label="top-32 mass on train-seen candidates  [right]"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.4,
               handlelength=1.6, columnspacing=1.4)
    kname = "bigram (t-1,t)" if key == "bigram" else "trigram (t-2,t-1,t)"
    fig.suptitle(f"Where the output mass goes: {LABELS[arm]} vs native · "
                 f"context = {kname} · epoch 3 (left), epochs 1-3 (right)",
                 fontsize=10.5, y=0.995)
    name = f"fig_marm_subspace_{arm}_{key}ctx"
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    write_caption(name, d, arm, key)
    print(f"-> {FIGS / name}.png")


def summary_figure(d):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2),
                             gridspec_kw={"wspace": 0.35})
    specs = [
        ("train", "ctxseen|seen", "A · train · seen cont. · p(true)",
         "true_prob_mean"),
        ("val", "ctxseen|novel", "C · val · novel cont. · p(true)",
         "true_prob_mean"),
        ("val", "ctxseen|novel", "C · val · novel cont. · mass on train-seen "
         "candidates", "seen_mass"),
    ]
    for ax, (side, grp, title, field) in zip(axes, specs):
        for arm in ("nogram", "bigram", "trigram", "both"):
            ys = [get(d, arm, ep, side, "bigram", grp)[field] for ep in EPOCHS]
            ax.plot(EP_X, ys, "-o", color=COLORS[arm], markersize=3.5,
                    label=LABELS[arm])
        ax.set_xticks(EP_X)
        ax.set_xticklabels(["epoch 1", "epoch 2", "epoch 3"])
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("probability")
    axes[0].legend(loc="upper left")
    fig.suptitle("All arms vs native · context = bigram (t-1,t) · val/train "
                 "positions with seen context", fontsize=10, y=1.02)
    name = "fig_marm_subspace_summary"
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    write_caption(name, d, "all", "bigram")
    print(f"-> {FIGS / name}.png")


def write_caption(name, d, arm, key):
    meta = d["meta"]
    txt = (f"Runs: module_arms_epoch_{{nogram,bigram,trigram,both}}_{{e1,e2,e3}}"
           f"_ckpt (seed 42, steps 337/674/1000 = epoch boundaries, table LR "
           f"128x, R=2^20 clean tables) · per-position top-32 from "
           f"eval_logits_rank.py --save_topk 32 on fixed train/val batches · "
           f"n-gram counts from train {meta['shard']} ({meta['n_tokens']:,} "
           f"tokens) · context key = {key}; 'seen' = co-occurred with the "
           f"context in the train shard · masses are top-32 only · CE = mean "
           f"per-position -log p(true token), ppl = exp(mean CE) · arm={arm} · "
           f"stats: code/tools/logits_subspace_stats.py -> "
           f"data/runs_scaling/marm_subspace_mass.json · "
           f"plot: docs/plot_scripts/plot_marm_subspace_mass.py")
    (FIGS / f"{name}.caption.txt").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bigram")
    ap.add_argument("--key", default="bigram", choices=["bigram", "trigram"])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--novel", action="store_true",
                    help="main-text figure: novel-continuation group, 3 epochs")
    ap.add_argument("--appendix", action="store_true",
                    help="appendix figure: seen-continuation groups")
    args = ap.parse_args()
    d = json.load(open(DATA))
    if args.summary:
        summary_figure(d)
    elif args.novel:
        novel_figure(d, args.arm, args.key)
    elif args.appendix:
        appendix_figure(d, args.arm, args.key)
    else:
        arm_figure(d, args.arm, args.key)


if __name__ == "__main__":
    main()
