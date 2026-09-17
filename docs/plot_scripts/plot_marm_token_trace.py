"""Token-precise suppression trace: how a correct continuation gets crushed.

One figure per picked example (from pick_suppression_examples.py --gradual):
context " im lower call" style, rows = epoch 1/2/3, columns = native vs
+both. Each cell: horizontal bars of the top-8 predicted tokens with their
probabilities (LINEAR scale), plus the true token appended as an extra bar
whenever it is outside the top-8 (labelled with its rank). Bar colour:
green = the true continuation token, dark = train-seen candidate (3-gram
(context, token) present in the train shard), light = other candidate.
Bottom strip: the true token's probability trajectory across epochs
(native vs +both, log y) -- the suppression curve.

Data: marm_suppression_examples_gradual.json produced on ophis-gpu by
code/tools/pick_suppression_examples.py --gradual (val positions, bigram
context seen in train with count f, true continuation novel for that
context, native top-1 correct at epoch 1, +both starting reasonable but
collapsing). Token strings decoded with the project tokenizer (tokenizer.pkl,
vocab 8192; subword units).

Usage:
  python docs/plot_scripts/plot_marm_token_trace.py \
    --json /path/to/marm_suppression_examples_gradual.json [--example 0]
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figs" / "theory" / "marm_token_trace"

ARM_TITLE = {"nogram": "native", "both": "+both"}
EP_TITLE = {"e1": "epoch 1", "e2": "epoch 2", "e3": "epoch 3"}
COLORS = {"nogram": "#8c8c8c", "both": "#d55e00"}
GREEN = "#1b7837"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
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


def esc(s):
    return s.replace("\n", "\\n").replace("\t", "\\t")


def draw_topk(ax, rec, true_str, bar_colors):
    ids = rec["topk_ids"]
    strs = rec["topk_strs"]
    probs = np.array(rec["topk_probs"], dtype=float)
    seen = rec["topk_seen"]
    trank = rec["true_rank"]
    tprob = rec["true_prob"]

    labels, vals, cols = [], [], []
    for s, p, se in zip(strs, probs, seen):
        labels.append(esc(s))
        vals.append(p)
        cols.append(bar_colors["seen"] if se else bar_colors["other"])
    # does the true token appear among the shown top-k?
    in_top = trank < len(ids)
    if not in_top:
        labels.append(f"{esc(true_str)}  (rank {trank})")
        vals.append(max(tprob, probs.min() * 0.35))
        cols.append(GREEN)
    for j, (s, se) in enumerate(zip(strs, seen)):
        if not se and trank == j:
            cols[j] = GREEN

    y = np.arange(len(labels))
    ax.barh(y, vals, height=0.72, color=cols, edgecolor="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
    for tick, c in zip(ax.get_yticklabels(), cols):
        if c == GREEN:
            tick.set_color(GREEN)
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    xmax = max(max(vals) * 1.22, 0.02)
    ax.set_xlim(0, xmax)
    for yi, v in zip(y, vals):
        ax.text(v + xmax * 0.015, yi, f"{v:.3f}", va="center",
                fontsize=6.4, color="#444444")
    ax.tick_params(axis="x", labelsize=7)


def draw_traj(ax, ex):
    eps = ("e1", "e2", "e3")
    xs = [1, 2, 3]
    for arm in ("nogram", "both"):
        probs = [max(ex[f"{arm}_{ep}"]["true_prob"], 1e-7) for ep in eps]
        ranks = [ex[f"{arm}_{ep}"]["true_rank"] for ep in eps]
        ax.plot(xs, probs, "-o", color=COLORS[arm], markersize=4.5, zorder=4)
        for x, p, r in zip(xs, probs, ranks):
            ax.annotate(f"{p:.4f}\n(rank {r})", xy=(x, p),
                        xytext=(x + 0.04, p * 1.6), fontsize=6.4,
                        color=COLORS[arm], ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 1.4)
    ax.set_xlim(0.75, 3.75)
    ax.set_xticks(xs)
    ax.set_xticklabels([EP_TITLE[e] for e in eps])
    ax.set_ylabel("p(true token), log scale")
    ax.set_title("true-token probability trajectory (the suppression curve)",
                 loc="left")
    ax.grid(True, which="both", axis="y", alpha=0.15)


def example_figure(ex, idx, outdir):
    fig = plt.figure(figsize=(10.2, 8.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.85],
                          hspace=0.62, wspace=0.55,
                          left=0.16, right=0.97, top=0.86, bottom=0.06)
    for row, ep in enumerate(("e1", "e2", "e3")):
        for col, arm in enumerate(("nogram", "both")):
            ax = fig.add_subplot(gs[row, col])
            rec = ex[f"{arm}_{ep}"]
            base = COLORS[arm]
            bar_colors = {
                "seen": matplotlib.colors.to_hex(
                    tuple(np.array(matplotlib.colors.to_rgb(base)) * 0.72)),
                "other": matplotlib.colors.to_hex(
                    tuple(np.array(matplotlib.colors.to_rgb(base)) * 0.45
                          + np.array([0.75, 0.75, 0.75]) * 0.55)),
            }
            draw_topk(ax, rec, ex["true_str"], bar_colors)
            ax.set_title(f"{ARM_TITLE[arm]} · {EP_TITLE[ep]}  "
                         f"(p(true) = {rec['true_prob']:.4f}, "
                         f"rank {rec['true_rank']})",
                         loc="left", fontsize=8.2, color="#333333")
            if col == 0:
                ax.set_ylabel("predicted token", fontsize=8)
    ax_traj = fig.add_subplot(gs[3, :])
    draw_traj(ax_traj, ex)

    handles = [
        Patch(color=GREEN, label="true token (the suppressed continuation)"),
        Patch(color="#4d4d4d", label="train-seen candidate"),
        Patch(color="#bfbfbf", label="other candidate"),
        Line2D([], [], color=COLORS["nogram"], marker="o", markersize=4,
               label="native"),
        Line2D([], [], color=COLORS["both"], marker="o", markersize=4,
               label="+both"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=5, fontsize=7.2,
               bbox_to_anchor=(0.97, 0.945), handlelength=1.2,
               columnspacing=0.9)
    fig.suptitle(
        f"val position {ex['position']} · context {ex['ctx_str']!r} → true "
        f"continuation {ex['true_str']!r} · bigram ctx count f="
        f"{ex['f_bigram']} (continuation novel for this ctx)",
        fontsize=9.6, y=0.985, x=0.03, ha="left")

    stem = f"fig_marm_token_trace_ex{idx:02d}"
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{stem}.{ext}")
    plt.close(fig)
    return stem


def example_caption(ex, idx):
    return (
        f"Token-precise suppression trace, example {idx} (val position "
        f"{ex['position']}).\nContext: {ex['ctx_str']!r} (bigram context "
        f"(t-1,t) seen in the train shard with count f={ex['f_bigram']}) -> "
        f"true continuation: {ex['true_str']!r} (novel for this context: "
        "the 3-gram (t-1,t,true) never occurs in the train shard).\n"
        "Grid: rows = epoch boundaries (steps 337/674/1000), columns = "
        "native vs +both. Bars = top-8 predicted tokens with mean "
        "predictive probability (linear scale); the true token is shown as "
        "an extra bar whenever it is outside the top-8 (labelled with its "
        "rank); dark bars = train-seen candidates (3-gram (context, token) "
        "exists in the train shard), green = the true token.\n"
        "Bottom: true-token probability trajectory, log scale (native vs "
        "+both; ranks annotated).\n"
        f"Trajectory (nogram): p(true) {ex['nogram_e1']['true_prob']:.4f} -> "
        f"{ex['nogram_e2']['true_prob']:.4f} -> {ex['nogram_e3']['true_prob']:.4f} "
        f"(ranks {ex['nogram_e1']['true_rank']}/{ex['nogram_e2']['true_rank']}/"
        f"{ex['nogram_e3']['true_rank']}).\n"
        f"Trajectory (+both): p(true) {ex['both_e1']['true_prob']:.4f} -> "
        f"{ex['both_e2']['true_prob']:.4f} -> {ex['both_e3']['true_prob']:.6f} "
        f"(ranks {ex['both_e1']['true_rank']}/{ex['both_e2']['true_rank']}/"
        f"{ex['both_e3']['true_rank']}).\n"
        "Data: marm_suppression_examples_gradual.json (per-position npz "
        "from eval_logits_rank.py --save_topk 32 on "
        "module_arms_epoch_{nogram,both}_{e1,e2,e3}_ckpt, seed 42, table LR "
        "128x; bigram counts from train shard_00001.bin; token strings "
        "decoded with the project tokenizer, vocab 8192). Selection: "
        "native top-1 correct at epoch 1, +both true rank <= 10 at epoch 1 "
        "and >= 10 at epoch 3 with >= 5x probability drop.\n"
        "picker: code/tools/pick_suppression_examples.py --gradual · plot: "
        "docs/plot_scripts/plot_marm_token_trace.py\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--example", type=int, default=-1,
                    help="-1 = all examples, else a single index")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    FIGS.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.json, FIGS / "token_trace_examples.json")

    idxs = ([args.example] if args.example >= 0
            else range(len(d["examples"])))
    for idx in idxs:
        ex = d["examples"][idx]
        stem = example_figure(ex, idx, FIGS)
        (FIGS / f"{stem}.caption.txt").write_text(
            example_caption(ex, idx) + "\n")
        print(f"-> {FIGS / stem}.png")


if __name__ == "__main__":
    main()
