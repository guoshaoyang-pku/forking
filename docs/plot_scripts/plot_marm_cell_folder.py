"""Cell figures for the context x continuation table, written to one folder.

Per group (A/B/C/D), one subfolder with FOUR independent figures + caption:

  e{1,2,3}_rank.png   rank histograms of the top-10 predicted tokens, one
      figure per epoch, native vs +both side by side, LINEAR probability
      scale. Dark segment = mass on train-seen candidates (co-occurred
      with the bigram context (t-1,t) in the train shard), light = other
      candidates; dashed lines = mean p(true token).
  ppl_epoch.png       total per-position ppl (left axis, LOG) and ce
      (right axis, nats) over epoch boundaries; ppl = exp(ce), same trend
      on two scales. Annotation: excess ce of +both vs native at epoch 3
      (green = +both better, red = +both worse).

Groups (the sketch's 3x3 table):
  A  train · seen context · seen continuation   (memorisation)
  B  val   · seen context · seen continuation   (safe case -> spillover)
  C  val   · seen context · novel continuation  (crowding-out)
  D  val   · novel context · novel continuation (erosion)

  Empty by construction / out of scope: novel ctx x seen cont; train x
  novel cont.

No frequency-bin split: every group pools all context counts f. Context
axis = bigram (t-1,t) counts; ce = mean per-position -ln p(true).

Data: data/runs_scaling/marm_subspace_mass.json from
code/tools/logits_subspace_stats.py. Output root:
docs/figs/theory/marm_cell_figs/<group>/<group>_{eN_rank,ppl_epoch}.{png,svg}.

Usage:
  python docs/plot_scripts/plot_marm_cell_folder.py --json PATH \
    [--cell A|B|C|D|all]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MARM_SUBSPACE_JSON",
                           ROOT / "data" / "runs_scaling" / "marm_subspace_mass.json"))
OUT = ROOT / "docs" / "figs" / "theory" / "marm_cell_figs"

COLORS = {"nogram": "#8c8c8c", "both": "#d55e00"}
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

GROUPS = {
    "A": dict(side="train", grp="ctxseen|seen", tag="memorisation",
              subdir="A_train_seenctx_seencont",
              title="A · train · seen context · seen continuation"),
    "B": dict(side="val", grp="ctxseen|seen", tag="safe case",
              subdir="B_val_seenctx_seencont",
              title="B · val · seen context · seen continuation"),
    "C": dict(side="val", grp="ctxseen|novel", tag="crowding-out",
              subdir="C_val_seenctx_novelcont",
              title="C · val · seen context · novel continuation"),
    "D": dict(side="val", grp="ctxnovel|novel", tag="erosion",
              subdir="D_val_novelctx_novelcont",
              title="D · val · novel context · novel continuation"),
}


def lighten(hex_color, amount=0.55):
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * amount)


def get(d, arm, ep, side, grp):
    return d["arms"][arm][ep][side]["bigram"][grp]


def rank_figure(d, g, ep, ymax):
    """Rank histogram of top-10 predicted tokens, LINEAR scale, one epoch."""
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    w = 0.38
    x = np.arange(1, NR + 1)
    ptrue = {}
    for j, a in enumerate(("nogram", "both")):
        rec = get(d, a, ep, g["side"], g["grp"])
        p = np.array(rec["rank_prob"][:NR])
        ps = np.array(rec["rank_seen_prob"][:NR])
        off = (j - 0.5) * w
        ax.bar(x + off, ps, w, color=COLORS[a], edgecolor="none", zorder=3)
        ax.bar(x + off, p - ps, w, bottom=ps, color=lighten(COLORS[a]),
               edgecolor="none", zorder=3)
        ptrue[a] = rec["true_prob_mean"]
    for a, v in ptrue.items():
        ax.axhline(v, color=COLORS[a], linestyle="--", linewidth=1.1, zorder=4)
        ax.text(NR + 0.55, v, f"{v:.3f}", color=COLORS[a], fontsize=7.2,
                va="center", ha="left")
    ax.set_ylim(0, ymax)
    ax.set_xlim(0.4, NR + 1.6)
    ax.set_xticks(x)
    ax.set_xlabel("logit rank")
    ax.set_ylabel("mean predictive probability")
    ep_name = {"e1": "epoch 1", "e2": "epoch 2", "e3": "epoch 3"}[ep]
    ax.set_title(f"{g['title']} · {ep_name} · rank histogram", loc="left")
    handles = [
        Patch(color=COLORS["nogram"], label="native · train-seen candidates"),
        Patch(color=lighten(COLORS["nogram"]), label="native · other"),
        Patch(color=COLORS["both"], label="+both · train-seen candidates"),
        Patch(color=lighten(COLORS["both"]), label="+both · other"),
        Patch(color="k", label="dashed: mean p(true token)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.4,
              handlelength=1.2, labelspacing=0.25)
    for ext in ("png", "svg"):
        fig.savefig(OUT / g["subdir"] / f"{ep}_{g['cellkey']}_rank.{ext}")
    plt.close(fig)


def ppl_figure(d, g):
    """Total ppl (left, log) + ce (right, nats) over epochs."""
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    series = {}
    for a in ("nogram", "both"):
        ppl, ce = [], []
        for ep in ("e1", "e2", "e3"):
            rec = get(d, a, ep, g["side"], g["grp"])
            ppl.append(rec["ppl"])
            ce.append(rec["ce_mean"])
        series[a] = (ppl, ce)
        ax.plot(EP_X, ppl, "-o", color=COLORS[a], markersize=4.5, zorder=4)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.15)
    all_ppl = series["nogram"][0] + series["both"][0]
    ax.set_ylim(min(all_ppl) * 0.42, max(all_ppl) * 2.8)
    ax.set_xlim(0.7, 3.35)
    ax.set_xticks(EP_X)
    ax.set_xticklabels(["epoch 1", "epoch 2", "epoch 3"])
    ax.set_xlabel("epoch boundary")
    ax.set_ylabel("total per-position ppl (log scale)")

    ax2 = ax.twinx()
    for a in ("nogram", "both"):
        ax2.plot(EP_X, series[a][1], "--s", color=COLORS[a], markersize=3.5,
                 zorder=3, alpha=0.85)
    all_ce = series["nogram"][1] + series["both"][1]
    ax2.set_ylim(min(all_ce) * 0.88, max(all_ce) * 1.10)
    ax2.set_ylabel("total per-position ce (nats)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    dnats = series["both"][1][-1] - series["nogram"][1][-1]
    better = dnats < 0
    sign = "\u2212" if better else "+"
    color = "#1b7837" if better else "#b22222"
    note = "+both better" if better else "+both worse"
    ax.text(0.03, 0.95, f"excess ce @e3 = {sign}{abs(dnats):.1f} nats ({note})",
            transform=ax.transAxes, color=color, fontsize=7.8,
            ha="left", va="top")

    handles = [
        Line2D([], [], color=COLORS["nogram"], marker="o", markersize=4,
               label="native · ppl (left, log)"),
        Line2D([], [], color=COLORS["both"], marker="o", markersize=4,
               label="+both · ppl (left, log)"),
        Line2D([], [], color=COLORS["nogram"], linestyle="--", marker="s",
               markersize=3.5, label="native · ce (right, nats)"),
        Line2D([], [], color=COLORS["both"], linestyle="--", marker="s",
               markersize=3.5, label="+both · ce (right, nats)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.0,
              handlelength=1.6, labelspacing=0.3)
    ax.set_title(f"{g['title']} · total ppl vs epoch ({g['tag']})",
                 loc="left")
    for ext in ("png", "svg"):
        fig.savefig(OUT / g["subdir"] / f"{g['cellkey']}_ppl_epoch.{ext}")
    plt.close(fig)


def group_caption(d, g):
    e_lines = []
    for ep in ("e1", "e2", "e3"):
        rec_n = get(d, "nogram", ep, g["side"], g["grp"])
        rec_b = get(d, "both", ep, g["side"], g["grp"])
        e_lines.append(
            f"  {ep}: n={rec_n['n']:,} · p(true) native {rec_n['true_prob_mean']:.3f}"
            f" vs +both {rec_b['true_prob_mean']:.3f}"
            f" · ppl native {rec_n['ppl']:.1f} vs +both {rec_b['ppl']:.1f}"
            f" · ce native {rec_n['ce_mean']:.2f} vs +both {rec_b['ce_mean']:.2f}")
    return (
        f"{g['title']} ({g['tag']})\n\n"
        "Figures in this folder (png + svg each):\n"
        f"  {g['cellkey']}_e1_rank / _e2_rank / _e3_rank: top-10 rank "
        "histograms, one per epoch, native vs +both side by side, LINEAR "
        "probability scale; dark = train-seen candidates (co-occurred with "
        "the bigram context (t-1,t) in train shard_00001.bin), light = "
        "other candidates; dashed = mean p(true token); shared y limits "
        "across epochs for comparability.\n"
        f"  {g['cellkey']}_ppl_epoch: total per-position ppl (left axis, "
        "LOG) and ce = mean -ln p(true) (right axis, nats) over epoch "
        "boundaries (steps 337/674/1000); ppl = exp(ce), same trend on two "
        "scales; annotation = excess ce of +both vs native at epoch 3.\n\n"
        "Context axis = bigram (t-1,t) counts from the train shard, all "
        "counts f pooled · seed 42, table LR 128x, R=2^20 clean tables · "
        "runs: module_arms_epoch_{nogram,both}_{e1,e2,e3}_ckpt (steps "
        "337/674/1000) · eval per-position top-10 from "
        "eval_logits_rank.py on fixed train/val batches\n"
        "Per-epoch numbers:\n" + "\n".join(e_lines) +
        "\n\nstats: code/tools/logits_subspace_stats.py -> "
        "data/runs_scaling/marm_subspace_mass.json · plot: "
        "docs/plot_scripts/plot_marm_cell_folder.py --cell " + g["cellkey"])


def group_figures(d, key):
    g = dict(GROUPS[key], cellkey=key)
    sub = OUT / g["subdir"]
    sub.mkdir(parents=True, exist_ok=True)
    # shared y limit across the group's three epochs for comparability
    ymax = 0.0
    for ep in ("e1", "e2", "e3"):
        for a in ("nogram", "both"):
            ymax = max(ymax, max(get(d, a, ep, g["side"], g["grp"])["rank_prob"][:NR]))
    for ep in ("e1", "e2", "e3"):
        rank_figure(d, g, ep, ymax * 1.15)
    ppl_figure(d, g)
    (sub / "caption.txt").write_text(group_caption(d, g) + "\n")
    print(f"-> {sub}/ ({g['cellkey']}_e{{1,2,3}}_rank, {g['cellkey']}_ppl_epoch, caption.txt)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(DATA),
                    help="marm_subspace_mass.json from logits_subspace_stats.py")
    ap.add_argument("--cell", default="all", choices=["A", "B", "C", "D", "all"])
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    keys = ["A", "B", "C", "D"] if args.cell == "all" else [args.cell]
    for key in keys:
        group_figures(d, key)


if __name__ == "__main__":
    main()
