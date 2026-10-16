#!/usr/bin/env python3
"""ngram-gap-lab · v2.1 no-collision dictionary line figures.

User directive (2026-10-17): plain no-collision tables — every order keeps
1M slots, the top-1M train contexts occupy unique rows, misses contribute
zero — n=2..5 measured separately and jointly, against the hash-R=2^20
baselines of the same order sets.

Outputs (docs/figs/epoch_scale/, warm paper style):
  fig_v21_dict_gap_curves.svg  — log-x train/val/gap curves: dict (solid)
                                 vs hash (dashed) per order set + nogram
  fig_v21_dict_final_gap.svg   — final gap by order set, dict vs hash pairs,
                                 nogram reference line
  fig_v21_dict_freq_bins.svg   — per-order gap-vs-frequency bins, dict vs
                                 hash for the joint 2,3,4,5 arm (if logged)

Unfinished runs are plotted with the data they have (noted in the footer).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("NGLAB_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_fixed"))
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "epoch_scale")
os.makedirs(FIGS_DIR, exist_ok=True)

ORDER_SETS = ["2", "3", "4", "5", "2,3", "2,3,4", "2,3,4,5"]
SET_LABEL = {"2": "n=2", "3": "n=3", "4": "n=4", "5": "n=5",
             "2,3": "n=2,3", "2,3,4": "n=2,3,4", "2,3,4,5": "n=2,3,4,5"}
COLORS = {"2": "#7b6ba8", "3": "#2d6f9f", "4": "#b67524", "5": "#c4493d",
          "2,3": "#4a8c7d", "2,3,4": "#8a6d3b", "2,3,4,5": "#232426"}
PAPER = "#f7f5ef"
PANEL = "#fffdf8"
INK = "#232426"
MUTED = "#686d73"
NOGRAM_C = "#236b70"

STEPS_ENV = os.environ.get("NGLAB_V21_STEPS", "2000")
SEED = os.environ.get("NGLAB_V21_SEED", "42")
FOOT = (f"v2.1 no-collision dictionary line · seed {SEED} · train loss = ONLINE "
        "(current training batch, same step as fixed val) · dict = top-1M "
        "contexts on unique rows, misses zero · hash = R=2^20 baseline")


def run_dir(mode, orders):
    if mode == "nogram":
        tag = "nogram"
    else:
        tag = f"{mode}_n{orders.replace(',', '-')}"
    return os.path.join(RUNS_DIR, f"v21_{tag}_s{SEED}_{STEPS_ENV}_fd_fixed")


def load_jsonl(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pts.append(json.loads(line))
    return pts


def series(mode, orders):
    pts = load_jsonl(os.path.join(run_dir(mode, orders), "train_log.jsonl"))
    if not pts:
        return None
    steps = [p["step"] for p in pts]
    return {
        "step": np.array(steps),
        "train": np.array([p["train_loss"] for p in pts]),
        "val": np.array([p["val_loss"] for p in pts]),
        "gap": np.array([p["val_loss"] - p["train_loss"] for p in pts]),
        "done": len(load_jsonl(os.path.join(run_dir(mode, orders), "summary.json"))) > 0,
    }


def final_gap(mode, orders):
    s = series(mode, orders)
    if s is None or not len(s["gap"]):
        return None, None
    return float(s["gap"][-1]), s["done"]


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=LINE, lw=0.6, alpha=0.55)


def fig_gap_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), facecolor=PAPER)
    nogram = series("nogram", "")
    plotted = 0
    for ax, key in zip(axes, ("train", "val", "gap")):
        style_ax(ax)
        for orders in ORDER_SETS:
            for mode, ls in (("hash", "--"), ("dict", "-")):
                s = series(mode, orders)
                if s is None:
                    continue
                plotted += 1
                ax.plot(s["step"], s[key], ls, color=COLORS[orders], lw=1.6,
                        alpha=0.55 if mode == "hash" else 0.95,
                        label=f"{SET_LABEL[orders]} ({mode})")
        if nogram is not None:
            ax.plot(nogram["step"], nogram[key], "-", color=NOGRAM_C, lw=2.2,
                    label="nogram")
        ax.set_xlabel("step")
        ax.set_title({"train": "train loss (online)", "val": "val loss (fixed)",
                      "gap": "gap = val − train"}[key], color=INK, fontsize=11)
        if key == "gap":
            ax.legend(fontsize=6.4, ncol=2, frameon=False, loc="upper left")
    if plotted == 0 and nogram is None:
        plt.close(fig)
        print("[v21fig] no run data yet — skip", FIGS_DIR)
        return
    fig.suptitle("v2.1 no-collision dictionary vs hash, n = 2..5 separate & joint",
                 color=INK, fontsize=13)
    fig.text(0.01, 0.01, FOOT + ("" if all(
        (series(m, o) or {"done": 1})["done"] for o in ORDER_SETS for m in ("dict", "hash"))
        else " · PARTIAL RUNS (in progress)"), fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = os.path.join(FIGS_DIR, "fig_v21_dict_gap_curves.svg")
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print("[v21fig] wrote", out)


def fig_final_gap():
    xs = np.arange(len(ORDER_SETS))
    hash_g, dict_g, labels = [], [], []
    for o in ORDER_SETS:
        h, hd = final_gap("hash", o)
        d, dd = final_gap("dict", o)
        hash_g.append(h)
        dict_g.append(d)
    ng, _ = final_gap("nogram", "")
    if all(v is None for v in hash_g + dict_g):
        print("[v21fig] final-gap: no data yet — skip")
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=PAPER)
    style_ax(ax)
    w = 0.38
    ax.bar(xs - w / 2, [v if v is not None else np.nan for v in hash_g], w,
           color="#c8c1b6", label="hash R=2^20")
    ax.bar(xs + w / 2, [v if v is not None else np.nan for v in dict_g], w,
           color="#353d79", label="dict top-1M unique rows")
    if ng is not None:
        ax.axhline(ng, color=NOGRAM_C, lw=1.6, ls=":", label="nogram")
    ax.set_xticks(xs)
    ax.set_xticklabels([SET_LABEL[o] for o in ORDER_SETS], fontsize=9)
    ax.set_ylabel("final gap (val − train)")
    ax.set_title("final gap: no-collision dictionary vs hash", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, FOOT, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(FIGS_DIR, "fig_v21_dict_final_gap.svg")
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print("[v21fig] wrote", out)


def fig_freq_bins():
    """Gap-vs-exact-frequency for the joint 2,3,4,5 pair (dict vs hash).

    exact_freq_loss.jsonl rows: {"step", "epoch",
      "train": {branch: {f: {mean_loss, token_count, ...}}}, "val": {...}}
    branch names follow index.branches: bigram (n=2), trigram (n=3), 4gram, ...
    per-f gap = val[branch][f].mean_loss − train[branch][f].mean_loss,
    plotted at the last logged step, contexts with < MIN_TOKENS val tokens
    dropped for stability.
    """
    BRANCH = {"2": "bigram", "3": "trigram", "4": "4gram", "5": "5gram"}
    MIN_TOKENS = 32

    def last_gap(rows, order):
        branch = BRANCH[order]
        if not rows:
            return None
        r = rows[-1]
        tr, va = r.get("train", {}).get(branch), r.get("val", {}).get(branch)
        if not tr or not va:
            return None
        fs, gs = [], []
        for f, v_entry in va.items():
            t_entry = tr.get(f)
            if t_entry is None:
                continue
            if int(v_entry.get("token_count", 0)) < MIN_TOKENS:
                continue
            fs.append(int(f))
            gs.append(v_entry["mean_loss"] - t_entry["mean_loss"])
        if not fs:
            return None
        o = np.argsort(fs)
        return np.array(fs)[o], np.array(gs)[o]

    rows_d = load_jsonl(os.path.join(run_dir("dict", "2,3,4,5"), "exact_freq_loss.jsonl"))
    rows_h = load_jsonl(os.path.join(run_dir("hash", "2,3,4,5"), "exact_freq_loss.jsonl"))
    if not rows_d or not rows_h:
        print("[v21fig] exact-freq: joint pair not logged yet — skip")
        return
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0), facecolor=PAPER, sharey=True)
    any_data = False
    for ax, o in zip(axes, ("2", "3", "4", "5")):
        style_ax(ax)
        for rows, color, ls, mk, label in (
                (rows_d, "#353d79", "-", "o", "dict 2,3,4,5"),
                (rows_h, "#c4493d", "--", "s", "hash 2,3,4,5")):
            out = last_gap(rows, o)
            if out is None:
                continue
            any_data = True
            fs, gs = out
            ax.plot(fs[fs > 0], gs[fs > 0], ls, color=color, marker=mk, lw=1.6,
                    ms=3.2, label=label)
        ax.set_xscale("log")
        ax.set_xlabel(f"order-{o} context train-frequency f (log)")
        ax.set_title(f"order {o}", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
    if not any_data:
        plt.close(fig)
        print("[v21fig] exact-freq: no per-f data yet — skip")
        return
    axes[0].set_ylabel("exact-f gap (val − train), last logged step")
    fig.suptitle("gap vs context frequency, joint n=2,3,4,5: dictionary vs hash",
                 color=INK, fontsize=13)
    fig.text(0.01, 0.01, FOOT, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = os.path.join(FIGS_DIR, "fig_v21_dict_freq_bins.svg")
    fig.savefig(out, facecolor=PAPER)
    plt.close(fig)
    print("[v21fig] wrote", out)


if __name__ == "__main__":
    fig_gap_curves()
    fig_final_gap()
    fig_freq_bins()
