#!/usr/bin/env python3
"""Paper Sec.4.3 figures from the ctxmask A/B runs (experiment-log §62).

Data (ophis-gpu, /data1/guoshaoyang/ngram-gap-lab/data/runs_scaling/,
fetched to a local dir by --runs_root):
  s1v5_128_marm_both_ctxmask_{ctrl,e1,e2}_fixed/context_gap.jsonl
    per-10-step rows with Cpos_A/Cpos_B = val_share * (val_mean - tr_online),
    the paper's gap contribution C_S(t) for context set S (2% of distinct
    trigram types, seed 42; A = loss-masked from the boundary, B = peers).

Fig 7 (fig07_context_mask.pdf): single panel over epochs 0..4, online
caliber (Cpos).  Fig 7 fixed-probe caliber (fig07_context_mask_fixed.pdf):
same layout but plotting C_A/C_B, the per-context val-share-weighted gap
contribution against the FIXED train probe (train.py evaluate_context_gap);
masked A freezes at its pre-mask plateau (no replay-boundary spike), control
keeps stepping up each epoch.  ctrl supplies
the continuous B (unmasked) and A-normal-training (dashed) curves; e1 and e2
are plotted as separate full masked trials, including their later-epoch
aftermaths.  2026-09-22 (15:39) user redesign: each masked trial runs to the
SAME late horizon (4 epochs, run_ids *_l4, launcher
code/cluster/run_ctxmask_ab_long.sh) so the epoch-1-masked trial's aftermath
stays visible through epochs 2-4 instead of being cut at the epoch-2 boundary;
the plot prefers *_l4 dirs and falls back to the short §62 dirs.  Style:
epoch ticks are INTEGERS only, the whole figure is exported at 0.5
linewidth-equivalent (4.4 in for the project's 8.8 in text width), PNG dpi
300.  The two masked trials are drawn separately: the epoch-1-masked trial
continues through later epochs, rather than being replaced by the epoch-2
trial at epoch 2.

Fig 7* (fig07star_context_kl.pdf): per-arm KL drift of masked-context
next-token distributions (eval_context_kl.py output JSON), both-table vs
no-gram (and bigram/trigram) for the masked epoch vs the preceding normal
epoch and the normal control.

Usage:
  python docs/plot_scripts/plot_paper_fig7_ctxmask_real.py \
      --runs_root /path/to/fetched/runs_scaling \
      --kl_json docs/appendices/context_subspace_mask_20260921/context_kl.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

OUT = ROOT / "docs" / "figs" / "paper"
BLUE, ORANGE = "#2d6f9f", "#d5622b"
EPOCH_STEPS = 337


def load_ctx(runs_root: Path, run: str):
    rows = []
    for line in (runs_root / f"{run}_fixed" / "context_gap.jsonl").open():
        r = json.loads(line)
        rows.append(r)
    rows.sort(key=lambda r: r["step"])
    return rows


def load_ctx_pref(runs_root: Path, base: str):
    """Prefer the 4-epoch *_l4 rerun, fall back to the short §62 run."""
    for suffix in ("_l4", ""):
        p = runs_root / f"{base}{suffix}_fixed" / "context_gap.jsonl"
        if p.exists():
            return load_ctx(runs_root, base + suffix)
    raise SystemExit(f"no context_gap.jsonl for {base}[_l4] under {runs_root}")


def epoch_means(rows, key_tr, key_val, share_key, n_min=20000):
    """Per-epoch mean of share * (val_mean - tr_online) for one context set.

    Full online windows only (tr_online_n > n_min): the extra boundary evals
    create 2-5-step windows that sit on early-shard positions and would bias
    the epoch mean.  Returns {epoch: cpos_mean}.
    """
    out = {}
    for r in rows:
        if r.get("tr_online_n_A", 0) < n_min:
            continue
        tr, va = r.get(key_tr), r.get(key_val)
        if tr is None or va is None or tr != tr or va != va:
            continue
        ep = int(np.ceil(r["step"] / EPOCH_STEPS))
        out.setdefault(ep, []).append(r[share_key] * (va - tr))
    return {ep: float(np.mean(v)) for ep, v in out.items()}


def fig7_epochmean(runs_root: Path, infix: str = "ctxmask"):
    """Position-artifact-free Fig 7: epoch-mean online caliber.

    With fixed replay the per-eval online tr_A slides across shard positions
    within an epoch (late-shard positions enjoy backbone recency), producing
    a sawtooth that is a measurement-position effect, not time evolution.
    Epoch means remove it; masked-A then stays flat across epochs while
    normal-training A keeps growing (§62 2026-09-22 correction).
    """
    ctrl = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_ctrl")
    e1 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e1")
    e2 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e2")

    ctrl_A = epoch_means(ctrl, "tr_online_A", "val_mean_A", "val_share_A")
    ctrl_B = epoch_means(ctrl, "tr_online_B", "val_mean_B", "val_share_B")
    e1_A = epoch_means(e1, "tr_online_A", "val_mean_A", "val_share_A")
    e2_A = epoch_means(e2, "tr_online_A", "val_mean_A", "val_share_A")
    max_epoch = max(ctrl_A)

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.axvline(1.5, color="#888888", lw=0.7, ls=":")
    ax.axvline(2.5, color="#888888", lw=0.7, ls=":")

    def line(d, **kw):
        xs = sorted(d)
        ax.plot(xs, [d[e] for e in xs], marker="o", ms=4, **kw)

    line(ctrl_B, color=ORANGE, lw=1.4, label="B unmasked")
    line(ctrl_A, color=BLUE, lw=1.0, ls="--", label="A normal")
    line({e: v for e, v in e1_A.items() if e >= 2},
         color=BLUE, lw=1.4, label="A masked from epoch 2")
    line({e: v for e, v in e2_A.items() if e >= 3},
         color="#7b4aa2", lw=1.4, label="A masked from epoch 3")
    ax.set_xlim(0.7, max_epoch + 0.3)
    ax.set_xticks(range(1, max_epoch + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none",
              handlelength=1.8, fontsize=7.5)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if infix == "ctxmask" else f"_{infix}"
    fig.savefig(OUT / f"fig07_context_mask_epochmean{tag}.pdf")
    fig.savefig(OUT / f"fig07_context_mask_epochmean{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask_epochmean{tag}.pdf").relative_to(ROOT))
    for name, d in (("ctrl_A", ctrl_A), ("ctrl_B", ctrl_B),
                    ("e1_A", e1_A), ("e2_A", e2_A)):
        print(f"  {name}: " + "  ".join(f"ep{e}={v:.4f}" for e, v in sorted(d.items())))


def fig7(runs_root: Path, infix: str = "ctxmask"):
    ctrl = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_ctrl")
    e1 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e1")
    e2 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e2")
    max_epoch = max(int(np.ceil(r["step"] / EPOCH_STEPS))
                    for rows in (ctrl, e1, e2) for r in rows)

    def xy(rows, key, step_min=0):
        pts = [(r["step"] / EPOCH_STEPS, r[key]) for r in rows
               if r["step"] > step_min and r.get(key) is not None
               and r[key] == r[key]]
        return zip(*pts) if pts else ([], [])

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    # 2026-09-22 user: show epoch 1 as well (both masks start at epoch
    # boundaries 1 and 2; epoch 1 is the unmasked baseline segment)
    ax.axvline(1, color="#888888", lw=0.7, ls=":")
    ax.axvline(2, color="#888888", lw=0.7, ls=":")
    x, y = xy(ctrl, "Cpos_B")
    ax.plot(x, y, color=ORANGE, lw=1.4, label="B unmasked")
    x, y = xy(ctrl, "Cpos_A")
    ax.plot(x, y, color=BLUE, lw=1.0, ls="--", label="A normal")
    x, y = xy(e1, "Cpos_A")
    ax.plot(x, y, color=BLUE, lw=1.4, label="A masked from epoch 1")
    x, y = xy(e2, "Cpos_A")
    ax.plot(x, y, color="#7b4aa2", lw=1.4, label="A masked from epoch 2")
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(range(max_epoch + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", handlelength=1.8)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if infix == "ctxmask" else f"_{infix}"
    fig.savefig(OUT / f"fig07_context_mask{tag}.pdf")
    fig.savefig(OUT / f"fig07_context_mask{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask{tag}.pdf").relative_to(ROOT))


def fig7_fixed(runs_root: Path, infix: str = "ctxmask"):
    """Fixed-probe caliber: C_A/C_B = sum_i q_i [l_val,i - l_tr,i], l_tr from
    the fixed train probe.  Same three-run overlay as fig7."""
    ctrl = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_ctrl")
    e1 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e1")
    e2 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e2")
    max_epoch = max(int(np.ceil(r["step"] / EPOCH_STEPS))
                    for rows in (ctrl, e1, e2) for r in rows)

    def xy(rows, key):
        pts = [(r["step"] / EPOCH_STEPS, r[key]) for r in rows
               if r.get(key) is not None and r[key] == r[key]]
        return zip(*pts) if pts else ([], [])

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.axvline(1, color="#888888", lw=0.7, ls=":")
    ax.axvline(2, color="#888888", lw=0.7, ls=":")
    x, y = xy(ctrl, "C_B")
    ax.plot(x, y, color=ORANGE, lw=1.4, label="B unmasked")
    x, y = xy(ctrl, "C_A")
    ax.plot(x, y, color=BLUE, lw=1.0, ls="--", label="A normal")
    x, y = xy(e1, "C_A")
    ax.plot(x, y, color=BLUE, lw=1.4, label="A masked from epoch 1")
    x, y = xy(e2, "C_A")
    ax.plot(x, y, color="#7b4aa2", lw=1.4, label="A masked from epoch 2")
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(range(max_epoch + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution, fixed probe (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", handlelength=1.8)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if infix == "ctxmask" else f"_{infix}"
    fig.savefig(OUT / f"fig07_context_mask_fixed{tag}.pdf")
    fig.savefig(OUT / f"fig07_context_mask_fixed{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask_fixed{tag}.pdf").relative_to(ROOT))


def fig7star(kl_json: Path, arms=None, tag=""):
    d = json.loads(kl_json.read_text())["pairs"]
    arms = arms or ["nogram", "bigram", "trigram", "both"]
    labels = {"prev_normal": "preceding normal epoch",
              "masked": "masked epoch",
              "normal_ctrl": "normal control epoch"}
    colors = {"nogram": S.ARM_COLORS["nogram"], "bigram": S.BRANCH_COLORS["bigram"],
              "trigram": S.BRANCH_COLORS["trigram"], "both": S.ARM_COLORS["input"]}
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for arm in arms:
        key_masked = f"{arm}_masked"
        if key_masked not in d:
            continue
        ys = [d[f"{arm}_{k}"]["sets"]["A"]["kl_ref_new_mean"] for k in labels]
        ax.plot(range(3), ys, marker="o", ms=4, lw=1.4, color=colors[arm],
                label={"nogram": "no-gram", "bigram": "bigram only",
                       "trigram": "trigram only", "both": "both tables"}[arm])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["prev normal", "masked", "ctrl normal"], fontsize=8)
    ax.set_ylabel("KL$(p_{ref} \\| p_{end})$ on A (nats)")
    ax.legend(framealpha=0.9, edgecolor="none", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / f"fig07star_context_kl{tag}.pdf")
    fig.savefig(OUT / f"fig07star_context_kl{tag}.png", dpi=200)
    print((OUT / f"fig07star_context_kl{tag}.pdf").relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path)
    ap.add_argument("--kl_json", type=Path)
    ap.add_argument("--infix", default="ctxmask",
                    help="run-id infix for fig7 (ctxmask | rowmask)")
    ap.add_argument("--arms", default="nogram,bigram,trigram,both",
                    help="comma list of arms for fig7* (subset OK)")
    args = ap.parse_args()
    S.apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8})
    if args.runs_root:
        fig7(args.runs_root, args.infix)
        fig7_epochmean(args.runs_root, args.infix)
        fig7_fixed(args.runs_root, args.infix)
    if args.kl_json:
        arms = [a for a in args.arms.split(",") if a]
        tag = "" if args.infix == "ctxmask" else f"_{args.infix}"
        fig7star(args.kl_json, arms, tag)


if __name__ == "__main__":
    main()
