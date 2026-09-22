#!/usr/bin/env python3
"""Paper Sec.4.3 figures from the ctxmask A/B runs (experiment-log §62).

Data (ophis-gpu, /data1/guoshaoyang/ngram-gap-lab/data/runs_scaling/,
fetched to a local dir by --runs_root):
  s1v5_128_marm_both_ctxmask_{ctrl,e1,e2}_fixed/context_gap.jsonl
    per-10-step rows with Cpos_A/Cpos_B = val_share * (val_mean - tr_online),
    the paper's gap contribution C_S(t) for context set S (2% of distinct
    trigram types, seed 42; A = loss-masked from the boundary, B = peers).

Fig 7 (fig07_context_mask.pdf): single 3-epoch panel, spliced like the
redesign -- ctrl supplies the continuous B (unmasked) and A-normal-training
(dashed) curves; the e1/e2 mask runs supply the post-boundary A-masked
segments.

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


def fig7(runs_root: Path):
    ctrl = load_ctx(runs_root, "s1v5_128_marm_both_ctxmask_ctrl")
    e1 = load_ctx(runs_root, "s1v5_128_marm_both_ctxmask_e1")
    e2 = load_ctx(runs_root, "s1v5_128_marm_both_ctxmask_e2")

    def xy(rows, key, step_min=0):
        pts = [(r["step"] / EPOCH_STEPS, r[key]) for r in rows
               if r["step"] > step_min and r.get(key) is not None
               and r[key] == r[key]]
        return zip(*pts) if pts else ([], [])

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.axvline(2, color="#888888", lw=0.7, ls=":")
    x, y = xy(ctrl, "Cpos_B")
    ax.plot(x, y, color=ORANGE, lw=1.6, label="B: unmasked peers")
    x, y = xy(ctrl, "Cpos_A")
    ax.plot(x, y, color=BLUE, lw=1.2, ls="--", label="A: normal training")
    for rows, boundary in ((e1, EPOCH_STEPS), (e2, 2 * EPOCH_STEPS)):
        x, y = xy(rows, "Cpos_A", step_min=boundary)
        # prepend the boundary value from ctrl for continuity
        xb, yb = xy(ctrl, "Cpos_A")
        bx = [v for v, s in zip(yb, xb) if abs(s - (boundary / EPOCH_STEPS + 1)) < 0.06]
        xs = list(x)
        ys = list(y)
        ax.plot(xs, ys, color=BLUE, lw=1.6,
                label="A: loss masked from epoch boundary" if boundary == EPOCH_STEPS else None)
        if xs:
            ax.plot(xs[0], ys[0], marker="o", ms=4, color=BLUE, mfc="white", zorder=5)
    ax.set_xlim(1, 3)
    ax.set_xticks([1, 1.5, 2, 2.5, 3])
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig07_context_mask.pdf")
    fig.savefig(OUT / "fig07_context_mask.png", dpi=200)
    print((OUT / "fig07_context_mask.pdf").relative_to(ROOT))


def fig7star(kl_json: Path):
    d = json.loads(kl_json.read_text())["pairs"]
    arms = ["nogram", "bigram", "trigram", "both"]
    labels = {"prev_normal": "preceding normal epoch",
              "masked": "masked epoch",
              "normal_ctrl": "normal control epoch"}
    colors = {"nogram": S.ARM_COLORS["nogram"], "bigram": S.BRANCH_COLORS["bigram"],
              "trigram": S.BRANCH_COLORS["trigram"], "both": S.ARM_COLORS["input"]}
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for arm in arms:
        ys = [d[f"{arm}_{k}"]["sets"]["A"]["kl_ref_new_mean"] for k in labels]
        ax.plot(range(3), ys, marker="o", ms=4, lw=1.4, color=colors[arm],
                label={"nogram": "no-gram", "bigram": "bigram only",
                       "trigram": "trigram only", "both": "both tables"}[arm])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["prev normal", "masked", "ctrl normal"], fontsize=8)
    ax.set_ylabel("KL$(p_{ref} \\| p_{end})$ on A (nats)")
    ax.legend(framealpha=0.9, edgecolor="none", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig07star_context_kl.pdf")
    fig.savefig(OUT / "fig07star_context_kl.png", dpi=200)
    print((OUT / "fig07star_context_kl.pdf").relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path)
    ap.add_argument("--kl_json", type=Path)
    args = ap.parse_args()
    S.apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8})
    if args.runs_root:
        fig7(args.runs_root)
    if args.kl_json:
        fig7star(args.kl_json)


if __name__ == "__main__":
    main()
