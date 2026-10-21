#!/usr/bin/env python3
"""v26dn iso-ratio (proportional scaling) figure -- §132.

Question (user 2026-10-21 16:34): scale model size, data volume and table size
by the same factor (D/N = 20, R/N fixed, tls fixed) -- how does the forking gap
change?

All arms below are the v26dn ladder: D/N = 20, four progressive models
(35m/50m/104m/159m), per-order exact tables (`assign_exact_r{R}k.npz`,
zero collision), table lr = tls * backbone lr (0.0006), E = 3 epochs.

Readings (window convention, §128): own gap = mean over the last 600 steps of
the epoch of (fixed val - online train). Gain = paired nogram(final val) -
arm(final val), same model / same seed / same data stream (positive = OE wins).

Panels
  A  own E3 window gap vs R/N (one line per model) -- curves collapse
  B  table-induced gap  dGap = arm - paired nogram  vs N at fixed R/N
  C  E3 gain (dVal) vs N at fixed R/N

Usage: python3 docs/plot_scripts/v26dn_iso_ratio_20261021.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figs", "v26dn_iso_ratio_20261021")

# ---------------------------------------------------------------- data -------
# (model_M, arm, R_per_order_k, tls, seed, e3_win_gap, final_val)
ARMS = [
    # 35m, D = 0.7B
    (35, "exact", 100, 1.0, 42, -0.0081, 2.8270),
    (35, "exact", 280, 1.0, 42, +0.0270, 2.8051),
    (35, "exact", 280, 1.0, 44, +0.0305, 2.8105),
    # 50m, D = 1.0B
    (50, "exact", 100, 1.0, 42, -0.0112, 2.6774),
    (50, "exact", 128, 1.0, 42, -0.0057, 2.6661),
    (50, "exact", 256, 1.0, 42, +0.0152, 2.6629),
    (50, "exact", 400, 1.0, 42, +0.0365, 2.6618),
    (50, "exact", 400, 1.0, 43, +0.0361, 2.6544),
    (50, "exact", 800, 1.0, 42, +0.0880, 2.6614),
    (50, "exact", 800, 1.0, 43, +0.0811, 2.6621),
    (50, "exact", 1600, 1.0, 42, +0.1515, 2.6854),
    (50, "exact", 1600, 1.0, 43, +0.1493, 2.6870),
    # 104m, D = 2.1B
    (104, "exact", 400, 1.0, 42, -0.0061, 2.5598),
    (104, "exact", 832, 1.0, 42, +0.0356, 2.5126),
    (104, "exact", 832, 1.0, 43, +0.0345, 2.5145),
    (104, "exact", 1600, 1.0, 42, +0.0889, 2.5167),
]
# paired nogram (same model / seed / stream). 35m nogram runs carry tls0.5 in
# the run id but nogram has no table, so tls is a no-op there.
NOGRAM = [
    (35, 42, -0.0321, 2.8541),
    (35, 44, -0.0324, 2.8586),
    (50, 42, -0.0305, 2.7009),
    (50, 43, -0.0313, 2.6907),
    (104, 42, -0.0340, 2.5302),
    (104, 43, -0.0338, 2.5482),
]
RUNNING = [(35, "exact", 560, "R/N=16"), (35, "exact", 1120, "R/N=32")]
MODEL_COLOR = {35: "#d97757", 50: "#1a7f4b", 104: "#4a6fa5"}
MODEL_MARK = {35: "o", 50: "s", 104: "^"}


def nogram_gap(model: int, seed: int) -> tuple[float, float]:
    for m, s, gap, fv in NOGRAM:
        if m == model and s == seed:
            return gap, fv
    raise KeyError((model, seed))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
    ax_a, ax_b, ax_c = axes

    # ---- A: own E3 gap vs R/N, one line per model -------------------------
    for model in (35, 50, 104):
        pts = sorted(
            (r / model, gap)
            for m, _arm, r, tls, _seed, gap, _fv in ARMS
            if m == model and tls == 1.0
        )
        xs, ys = zip(*pts)
        ax_a.plot(xs, ys, marker=MODEL_MARK[model], color=MODEL_COLOR[model],
                  lw=2.0, ms=7, label=f"{model}M (D/N=20, tls=1)")
    for model in (35, 50, 104):
        g, _ = nogram_gap(model, 42)
        ax_a.axhline(g, color=MODEL_COLOR[model], ls=":", lw=1.2, alpha=0.7)
    ax_a.axhline(0.0, color="#888", lw=0.8)
    ax_a.set_xlabel("R / N  (table rows per order ÷ backbone params ×1000)")
    ax_a.set_ylabel("own E3 window gap  (val − train)")
    ax_a.set_title("A · gap scales with R/N, not with model size\n(dotted = paired nogram baseline)",
                   fontsize=11)
    ax_a.legend(fontsize=9, frameon=False)
    ax_a.grid(alpha=0.25)

    # ---- B: table-induced gap vs N at fixed R/N ---------------------------
    for rn, color, mark in ((8, "#1a7f4b", "o"), (16, "#4a6fa5", "s")):
        xs, ys = [], []
        for model in (35, 50, 104):
            for m, _arm, r, tls, seed, gap, _fv in ARMS:
                if m == model and tls == 1.0 and abs(r / model - rn) < 0.75:
                    ng, _ = nogram_gap(model, seed)
                    xs.append(model)
                    ys.append(gap - ng)
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax_b.plot([xs[i] for i in order], [ys[i] for i in order], marker=mark,
                      color=color, lw=2.0, ms=7, label=f"R/N = {rn}")
    ax_b.set_xscale("log")
    ax_b.set_xticks([35, 50, 104])
    ax_b.set_xticklabels(["35M", "50M", "104M"])
    ax_b.set_xlabel("model size N  (D = 20N, R = (R/N)·N  — all proportional)")
    ax_b.set_ylabel("table-induced gap Δgap = arm − nogram")
    ax_b.set_title("B · proportional scaling is gap-neutral\n(Δgap flat over a ×3 scale-up)", fontsize=11)
    ax_b.legend(fontsize=9, frameon=False)
    ax_b.grid(alpha=0.25)

    # ---- C: E3 gain vs N at fixed R/N ------------------------------------
    for rn, color, mark in ((8, "#1a7f4b", "o"), (16, "#4a6fa5", "s")):
        xs, ys = [], []
        for model in (35, 50, 104):
            for m, _arm, r, tls, seed, _gap, fv in ARMS:
                if m == model and tls == 1.0 and abs(r / model - rn) < 0.75:
                    _ng, ng_fv = nogram_gap(model, seed)
                    xs.append(model)
                    ys.append((ng_fv - fv) * 1000.0)
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax_c.plot([xs[i] for i in order], [ys[i] for i in order], marker=mark,
                      color=color, lw=2.0, ms=7, label=f"R/N = {rn}")
    ax_c.axhline(0.0, color="#888", lw=0.8)
    ax_c.set_xscale("log")
    ax_c.set_xticks([35, 50, 104])
    ax_c.set_xticklabels(["35M", "50M", "104M"])
    ax_c.set_xlabel("model size N  (D = 20N, R = (R/N)·N)")
    ax_c.set_ylabel("E3 gain Δval  (mK, nogram − arm)")
    ax_c.set_title("C · the gain does NOT scale with N at fixed R/N\n(same table ratio buys less as N grows)", fontsize=11)
    ax_c.legend(fontsize=9, frameon=False)
    ax_c.grid(alpha=0.25)

    for ax in (ax_a, ax_b, ax_c):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "v26dn iso-ratio ladder (D/N=20, exact tables, E=3, tls=1): "
        "scaling N, D and R together moves the gap only through R/N",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    for ext in ("svg", "png"):
        path = os.path.join(OUT_DIR, f"fig_iso_ratio.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print("wrote", path)


if __name__ == "__main__":
    main()
