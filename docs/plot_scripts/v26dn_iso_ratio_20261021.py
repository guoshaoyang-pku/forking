#!/usr/bin/env python3
"""v26dn iso-ratio (proportional scaling) figure -- §132.

Question (user 2026-10-21 16:34): scale model size, data volume and table size
by the same factor (D/N = 20, R/N fixed, tls fixed) -- how does the forking gap
change?

All arms are the v26dn ladder: D/N = 20, per-order exact tables
(`assign_exact_r{R}k.npz`, zero collision), table lr = tls * backbone lr
(0.0006), E = 3 epochs, tls = 1 unless noted.

Readings (window convention, §128): own gap = mean over the last 600 steps of
the epoch of (fixed val - online train); dGap = arm - same-model same-seed
nogram; dVal = nogram final val - arm final val (mK, positive = OE wins).

Every arm carries a divergence screen verdict (§132 k-m): a late runaway that
re-inflates the train loss invalidates the end-of-run gap and gain. Diverged
arms are drawn as red crosses and excluded from fits and means; seed replicates
are aggregated to a mean with a min-max range instead of being connected.
The 17M rung is drawn hollow: 3 of its 4 exact arms diverged and the survivor
is contaminated, so the rung anchors no fit.

Panels
  A  own E3 window gap vs R/N, one line per model (means over seeds)
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
RED = "#b3261e"

# ---------------------------------------------------------------- data -------
# Clean arms only (`win_gap` = own E3 window gap, `fv` = final val).
# model_M, R_per_order_k, seed, win_gap, final_val
ARMS = [
    # 35m, D = 0.7B
    (35, 100, 42, -0.0081, 2.8270),
    (35, 280, 42, +0.0270, 2.8051),
    (35, 280, 44, +0.0305, 2.8105),
    (35, 560, 42, +0.0747, 2.8173),
    # 50m, D = 1.0B
    (50, 100, 42, -0.0112, 2.6774),
    (50, 128, 42, -0.0057, 2.6661),
    (50, 256, 42, +0.0152, 2.6629),
    (50, 400, 42, +0.0365, 2.6618),
    (50, 400, 43, +0.0361, 2.6544),
    (50, 800, 42, +0.0880, 2.6614),
    (50, 800, 43, +0.0811, 2.6621),
    (50, 1600, 42, +0.1515, 2.6854),
    (50, 1600, 43, +0.1493, 2.6870),
    # 104m, D = 2.1B
    (104, 400, 42, -0.0061, 2.5598),
    (104, 832, 42, +0.0356, 2.5126),
    (104, 832, 43, +0.0345, 2.5145),
    (104, 1600, 42, +0.0889, 2.5167),
    # 17m, D = 0.34B -- only survivor of four; kept out of every fit (see §132 k)
    (17, 272, 42, +0.0304, 3.2880),
]
# The one 17M arm that did not run away: its screen is borderline (train level
# 1.04x nogram, E2 excursion, gap still growing), so it is drawn hollow.
CONTAMINATED = {(17, 272, 42)}
# Diverged arms: train loss re-inflates late, end-of-run gap/gain is an artifact.
# model_M, R_per_order_k, seed, reported win_gap, reported dVal_mK, note
DIVERGED = [
    (17, 136, 42, +0.0018, None, "17M R/N=8"),
    (17, 136, 43, +0.0026, None, "17M R/N=8"),
    (17, 544, 42, +0.0033, None, "17M R/N=32"),
    (35, 1120, 42, +0.0446, None, "35M R/N=32"),
    (50, 800, 42, None, None, "50M R/N=16, E=5"),
]
# paired nogram (same model / seed / stream). 35m nogram runs carry tls0.5 in
# the run id but nogram has no table, so tls is a no-op there.
# model_M, seed, win_gap, final_val
NOGRAM = [
    (17, 42, -0.0279, 3.1554),
    (17, 43, -0.0269, 3.1450),
    (35, 42, -0.0321, 2.8541),
    (35, 44, -0.0324, 2.8586),
    (50, 42, -0.0305, 2.7009),
    (50, 43, -0.0313, 2.6907),
    (104, 42, -0.0340, 2.5302),
    (104, 43, -0.0338, 2.5482),
]
# fitted law dGap = A * N^a * (R/N)^p over the 12 clean 35/50/104M R/N=8/16/32
# points, log-space rms 3.7% -- §132 (j)
LAW_A, LAW_A_N, LAW_A_RN = 0.00843, 0.120, 0.760
MODEL_COLOR = {17: "#8a6f9e", 35: "#d97757", 50: "#1a7f4b", 104: "#4a6fa5"}
MODEL_MARK = {17: "D", 35: "o", 50: "s", 104: "^"}
RN_COLOR = {8: "#1a7f4b", 16: "#4a6fa5"}
RN_MARK = {8: "o", 16: "s"}


def law_dgap(model: int, rn: float) -> float:
    """fitted local dGap = A * N^a * (R/N)^p."""
    return LAW_A * model**LAW_A_N * rn**LAW_A_RN


def nogram(model: int, seed: int) -> tuple[float, float]:
    for m, s, gap, fv in NOGRAM:
        if m == model and s == seed:
            return gap, fv
    raise KeyError((model, seed))


def rows(model: int, rn: float) -> list[tuple[int, float, float]]:
    """clean (seed, own_gap, final_val) rows at one (model, R/N)."""
    return sorted(
        (seed, gap, fv)
        for m, r_k, seed, gap, fv in ARMS
        if m == model and abs(r_k / model - rn) < 0.75
    )


def paired(model: int, rn: float) -> list[tuple[int, float, float]]:
    """(seed, dGap, dVal_mK) for clean arms, mean-ready."""
    out = []
    for seed, gap, fv in rows(model, rn):
        ng_gap, ng_fv = nogram(model, seed)
        out.append((seed, gap - ng_gap, (ng_fv - fv) * 1000.0))
    return out


def mean_range(values: list[float]) -> tuple[float, float, float]:
    mid = sum(values) / len(values)
    return mid, min(values), max(values)


def anchor(models: tuple[int, ...], rn: float, which: int) -> tuple[list[int], list[float], list[float], list[float]]:
    """mean of the `which`-th paired quantity (1 = dGap, 2 = dVal_mK) with range."""
    xs, means, lows, highs = [], [], [], []
    for model in models:
        pts = paired(model, rn)
        if not pts:
            continue
        vals = [p[which] for p in pts]
        xs.append(model)
        means.append(sum(vals) / len(vals))
        lows.append(min(vals))
        highs.append(max(vals))
    return xs, means, lows, highs


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))
    ax_a, ax_b, ax_c = axes

    # ---- A: own E3 gap vs R/N, mean over seeds per model -------------------
    for model in (17, 35, 50, 104):
        rns = sorted({r_k / m for m, r_k, *_ in ARMS if m == model})
        line_x, line_y = [], []
        for rn in rns:
            pts = rows(model, rn)
            if not pts:
                continue
            gaps = [p[1] for p in pts]
            for seed, gap, _fv in pts:
                open_pt = (model, int(round(rn * model)), seed) in CONTAMINATED
                ax_a.plot([rn], [gap], marker=MODEL_MARK[model], ms=6 if open_pt else 4,
                          mfc="none" if open_pt else MODEL_COLOR[model],
                          mec=MODEL_COLOR[model], mew=1.4, ls="none",
                          alpha=0.85 if open_pt else 0.5)
            mid = sum(gaps) / len(gaps)
            line_x.append(rn)
            line_y.append(mid)
        if line_x:
            ax_a.plot(line_x, line_y, marker=MODEL_MARK[model], color=MODEL_COLOR[model],
                      lw=2.0, ms=7, label=f"{model}M (D/N=20)")
    for model in (17, 35, 50, 104):
        ax_a.axhline(nogram(model, 42)[0], color=MODEL_COLOR[model], ls=":", lw=1.2, alpha=0.75)
    for model, r_k, _seed, gap, _dv, _note in DIVERGED:
        if gap is None:
            continue
        ax_a.plot([r_k / model], [gap], marker="x", ms=11, mew=2.2, color=RED, ls="none")
    ax_a.axhline(0.0, color="#888", lw=0.8)
    ax_a.set_xlabel("R / N  (table rows per order ÷ backbone params in M)")
    ax_a.set_ylabel("own E3 window gap  (val − train)")
    ax_a.set_title("A · gap follows R/N, not model size\n"
                   "(dotted = paired nogram; ✗ = diverged run, gap collapses to 0)",
                   fontsize=10.5)
    ax_a.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax_a.grid(alpha=0.25)

    # ---- B: table-induced gap vs N at fixed R/N ---------------------------
    for rn in (8, 16):
        xs, means, lows, highs = anchor((35, 50, 104), rn, 1)
        ax_b.errorbar(xs, means, yerr=[[m - lo for m, lo in zip(means, lows)],
                                      [hi - m for m, hi in zip(means, highs)]],
                      marker=RN_MARK[rn], color=RN_COLOR[rn], lw=2.0, ms=7, capsize=3,
                      label=f"R/N = {rn}  (mean of seeds, range)")
        law_x = [35, 104]
        ax_b.plot(law_x, [law_dgap(n, rn) for n in law_x], color=RN_COLOR[rn], ls="--",
                  lw=1.3, alpha=0.85)
        if rn == 16:
            ax_b.plot([17], [law_dgap(17, 16)], marker="D", ms=6, color=RN_COLOR[rn],
                      ls="none", alpha=0.5,
                      label="fit 0.0084·N$^{0.12}$·(R/N)$^{0.76}$")
    surv = paired(17, 16)
    if surv:
        ax_b.plot([17], [surv[0][1]], marker="D", ms=8, mfc="none", mec=MODEL_COLOR[17],
                  mew=1.8, ls="none", label="17M survivor (excluded)")
    for model, r_k, seed, gap, _dval, _note in DIVERGED:
        if gap is None:
            continue  # E=5 arm: no 3-epoch gap is defined
        ng_gap, _ng_fv = nogram(model, seed)
        ax_b.plot([model], [gap - ng_gap], marker="x", ms=11, mew=2.2, color=RED,
                  ls="none", zorder=5)
    ax_b.annotate("diverged", xy=(17, 0.030), xytext=(18.5, -0.062), fontsize=8.5, color=RED,
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax_b.annotate("diverged\n(E3 runaway)", xy=(35, 0.0767), xytext=(38, 0.145), fontsize=8.5,
                  color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax_b.axvspan(15.0, 27.0, color=RED, alpha=0.06)
    ax_b.annotate("table unstable below ~35M\n(3 of 4 exact arms at 17M diverge)",
                  xy=(17.0, 0.005), xytext=(16.4, 0.215), fontsize=8.5, color=RED)
    ax_b.axhline(0.0, color="#888", lw=0.8)
    ax_b.set_ylim(-0.09, 0.24)
    ax_b.set_xlim(14.5, 130)
    ax_b.set_xscale("log")
    ax_b.set_xticks([17, 35, 50, 104])
    ax_b.set_xticklabels(["17M", "35M", "50M", "104M"])
    ax_b.minorticks_off()
    ax_b.set_xlabel("model size N  (D = 20N, R = (R/N)·N  — all proportional)")
    ax_b.set_ylabel("table-induced gap Δgap = arm − nogram")
    ax_b.set_title("B · proportional scaling is gap-neutral from 35M up\n"
                   "(Δgap flat over ×3; the low-N end is a different failure)", fontsize=10.5)
    ax_b.legend(fontsize=8, frameon=False, loc="lower right")
    ax_b.grid(alpha=0.25)

    # ---- C: E3 gain vs N at fixed R/N -------------------------------------
    for rn in (8, 16):
        xs, means, lows, highs = anchor((35, 50, 104), rn, 2)
        ax_c.errorbar(xs, means, yerr=[[m - lo for m, lo in zip(means, lows)],
                                       [hi - m for m, hi in zip(means, highs)]],
                      marker=RN_MARK[rn], color=RN_COLOR[rn], lw=2.0, ms=7, capsize=3,
                      label=f"R/N = {rn}  (mean of seeds, range)")
    if surv:
        ax_c.plot([17], [surv[0][2]], marker="D", ms=8, mfc="none", mec=MODEL_COLOR[17],
                  mew=1.8, ls="none", label="17M survivor (excluded)")
    ax_c.axhline(0.0, color="#888", lw=0.8)
    ax_c.axvspan(15.0, 27.0, color=RED, alpha=0.06)
    ax_c.set_xlim(14.5, 130)
    ax_c.set_ylim(-170, 62)
    ax_c.set_xscale("log")
    ax_c.set_xticks([17, 35, 50, 104])
    ax_c.set_xticklabels(["17M", "35M", "50M", "104M"])
    ax_c.minorticks_off()
    ax_c.set_xlabel("model size N  (D = 20N, R = (R/N)·N)")
    ax_c.set_ylabel("E3 gain Δval  (mK, nogram − arm)")
    ax_c.set_title("C · the gain does NOT scale with N at fixed R/N\n"
                   "(same table ratio buys less as N grows)", fontsize=10.5)
    ax_c.legend(fontsize=8.5, frameon=False, loc="lower left")
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
