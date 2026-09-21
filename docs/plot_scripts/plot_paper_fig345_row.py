#!/usr/bin/env python3
"""Paper Sec.3 scaling-law row: (a) frequency, (b) capacity, (c) corpus dose.

Three panels designed to sit side by side as subfigures (a)/(b)/(c) of one
LaTeX figure in overleaf-paper (2026-09-21 user decisions):
  (a) Fig3 merged into a SINGLE panel: bigram+trigram, log-log, 12 geometric
      mass-weighted bins +- within-bin std, dashed lines = the canonical
      Sec.4.1 unbinned direct fits (val-token weighted; must match text).
  (b) capacity drawn LOG-LOG (user: "fig4还是得画双对数"), no fit line
      (user: "不用强行拟合"); non-positive gaps omitted from log display.
  (c) corpus dose with LINEAR y (user: "fig5需要y轴非对数") and x = training
      corpus tokens in millions (user: "横轴展示token的数量，单位是M").

Data (seed 42; input injection; table LR 128x backbone where a table exists):
  (a) docs/appendices/s1_scaling_three_axis/s1_frequency_exact_points.csv
      both tables enabled (2^20 rows each), fixed-probe gap at step 1000
      (runs data/runs_scaling/s1v5_128_marm_both_fixed).
  (b) docs/appendices/s1_scaling_three_axis/s1_table_size_points.csv
      bigram-only / trigram-only online gap at step 1000, R = 1 .. 2.347e6
      (runs data/runs_scaling/s1v5_128_tbl_*_fixed).
  (c) docs/appendices/s1_scaling_three_axis/s1_dose_epoch_endpoints.csv
      trigram-only (2^20 rows), 20 corpus doses 0.125x..20x D1, gaps at the
      ends of epochs 2 and 3; x = epoch_batches * 72 * 2048 tokens / 1e6.

Fit-coefficient provenance for (a): CANON_FIT below mirrors the paper text
Section 4.1 (handoff canonical table, 2026-09-21); the underlying unbinned
val-token-weighted fits come from data/runs_scaling/s1v5_128_marm_both_fixed
(ophis-gpu) and are displayed in overleaf figures/fig_v5_good_turing_kernel.pdf.

Outputs (docs/figs/paper/):
  fig03a_frequency.pdf/.png
  fig03b_capacity.pdf/.png
  fig03c_corpus.pdf/.png
  fig345_row_preview.png        (side-by-side montage, review only)
Copy the three PDFs to overleaf-paper/figures/ with the same names.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

AX = ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
OUT = ROOT / "docs" / "figs" / "paper"

PANEL_W, PANEL_H = 2.2, 1.95
MIN_CONTEXTS = 32
N_BINS = 12
TOKENS_PER_BATCH = 72 * 2048  # device batch 72 x seq 2048 prediction tokens

# Canonical Sec.4.1 unbinned direct-fit coefficients (val-token weighted);
# dashed lines must match the main text exactly.
CANON_FIT = {"bigram": (7.206, -0.233), "trigram": (3.759, -0.318)}


def plain_fmt(v, _pos):
    if v >= 1:
        s = f"{v:.0f}" if v == int(v) else f"{v:g}"
    else:
        s = f"{v:g}"
    return s


def apply_panel_style():
    S.apply_style()
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
    })


def _save_panel(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # fixed canvas (no bbox_inches) so the three panels share exact dimensions
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print((OUT / f"{name}.pdf").relative_to(ROOT))


def freq_rows(branch):
    return [(float(r["f"]), float(r["gap"]), float(r["shared_token_mass"]))
            for r in csv.DictReader((AX / "s1_frequency_exact_points.csv").open())
            if r["branch"] == branch and int(r["step"]) == 1000
            and float(r["gap"]) > 0 and int(r["shared_contexts"]) >= MIN_CONTEXTS]


def geometric_bins(rows, n_bins=N_BINS):
    fs = np.array([r[0] for r in rows])
    edges = np.geomspace(fs.min(), fs.max() * 1.0001, n_bins + 1)
    out = []
    for i in range(n_bins):
        sel = [r for r in rows if edges[i] <= r[0] < edges[i + 1]]
        if not sel:
            continue
        f = np.array([s[0] for s in sel])
        g = np.array([s[1] for s in sel])
        w = np.array([s[2] for s in sel])
        fbar = np.exp(np.average(np.log(f), weights=w))
        gbar = np.average(g, weights=w)
        std = np.sqrt(np.average((g - gbar) ** 2, weights=w))
        out.append((fbar, gbar, std))
    return out


def draw_frequency(ax):
    fit_handles = []
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        bins = geometric_bins(freq_rows(branch))
        color = S.BRANCH_COLORS[branch]
        bx = np.array([b[0] for b in bins])
        bg = np.array([b[1] for b in bins])
        bs = np.array([b[2] for b in bins])
        ax.errorbar(bx, bg, yerr=bs, fmt=marker, color=color, ms=3.8, lw=0,
                    elinewidth=0.8, ecolor="#666666", capsize=1.8,
                    capthick=0.8, zorder=3, label=branch)
        amp, slope = CANON_FIT[branch]
        fx = np.geomspace(bx.min(), bx.max(), 80)
        ax.plot(fx, amp * fx ** slope, ls="--", lw=1.1, color=color, zorder=4)
    fit_handles.append(Line2D([], [], ls="--", lw=1.1, color="#444444",
                              label="direct fit (Sec. 4)"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + fit_handles, loc="upper right",
              framealpha=0.9, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, None)
    ax.set_ylim(10 ** -0.3, 10 ** 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xlabel("context frequency $F$")
    ax.set_ylabel("probe gap (nats)")


def draw_capacity(ax):
    rows = list(csv.DictReader((AX / "s1_table_size_points.csv").open()))
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        pts = sorted((int(r["R"]), float(r["gap_1000"]))
                     for r in rows if r["branch"] == branch)
        pos = [(R, g) for R, g in pts if g > 0]
        ax.plot([p[0] for p in pos], [p[1] for p in pos], marker=marker,
                ls="none", ms=3.8, color=S.BRANCH_COLORS[branch],
                label=f"{branch}-only")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 5e6)
    ax.set_ylim(0.011, 6)
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    ax.set_xlabel("table rows $R$")
    ax.set_ylabel("online gap (nats)")


def draw_corpus(ax):
    rows = list(csv.DictReader((AX / "s1_dose_epoch_endpoints.csv").open()))
    pts = sorted((int(r["epoch_batches"]) * TOKENS_PER_BATCH / 1e6,
                  float(r["gap_epoch_2"]), float(r["gap_epoch_3"]))
                 for r in rows)
    x = [p[0] for p in pts]
    for idx, (color, marker, label) in enumerate((
            ("#4f8f62", "s", "end of epoch 2"),
            ("#c0392b", "^", "end of epoch 3"))):
        ax.plot(x, [p[idx + 1] for p in pts], marker=marker, ms=3.6,
                lw=0.9, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlim(4.5, 1600)
    ax.set_ylim(-0.2, 4.1)
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1000"])
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_xlabel("corpus tokens (M)")
    ax.set_ylabel("online gap (nats)")


def make_panel(draw, name):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    draw(ax)
    fig.tight_layout(pad=0.4)
    _save_panel(fig, name)


def make_preview():
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.15))
    for ax, draw, tag in zip(axes, (draw_frequency, draw_capacity,
                                    draw_corpus), "abc"):
        draw(ax)
        ax.set_title(f"({tag})", fontsize=9, pad=2)
    fig.tight_layout(pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig345_row_preview.png", dpi=200)
    plt.close(fig)
    print((OUT / "fig345_row_preview.png").relative_to(ROOT))


def main():
    apply_panel_style()
    make_panel(draw_frequency, "fig03a_frequency")
    make_panel(draw_capacity, "fig03b_capacity")
    make_panel(draw_corpus, "fig03c_corpus")
    make_preview()


if __name__ == "__main__":
    main()
