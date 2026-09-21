#!/usr/bin/env python3
"""Candidate variants for paper Fig3 (combined grams + linear log-log fit),
Fig4 (capacity, linear y), Fig5 (corpus dose, linear y). Review-only outputs."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

AX = ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
OUT = ROOT / "docs" / "figs" / "main" / "variants"
MIN_CONTEXTS = 32
N_BINS = 12
# Canonical Sec4.1 unbinned direct-fit coefficients (val-token weighted);
# dashed lines must match the main text exactly.
CANON_FIT = {"bigram": (7.206, -0.233), "trigram": (3.759, -0.318)}


def freq_rows(branch):
    return [(float(r["f"]), float(r["gap"]), float(r["shared_token_mass"]))
            for r in csv.DictReader((AX / "s1_frequency_exact_points.csv").open())
            if r["branch"] == branch and int(r["step"]) == 1000
            and float(r["gap"]) > 0 and int(r["shared_contexts"]) >= MIN_CONTEXTS]


def geometric_bins(rows, n_bins=N_BINS):
    fs = np.array([r[0] for r in rows])
    lo, hi = fs.min(), fs.max()
    edges = np.geomspace(lo, hi * 1.0001, n_bins + 1)
    out = []
    for i in range(n_bins):
        sel = [r for r in rows if edges[i] <= r[0] < edges[i + 1]]
        if not sel:
            continue
        f = np.array([s[0] for s in sel]); g = np.array([s[1] for s in sel]); w = np.array([s[2] for s in sel])
        fbar = np.exp(np.average(np.log(f), weights=w))
        gbar = np.average(g, weights=w)
        out.append((fbar, gbar, np.sqrt(np.average((g - gbar) ** 2, weights=w))))
    return out


def loglog_fit(rows):
    x = np.log([r[0] for r in rows]); y = np.log([r[1] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def fig3_combined():
    S.apply_style()
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    for branch in ("bigram", "trigram"):
        rows = freq_rows(branch)
        color = S.BRANCH_COLORS[branch]
        bx = np.array([b[0] for b in geometric_bins(rows)])
        bg = np.array([b[1] for b in geometric_bins(rows)])
        bs = np.array([b[2] for b in geometric_bins(rows)])
        ax.errorbar(bx, bg, yerr=bs, fmt="o", color=color, ms=5, lw=0,
                    elinewidth=0.8, ecolor="#666666", capsize=2, capthick=0.8,
                    zorder=3, label=f"{branch} binned means")
        amp, slope = CANON_FIT[branch]
        fx = np.geomspace(bx.min(), bx.max(), 80)
        ax.plot(fx, amp * fx ** slope, ls="--", lw=1.2, color=color,
                zorder=4, label=f"{branch} fit: $G\\propto f^{{{slope:.3f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(10 ** -0.3, 10 ** 1.0)
    ax.set_xlabel("exact train hit-count per context f")
    ax.set_ylabel("gap (val - train probe) @ step 1000")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "cand_fig3_combined_fit.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cand_fig3_combined_fit.pdf", bbox_inches="tight")
    plt.close(fig)


def table_rows(branch):
    return [(float(r["R"]), float(r["final_gap"]))
            for r in csv.DictReader((AX / "s1_table_size_points.csv").open())
            if r["axis"] == "table_size" and r["branch"] == branch]


def fig4_linear_y():
    S.apply_style()
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    for branch in ("bigram", "trigram"):
        rows = table_rows(branch)
        color = S.BRANCH_COLORS[branch]
        R = np.array([r[0] for r in rows]); G = np.array([r[1] for r in rows])
        ax.plot(R, G, "o", color=color, ms=4.5, label=branch)
        sel = G > 0.02
        slope, intercept = np.polyfit(np.log(R[sel]), np.log(G[sel] - 0.02), 1)
        fx = np.geomspace(R[sel].min(), R.max(), 80)
        ax.plot(fx, 0.02 + np.exp(intercept + slope * np.log(fx)), ls="--", lw=1.2, color=color,
                label=f"{branch} fit: $\\beta={slope:.3f}$")
    ax.set_xscale("log")
    ax.set_xlabel("table rows R")
    ax.set_ylabel("gap @ step 1000")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cand_fig4_capacity_linearY.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cand_fig4_capacity_linearY.pdf", bbox_inches="tight")
    plt.close(fig)


def dose_rows():
    return [(float(r["dose"]), float(r["gap_epoch_2"]), float(r["gap_epoch_3"]))
            for r in csv.DictReader((AX / "s1_dose_epoch_endpoints.csv").open())]


def fig5_linear_y():
    S.apply_style()
    rows = dose_rows()
    d = np.array([r[0] for r in rows])
    g2 = np.array([r[1] for r in rows]); g3 = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    ax.plot(d, g2, "s", color="#4f8f62", ms=4.5, label="epoch-2 endpoint")
    ax.plot(d, g3, "^", color="#7d5ba6", ms=4.5, label="epoch-3 endpoint")
    ax.set_xscale("log")
    ax.set_xlabel("normalized training corpus size d")
    ax.set_ylabel("gap at epoch end")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cand_fig5_corpus_linearY.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "cand_fig5_corpus_linearY.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig3_combined()
    fig4_linear_y()
    fig5_linear_y()
    for p in sorted(OUT.glob("cand_*.png")):
        print(p.relative_to(ROOT))
