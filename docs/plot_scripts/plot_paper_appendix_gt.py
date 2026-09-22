#!/usr/bin/env python3
"""Paper appendix Good--Turing figures (replaces fig_b_t_MF and
fig_v5_good_turing_kernel), recomputed from committed CSV/JSON only.

(A) fig_app_gt_missing_mass: exact-F Good--Turing missing mass
    M(F) = N1(F) / (F n(F)) for bigram and trigram contexts, all positive
    layers; the fit is restricted to F <= 1000 (the window in which the gap is
    measured) and weights each context equally (n(F)); the equal-weight
    all-F fit is drawn for contrast to show the bias from the sparse tail.
(B) fig_app_gt_kernel: measured step-1000 gap in the seven matched geometric
    bins against g * M(F) with a single amplitude g, plus the per-bin ratio.

Data:
  docs/appendices/gt_independent_20260920/good_turing_{bigram,trigram}.csv
  docs/appendices/gt_independent_20260920/comparison_summary.json
      (pooled_bins: same 7 geometric bins for gap and M, run
      s1v5_128_frequency_main, seed 42, step 1000, shared_contexts >= 32)
Outputs (docs/figs/paper/): fig_app_gt_missing_mass.{pdf,png},
fig_app_gt_kernel.{pdf,png}; copy PDFs to overleaf-paper/figures/.
"""
from __future__ import annotations

import csv
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

GT = ROOT / "docs" / "appendices" / "gt_independent_20260920"
OUT = ROOT / "docs" / "figs" / "paper"
CANON_FIT = {"bigram": (7.206, -0.233), "trigram": (3.759, -0.318)}
FMAX_FIT = 1000.0


def wfit(x, y, w):
    lx, ly = np.log(x), np.log(y)
    xb, yb = np.average(lx, weights=w), np.average(ly, weights=w)
    b = np.sum(w * (lx - xb) * (ly - yb)) / np.sum(w * (lx - xb) ** 2)
    return float(np.exp(yb - b * xb)), float(b)


def load_gt(branch):
    rows = list(csv.DictReader((GT / f"good_turing_{branch}.csv").open()))
    f = np.array([float(r["f"]) for r in rows])
    n = np.array([float(r["n_contexts"]) for r in rows])
    M = np.array([float(r["M"]) for r in rows])
    keep = M > 0
    return f[keep], n[keep], M[keep]


def style():
    S.apply_style()
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})


def fig_missing_mass(summary):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, branch in zip(axes, ("bigram", "trigram")):
        f, n, M = load_gt(branch)
        color = S.BRANCH_COLORS[branch]
        win = f <= FMAX_FIT
        C_ctx, b_ctx = wfit(f[win], M[win], n[win])
        C_tok, b_tok = wfit(f[win], M[win], f[win] * n[win])
        C_eq, b_eq = wfit(f, M, np.ones_like(f))
        summary[branch] = dict(alpha_ctx=-b_ctx, C_ctx=C_ctx, alpha_tok=-b_tok, C_tok=C_tok,
                               alpha_eq_all=-b_eq, C_eq_all=C_eq,
                               tok_frac_win=float((f * n)[win].sum() / (f * n).sum()),
                               ctx_frac_win=float(n[win].sum() / n.sum()),
                               layers=int(len(f)), layers_win=int(win.sum()))
        big = n >= 8
        ax.scatter(f[~big], M[~big], s=5, color="#bbbbbb", lw=0, zorder=2,
                   label="layers with $n(F)<8$")
        ax.scatter(f[big], M[big], s=7, color=color, lw=0, alpha=0.85, zorder=3,
                   label="layers with $n(F)\\geq 8$")
        fx = np.geomspace(1, f.max(), 200)
        ax.plot(fx, C_eq * fx ** b_eq, ls=":", lw=1.0, color="#777777", zorder=4,
                label=f"equal weight, all $F$: $\\alpha={-b_eq:.2f}$")
        fw = fx[fx <= FMAX_FIT]
        ax.plot(fw, C_ctx * fw ** b_ctx, lw=1.6, color="#222222", zorder=5,
                label=f"per-context weight, $F\\leq10^3$: $\\alpha={-b_ctx:.3f}$")
        ax.plot(fx[fx > FMAX_FIT], C_ctx * fx[fx > FMAX_FIT] ** b_ctx, lw=1.2, ls="--",
                color="#222222", zorder=5)
        ax.axvspan(FMAX_FIT, f.max() * 1.5, color="#000000", alpha=0.05, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, f.max() * 1.5)
        ax.set_ylim(3e-4, 1.5)
        ax.set_xlabel(f"{branch} context count $F$")
        ax.set_title(f"{branch}: $\\hat f(F)\\approx{C_ctx:.2f}\\,F^{{-{-b_ctx:.3f}}}$",
                     fontsize=9, pad=3)
        ctx_tail = 100 * (1 - summary[branch]["ctx_frac_win"])
        ctx_txt = f"{ctx_tail:.1f}%" if ctx_tail >= 0.05 else "<0.1%"
        ax.text(FMAX_FIT * 1.3, 0.6, f"{ctx_txt} of contexts\n"
                f"{100 * (1 - summary[branch]['tok_frac_win']):.0f}% of tokens",
                fontsize=7, color="#555555", va="top")
        ax.legend(loc="lower left", framealpha=0.9, edgecolor="none", handlelength=1.8)
    axes[0].set_ylabel("Good\u2013Turing novel-continuation rate $\\hat f(F)$")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_app_gt_missing_mass.{ext}", dpi=200)
    plt.close(fig)


def fig_kernel(summary):
    comp = json.loads((GT / "comparison_summary.json").read_text())["branches"]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.2), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08, "wspace": 0.25})
    for j, branch in enumerate(("bigram", "trigram")):
        bins = comp[branch]["pooled_bins"]
        f = np.array([b["f_mid"] for b in bins])
        gap = np.array([b["gap"] for b in bins])
        M = np.array([b["M"] for b in bins])
        w = np.array([b["weight"] for b in bins])
        ratio = gap / M
        g = float(np.exp(np.mean(np.log(ratio))))  # geometric mean over the 7 bins
        cv = float(ratio.std() / ratio.mean())
        corr = float(np.corrcoef(np.log(gap), np.log(M))[0, 1])
        summary[branch].update(g=g, ratio_cv=cv, log_corr=corr, f_max_bins=float(f.max()))
        color = S.BRANCH_COLORS[branch]
        ax, axr = axes[0, j], axes[1, j]
        fx = np.geomspace(f.min(), f.max(), 80)
        amp, slope = CANON_FIT[branch]
        ax.plot(fx, amp * fx ** slope, ls="--", lw=1.0, color=color, alpha=0.7,
                label=f"direct fit ${amp:.2f}\\,F^{{{slope:.3f}}}$")
        ax.plot(f, g * M, marker="s", ms=4, mfc="white", color="#222222", lw=1.2,
                label=f"$D\\,\\hat f(F)$, $D={g:.2f}$")
        ax.plot(f, gap, marker="o", ms=4.5, color=color, lw=0, label="measured gap")
        ax.set_xscale("log")
        ax.set_ylim(0, 8 if branch == "bigram" else 4.5)
        ax.set_title(branch, fontsize=9, pad=3)
        ax.legend(loc="upper right", framealpha=0.9, edgecolor="none", handlelength=1.8)
        axr.axhline(g, color="#222222", lw=0.8)
        axr.axhspan(g * (1 - cv), g * (1 + cv), color="#222222", alpha=0.08, lw=0)
        axr.plot(f, ratio, marker="o", ms=3.5, color=color, lw=1.0)
        axr.set_xscale("log")
        axr.set_ylim(0, g * 1.8)
        axr.set_xlabel(f"{branch} context count $F$")
        axr.text(0.02, 0.85, f"CV {100 * cv:.0f}%", transform=axr.transAxes, fontsize=7,
                 va="top", color="#333333")
    axes[0, 0].set_ylabel("step-1000 gap (nats)")
    axes[1, 0].set_ylabel("gap / $\\hat f$")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.14)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_app_gt_kernel.{ext}", dpi=200)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    fig_missing_mass(summary)
    fig_kernel(summary)
    print(json.dumps(summary, indent=1))
    print((OUT / "fig_app_gt_missing_mass.pdf").relative_to(ROOT))
    print((OUT / "fig_app_gt_kernel.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
