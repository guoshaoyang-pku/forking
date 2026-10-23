#!/usr/bin/env python3
"""Full-space Good--Turing power-law diagnostic.

Plots every positive exact-f M(f) layer from the independent chunk-semantic
recount, with context-weighted geometric-bin means and an all-space weighted
power-law fit.  The lower panels show multiplicative residuals, while the
registered model-side gap bins are shown only over their observed f range.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "appendices" / "gt_independent_20260920"
GAP = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_frequency_exact_points.csv"
OUT = ROOT / "docs" / "figs" / "theory" / "fig_good_turing_fullspace_powerlaw.png"
SUMMARY = DATA / "fullspace_summary.csv"


def logfit(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    lx, ly = np.log(x), np.log(y)
    xb, yb = np.average(lx, weights=w), np.average(ly, weights=w)
    slope = np.sum(w * (lx - xb) * (ly - yb)) / np.sum(w * (lx - xb) ** 2)
    intercept = yb - slope * xb
    pred = intercept + slope * lx
    r2 = 1.0 - np.sum(w * (ly - pred) ** 2) / np.sum(w * (ly - yb) ** 2)
    return float(slope), float(intercept), float(r2)


def bins(f, y, w, nbin=32):
    edges = np.unique(np.round(np.geomspace(1, f.max() + 1, nbin + 1)).astype(int))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        z = (f >= lo) & (f < hi) & (y > 0)
        if not z.any():
            continue
        ww = w[z]
        out.append((np.exp(np.average(np.log(f[z]), weights=ww)),
                    np.average(y[z], weights=ww), ww.sum()))
    return np.asarray(out, dtype=float)


def gap_bins(branch, m_lookup):
    rows = []
    for row in csv.DictReader(GAP.open()):
        if row["branch"] != branch or int(row["shared_contexts"]) < 32:
            continue
        f = int(row["f"]); g = float(row["gap"])
        if g > 0 and f in m_lookup:
            rows.append((f, g, f * int(row["shared_contexts"])))
    if not rows:
        return np.empty((0, 3))
    f = np.asarray([x[0] for x in rows], float)
    edges = np.unique(np.round(np.geomspace(f.min(), f.max() + 1, 8)).astype(int))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        z = (f >= lo) & (f < hi)
        if not z.any():
            continue
        w = np.asarray([rows[j][2] for j in np.where(z)[0]], float)
        vals = [rows[j] for j in np.where(z)[0]]
        out.append((np.exp(np.average(np.log([x[0] for x in vals]), weights=w)),
                    np.average([x[1] for x in vals], weights=w), w.sum()))
    return np.asarray(out, dtype=float)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.3, 1.0]})
    colors = {"bigram": "#2166ac", "trigram": "#b2182b"}
    for col, branch in enumerate(("bigram", "trigram")):
        data = list(csv.DictReader((DATA / f"good_turing_{branch}.csv").open()))
        f = np.asarray([int(r["f"]) for r in data], float)
        m = np.asarray([float(r["M"]) for r in data])
        n = np.asarray([int(r["n_contexts"]) for r in data], float)
        positive = m > 0
        for weighting, weights in (("equal_exact_f", np.ones_like(n)),
                                   ("context_weighted", n),
                                   ("token_weighted", f * n)):
            s_all, _, r_all = logfit(f[positive], m[positive], weights[positive])
            summary_rows.append({
                "branch": branch, "domain": "all_positive_exact_f",
                "weighting": weighting, "beta": -s_all,
                "r2_log": r_all, "n_exact_f": int(positive.sum()),
                "f_min": int(f[positive].min()), "f_max": int(f[positive].max()),
                "zero_M_layers": int((m <= 0).sum()),
            })
        slope, intercept, r2 = logfit(f[positive], m[positive], n[positive])
        beta = -slope
        bb = bins(f[positive], m[positive], n[positive])
        ax = axes[0, col]
        ax.scatter(f[positive], m[positive], s=5, alpha=0.16, color=colors[branch],
                   rasterized=True, label=f"exact-f layers (n={positive.sum():,})")
        ax.plot(bb[:, 0], bb[:, 1], "o-", ms=4, lw=1.6, color=colors[branch],
                label="32 geometric bins, n(f)-weighted")
        xx = np.geomspace(f[positive].min(), f[positive].max(), 300)
        ax.plot(xx, np.exp(intercept) * xx ** slope, "--", color="#222222",
                lw=1.5, label=f"all-space fit: β={beta:.3f}, R²={r2:.3f}")
        gaps = gap_bins(branch, {int(x): float(y) for x, y in zip(f, m)})
        if len(gaps):
            m_interp = np.exp(np.interp(np.log(gaps[:, 0]), np.log(bb[:, 0]), np.log(bb[:, 1])))
            scale = np.median(gaps[:, 1] / m_interp)
            ax.scatter(gaps[:, 0], gaps[:, 1] / scale, s=38 + 40 * gaps[:, 2] / gaps[:, 2].max(),
                       facecolors="none", edgecolors="#111111", linewidths=1.1,
                       label="gap bins (M-scaled)")
            ax.axvspan(gaps[:, 0].min(), gaps[:, 0].max(), color="#777777", alpha=0.07,
                       label="observed gap range")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{branch}: full corpus frequency space")
        ax.set_xlabel("context frequency f"); ax.set_ylabel("M(f)")
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8, loc="lower left")

        # Residuals of the exact-f values and geometric means against the
        # weighted all-space power law.
        axr = axes[1, col]
        pred = np.exp(intercept) * f[positive] ** slope
        axr.axhline(1.0, color="#222222", lw=1.0)
        axr.scatter(f[positive], m[positive] / pred, s=4, alpha=0.15,
                    color=colors[branch], rasterized=True)
        bpred = np.exp(intercept) * bb[:, 0] ** slope
        axr.plot(bb[:, 0], bb[:, 1] / bpred, "o-", ms=3, lw=1.2, color=colors[branch])
        axr.set_xscale("log"); axr.set_yscale("log")
        axr.set_xlabel("context frequency f"); axr.set_ylabel("M / power-law")
        axr.grid(alpha=0.25, which="both")
        axr.set_title(f"residuals: median={np.median(m[positive] / pred):.2f}")
    fig.suptitle("Independent Good–Turing missing-mass power law across the full exact-f space", fontsize=14)
    fig.savefig(OUT, dpi=190)
    with SUMMARY.open("w", newline="") as handle:
        fields = ["branch", "domain", "weighting", "beta", "r2_log",
                  "n_exact_f", "f_min", "f_max", "zero_M_layers"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(OUT.relative_to(ROOT))
    print(SUMMARY.relative_to(ROOT))


if __name__ == "__main__":
    main()
