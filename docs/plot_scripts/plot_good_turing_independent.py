#!/usr/bin/env python3
"""Plot the independent chunk-semantic Good--Turing audit.

The left column shows independently counted M(f) and the registered gap points
after the same context-support filter.  The right column shows exact-f M(f)
with the low-f [1,100] fit and the broader fit, making the finite-window
effect visible for each branch.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M_DIR = ROOT / "docs" / "appendices" / "gt_independent_20260920"
GAP = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_frequency_exact_points.csv"
OUT = ROOT / "docs" / "figs" / "theory" / "fig_good_turing_independent_audit.png"


def slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    lx, ly = np.log(x), np.log(y)
    xb, yb = np.average(lx, weights=w), np.average(ly, weights=w)
    return float(np.sum(w * (lx - xb) * (ly - yb)) / np.sum(w * (lx - xb) ** 2))


def main() -> None:
    gap_rows = list(csv.DictReader(GAP.open()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    colors = {"bigram": "#2166ac", "trigram": "#b2182b"}
    for i, branch in enumerate(("bigram", "trigram")):
        m_rows = list(csv.DictReader((M_DIR / f"good_turing_{branch}.csv").open()))
        mf = np.asarray([int(r["f"]) for r in m_rows], float)
        mv = np.asarray([float(r["M"]) for r in m_rows])
        mn = np.asarray([float(r["n_contexts"]) for r in m_rows])
        ax = axes[i, 1]
        ax.loglog(mf, mv, color=colors[branch], lw=1.2, alpha=0.85, label="independent M(f)")
        for hi, ls, alpha in ((100, "--", 1.0), (4096, ":", 0.85)):
            keep = (mf >= 1) & (mf <= hi) & (mv > 0)
            b = -slope(mf[keep], mv[keep], mn[keep])
            xx = np.geomspace(mf[keep].min(), mf[keep].max(), 80)
            yy = np.exp(np.average(np.log(mv[keep]) + b * np.log(mf[keep]), weights=mn[keep])) * xx ** (-b)
            ax.loglog(xx, yy, ls=ls, color="#333333", alpha=alpha, label=f"M fit f≤{hi}: β={b:.3f}")
        ax.set_xlabel("exact context count f")
        ax.set_ylabel("M(f)")
        ax.set_title(f"{branch}: independent chunk-semantic count")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

        points = []
        for r in gap_rows:
            if r["branch"] != branch or int(r["shared_contexts"]) < 32:
                continue
            f = int(r["f"]); g = float(r["gap"])
            if f not in {int(x) for x in mf} or g <= 0:
                continue
            points.append((f, g, f * int(r["shared_contexts"])))
        fs = np.asarray([x[0] for x in points], float)
        bounds = np.unique(np.geomspace(fs.min(), fs.max() + 1, 8).astype(int))
        pooled = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            sel = [x for x in points if lo <= x[0] < hi]
            if not sel:
                continue
            w = np.asarray([x[2] for x in sel], float)
            fbar = np.exp(np.average(np.log([x[0] for x in sel]), weights=w))
            gbar = np.average([x[1] for x in sel], weights=w)
            pooled.append((fbar, gbar, w.sum()))
        pf = np.asarray([x[0] for x in pooled]); pg = np.asarray([x[1] for x in pooled]); pw = np.asarray([x[2] for x in pooled])
        ax = axes[i, 0]
        ax.loglog(mf, mv, color="#888888", alpha=0.7, lw=1.0, label="M(f), independent")
        ax.scatter(pf, pg, s=24 + 56 * pw / pw.max(), color=colors[branch], label="gap bins")
        b_gap = -slope(pf, pg, np.ones_like(pf))
        # Report the exact-bin fit from the committed comparison summary.
        comp = list(csv.DictReader((M_DIR / "comparison_summary.csv").open()))
        row = next(x for x in comp if x["branch"] == branch and x["comparison"] == "same_7_bins")
        b_m = float(row["beta_M"])
        xx = np.geomspace(pf.min(), pf.max(), 80)
        amp = np.exp(np.mean(np.log(pg) + b_gap * np.log(pf)))
        ax.loglog(xx, amp * xx ** (-b_gap), color=colors[branch], ls="--", lw=1.3, label=f"gap fit β={b_gap:.3f}")
        positive_m = mv > 0
        m_at_bins = np.exp(np.interp(np.log(pf), np.log(mf[positive_m]), np.log(mv[positive_m])))
        amp_m = np.exp(np.mean(np.log(m_at_bins) + b_m * np.log(pf)))
        ax.loglog(xx, amp_m * xx ** (-b_m), color="#333333", ls=":", lw=1.3, label=f"M fit β={b_m:.3f}")
        ax.set_xlabel("exact context count f")
        ax.set_ylabel("gap or M(f), arbitrary amplitude")
        ax.set_title(f"{branch}: same 7 bins, shared contexts ≥ 32")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Independent Good–Turing audit: chunk semantics, seed 42 shard", fontsize=13)
    fig.savefig(OUT, dpi=180)
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
