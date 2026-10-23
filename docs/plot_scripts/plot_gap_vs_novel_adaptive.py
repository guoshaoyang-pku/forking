#!/usr/bin/env python3
"""Overlay model gap and Good--Turing novel mass with adaptive bins.

``M(f)`` is the corpus-side Good--Turing estimate of novel continuation
probability.  Gap is the registered shared-context validation-minus-train
loss.  The two are aggregated over the same frequency bins, while retaining
their natural weights: M is context-layer weighted by n(f), and gap is token
mass weighted by f * shared_contexts.  Sparse high-frequency bins are merged
until each M bin contains at least MIN_CONTEXTS corpus contexts.

The output figure has, for each branch, an absolute dual-axis panel and a
log--log panel normalized at the first common bin.  Gap points stop where the
registered shared-context diagnostic stops; M continues over the full corpus
frequency range.
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
GAP_FILE = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_frequency_exact_points.csv"
FIG = ROOT / "docs" / "figs" / "theory" / "fig_gap_vs_novel_good_turing_adaptive.png"
CSV_OUT = M_DIR / "gap_vs_novel_adaptive_bins.csv"
SUMMARY_OUT = M_DIR / "gap_vs_novel_adaptive_summary.csv"
MIN_CONTEXTS = 128
INITIAL_BINS = 12


def weighted_powerlaw(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (weights > 0)
    x, y, weights = x[keep], y[keep], weights[keep]
    lx, ly = np.log(x), np.log(y)
    xb, yb = np.average(lx, weights=weights), np.average(ly, weights=weights)
    slope = float(np.sum(weights * (lx - xb) * (ly - yb)) / np.sum(weights * (lx - xb) ** 2))
    pred = yb + slope * (lx - xb)
    r2 = float(1.0 - np.sum(weights * (ly - pred) ** 2) / np.sum(weights * (ly - yb) ** 2))
    return {"beta": -slope, "slope": slope, "r2_log": r2, "n": int(keep.sum()),
            "intercept": float(yb - slope * xb)}


def adaptive_edges(f: np.ndarray, n_contexts: np.ndarray) -> np.ndarray:
    edges = list(np.unique(np.round(np.geomspace(1, f.max() + 1, INITIAL_BINS + 1)).astype(int)))
    while len(edges) > 2:
        counts = [int(n_contexts[(f >= lo) & (f < hi)].sum())
                  for lo, hi in zip(edges[:-1], edges[1:])]
        small = [i for i, count in enumerate(counts) if count < MIN_CONTEXTS]
        if not small:
            break
        i = small[-1]
        # Merge the sparse bin into its lower neighbour; the final high-f
        # interval therefore becomes wider as the data become sparse.
        if i == 0:
            edges.pop(1)
        else:
            edges.pop(i)
    return np.asarray(edges, dtype=int)


def load_branch(branch: str) -> tuple[dict, list[dict]]:
    mrows = list(csv.DictReader((M_DIR / f"good_turing_{branch}.csv").open()))
    mf = np.asarray([int(row["f"]) for row in mrows], dtype=int)
    mv = np.asarray([float(row["M"]) for row in mrows], dtype=float)
    mn = np.asarray([int(row["n_contexts"]) for row in mrows], dtype=int)
    gap_rows = []
    for row in csv.DictReader(GAP_FILE.open()):
        if row["branch"] != branch:
            continue
        f = int(row["f"]); contexts = int(row["shared_contexts"]); gap = float(row["gap"])
        # Match the registered gap fit: exact-f layers need at least 32
        # shared contexts, otherwise the apparent high-f tail is dominated by
        # one-context noise.
        if f > 0 and contexts >= 32 and np.isfinite(gap):
            gap_rows.append((f, gap, contexts, f * contexts))
    edges = adaptive_edges(mf, mn)
    bins = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        msel = (mf >= lo) & (mf < hi) & (mv > 0)
        if not msel.any():
            continue
        mweights = mn[msel].astype(float)
        mfbar = float(np.exp(np.average(np.log(mf[msel]), weights=mweights)))
        mbar = float(np.average(mv[msel], weights=mweights))
        gap_members = [row for row in gap_rows if lo <= row[0] < hi and row[1] > 0]
        if gap_members:
            gweights = np.asarray([row[3] for row in gap_members], dtype=float)
            gfbar = float(np.exp(np.average(np.log([row[0] for row in gap_members]), weights=gweights)))
            gbar = float(np.average([row[1] for row in gap_members], weights=gweights))
            gcontexts = int(sum(row[2] for row in gap_members))
            gtokens = int(sum(row[3] for row in gap_members))
            gpoints = len(gap_members)
        else:
            gfbar = gbar = float("nan")
            gcontexts = gtokens = gpoints = 0
        bins.append({
            "branch": branch, "bin": index, "f_lo": int(lo), "f_hi": int(hi - 1),
            "f_mid_M": mfbar, "M": mbar, "M_contexts": int(mn[msel].sum()),
            "M_layers": int(msel.sum()), "f_mid_gap": gfbar, "gap": gbar,
            "gap_contexts": gcontexts, "gap_tokens": gtokens,
            "gap_exact_f_points": gpoints,
        })
    return {"f": mf, "M": mv, "n": mn, "edges": edges}, bins


def main() -> None:
    FIG.parent.mkdir(parents=True, exist_ok=True)
    all_bins = []
    branch_data = {}
    summary = []
    for branch in ("bigram", "trigram"):
        data, bins = load_branch(branch)
        branch_data[branch] = (data, bins)
        all_bins.extend(bins)
        m = [row for row in bins if row["M"] > 0]
        g = [row for row in bins if np.isfinite(row["gap"]) and row["gap"] > 0]
        common = [row for row in bins if np.isfinite(row["gap"]) and row["gap"] > 0 and row["M"] > 0]
        mf = np.asarray([row["f_mid_M"] for row in m]); mv = np.asarray([row["M"] for row in m]); mn = np.asarray([row["M_contexts"] for row in m], float)
        gf = np.asarray([row["f_mid_gap"] for row in g]); gv = np.asarray([row["gap"] for row in g]); gw = np.asarray([row["gap_tokens"] for row in g], float)
        cf = np.asarray([row["f_mid_M"] for row in common]); cm = np.asarray([row["M"] for row in common]); cg = np.asarray([row["gap"] for row in common]);
        summary.append({"branch": branch, "bin_count": len(bins), "gap_bin_count": len(g),
                        "f_min": int(data["f"].min()), "f_max": int(data["f"].max()),
                        "gap_f_max": float(gf.max()) if len(gf) else "",
                        "M_beta_all_context_weighted": weighted_powerlaw(mf, mv, mn)["beta"],
                        "M_r2_all_context_weighted": weighted_powerlaw(mf, mv, mn)["r2_log"],
                        "gap_beta_observed_bins": weighted_powerlaw(gf, gv, gw)["beta"] if len(gf) >= 2 else "",
                        "gap_r2_observed_bins": weighted_powerlaw(gf, gv, gw)["r2_log"] if len(gf) >= 2 else "",
                        "M_beta_common_bins": weighted_powerlaw(cf, cm, np.ones_like(cf))["beta"] if len(cf) >= 2 else "",
                        "gap_beta_common_bins": weighted_powerlaw(cf, cg, np.ones_like(cf))["beta"] if len(cf) >= 2 else "",
                        "log_corr_gap_M_common_bins": float(np.corrcoef(np.log(cm), np.log(cg))[0, 1]) if len(cf) >= 2 else "",
                        "gap_over_M_cv_common_bins": float(np.std(cg / cm) / np.mean(cg / cm)) if len(cf) >= 2 else ""})
    with CSV_OUT.open("w", newline="") as handle:
        fields = list(all_bins[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_bins)
    with SUMMARY_OUT.open("w", newline="") as handle:
        fields = list(summary[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)

    colors = {"bigram": "#2166ac", "trigram": "#b2182b"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for col, branch in enumerate(("bigram", "trigram")):
        data, bins = branch_data[branch]; color = colors[branch]
        m = [row for row in bins if row["M"] > 0]
        g = [row for row in bins if np.isfinite(row["gap"]) and row["gap"] > 0]
        common = [row for row in bins if np.isfinite(row["gap"]) and row["gap"] > 0]
        mf = np.asarray([row["f_mid_M"] for row in m]); mv = np.asarray([row["M"] for row in m]); mn = np.asarray([row["M_contexts"] for row in m], float)
        gf = np.asarray([row["f_mid_gap"] for row in g]); gv = np.asarray([row["gap"] for row in g]); gw = np.asarray([row["gap_tokens"] for row in g], float)
        ax = axes[0, col]; right = ax.twinx()
        ax.loglog(gf, gv, "o-", color="#111111", lw=2.0, ms=5, label="gap (token-mass weighted)")
        right.loglog(mf, mv, "o-", color=color, lw=1.8, ms=4, label="novel mass M(f) (context weighted)")
        ax.axvspan(gf.min(), gf.max(), color="#777777", alpha=0.08, label="gap observed range")
        ax.set_xlabel("context frequency f"); ax.set_ylabel("gap (nats)"); right.set_ylabel("M(f): GT novel continuation mass")
        ax.set_title(f"{branch}: absolute values, adaptive bins")
        ax.grid(alpha=0.25, which="both")
        handles, labels = ax.get_legend_handles_labels(); h2, l2 = right.get_legend_handles_labels(); ax.legend(handles + h2, labels + l2, fontsize=8, loc="lower left")

        ax = axes[1, col]
        first_f = common[0]["f_mid_M"]; m0 = common[0]["M"]; g0 = common[0]["gap"]
        cmf = np.asarray([row["f_mid_M"] for row in common]); cmv = np.asarray([row["M"] for row in common]); cgv = np.asarray([row["gap"] for row in common])
        ax.loglog(cmf, cmv / m0, "o-", color=color, lw=1.8, ms=5, label="M(f) / M(first bin)")
        ax.loglog(cmf, cgv / g0, "s--", color="#111111", lw=1.8, ms=5, label="gap / gap(first bin)")
        all_mx = mf / first_f; all_my = mv / m0
        ax.loglog(mf, mv / m0, ".", color=color, alpha=0.35, ms=3)
        ax.axhline(1.0, color="#777777", lw=0.8)
        ax.set_xlabel("context frequency f"); ax.set_ylabel("normalized value")
        ax.set_title(f"{branch}: shared trend over observed gap range")
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Gap versus Good–Turing novel continuation mass: adaptive frequency bins", fontsize=14)
    fig.savefig(FIG, dpi=190)
    print(FIG.relative_to(ROOT)); print(CSV_OUT.relative_to(ROOT)); print(SUMMARY_OUT.relative_to(ROOT))
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
