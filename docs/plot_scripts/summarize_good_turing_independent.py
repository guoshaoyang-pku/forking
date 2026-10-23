#!/usr/bin/env python3
"""Compare independently counted M(f) with the registered gap bins.

The input M(f) CSVs are produced by compute_good_turing_independent.py using
the training chunk semantics.  The gap points remain the registered seed-42
step-1000 fixed-train-probe artifacts.  This script uses the same seven
geometric bins, context threshold, positive-gap filter, and token-mass
weighting as analyze_v5_scaling.py, then reports both equal-bin and token-mass
weighted slopes.  It does not overwrite the historical theory summary.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT = ROOT / "docs" / "appendices" / "gt_independent_20260920"
GAP_POINTS = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_frequency_exact_points.csv"
OUT = INDEPENDENT / "comparison_summary.json"
OUT_CSV = INDEPENDENT / "comparison_summary.csv"


def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    lx, ly = np.log(x), np.log(y)
    xb, yb = np.average(lx, weights=w), np.average(ly, weights=w)
    slope = float(np.sum(w * (lx - xb) * (ly - yb)) / np.sum(w * (lx - xb) ** 2))
    pred = yb + slope * (lx - xb)
    r2 = float(1.0 - np.sum(w * (ly - pred) ** 2) / np.sum(w * (ly - yb) ** 2))
    return {"slope": slope, "beta": -slope, "r2_log": r2, "n": int(len(x)), "weight_sum": float(w.sum())}


def exact_m_window(m_rows: dict[int, dict], lo: int, hi: int, weight_name: str) -> dict:
    """Fit independently counted M(f) over an exact-f window."""
    entries = [row for f, row in m_rows.items() if lo <= f <= hi and float(row["M"]) > 0]
    x = np.asarray([int(row["f"]) for row in entries], dtype=float)
    y = np.asarray([float(row["M"]) for row in entries], dtype=float)
    if weight_name == "contexts":
        w = np.asarray([float(row["n_contexts"]) for row in entries], dtype=float)
    elif weight_name == "context_tokens":
        w = np.asarray([float(row["context_tokens"]) for row in entries], dtype=float)
    elif weight_name == "equal":
        w = np.ones_like(x)
    else:
        raise ValueError(weight_name)
    out = fit(x, y, w)
    out.update({"f_lo": lo, "f_hi": hi, "weighting": weight_name})
    return out


def main() -> None:
    gap_rows = list(csv.DictReader(GAP_POINTS.open()))
    result = {"inputs": {"gap_points": str(GAP_POINTS.relative_to(ROOT)),
                             "M_dir": str(INDEPENDENT.relative_to(ROOT)),
                             "run_id": "s1v5_128_frequency_main", "step": 1000,
                             "seed": 42,
                             "gap_definition": "fixed validation loss - fixed train probe loss"},
              "branches": {}}
    result["inputs"]["frequency_semantics_note"] = (
        "The registered gap CSV comes from the historical run/index; the new M(f)"
        " uses chunk-semantic frequencies. The old raw-concat index differs on"
        " 20027 bigram and 19542 trigram contexts (metadata.json)."
    )
    for branch in ("bigram", "trigram"):
        m_rows = {int(r["f"]): r for r in csv.DictReader(
            (INDEPENDENT / f"good_turing_{branch}.csv").open()
        )}
        exact = []
        for row in gap_rows:
            if row["branch"] != branch:
                continue
            f = int(row["f"]); c = int(row["shared_contexts"]); gap = float(row["gap"])
            # Keep every exact-f row whose context support passes the
            # registered threshold.  The M value will be pooled over the same
            # members as the gap; missing exact f values cannot occur because
            # the independent stream contains the same context set.
            if f > 0 and c >= 32 and np.isfinite(gap) and f in m_rows:
                exact.append((f, gap, f * c, c, float(m_rows[f]["M"])))
        fs = np.asarray([x[0] for x in exact], dtype=float)
        # Match pooled_exact_rows(): bounds use max f + 1 and integerized
        # geometric spacing, with seven populated bins in this run.
        bounds = np.unique(np.geomspace(fs.min(), fs.max() + 1, 8).astype(int))
        pooled = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            members = [x for x in exact if lo <= x[0] < hi]
            if not members:
                continue
            mass = np.asarray([x[2] for x in members], dtype=float)
            pooled.append({
                "f_mid": float(np.exp(np.average(np.log([x[0] for x in members]), weights=mass))),
                "gap": float(np.average([x[1] for x in members], weights=mass)),
                "M": float(np.average([x[4] for x in members], weights=mass)),
                "weight": float(mass.sum()),
                "contexts": int(sum(x[3] for x in members)),
                "f_lo": int(lo), "f_hi": int(hi - 1),
            })
        # Registered gap slope is equal-weighted over positive pooled bins.
        positive = [x for x in pooled if x["gap"] > 0 and x["M"] > 0]
        x = np.asarray([r["f_mid"] for r in positive]); g = np.asarray([r["gap"] for r in positive]); m = np.asarray([r["M"] for r in positive]); mass = np.asarray([r["weight"] for r in positive])
        result["branches"][branch] = {
            "window": {"context_min": 32, "geometric_bins": 7, "positive_gap_only": True,
                        "f_min": float(x.min()), "f_max": float(x.max())},
            "pooled_bins": positive,
            "gap_equal_bin": fit(x, g, np.ones_like(x)),
            "M_equal_bin": fit(x, m, np.ones_like(x)),
            "gap_token_mass_weighted": fit(x, g, mass),
            "M_token_mass_weighted": fit(x, m, mass),
            "beta_difference_equal_bin": float(fit(x, g, np.ones_like(x))["beta"] - fit(x, m, np.ones_like(x))["beta"]),
            "beta_difference_token_mass": float(fit(x, g, mass)["beta"] - fit(x, m, mass)["beta"]),
            "M_exact_windows": {
                str(hi): {name: exact_m_window(m_rows, 1, hi, name)
                          for name in ("contexts", "context_tokens", "equal")}
                for hi in (100, 4096)
            },
            "low_window_comparison": {
                "window": "exact f in [1,100]; same six populated gap bins after shared_contexts >= 32",
                "gap_beta_equal_bins": None,
                "M_beta_equal_bins": None,
                "gap_beta_token_mass_bins": None,
                "M_beta_token_mass_bins": None,
            },
        }
        # Repeat the same binning after restricting exact f to [1,100],
        # which is the historical M(f) window for the trigram check.
        low_exact = [x for x in exact if x[0] <= 100 and x[1] > 0]
        lfs = np.asarray([x[0] for x in low_exact], dtype=float)
        lbounds = np.unique(np.geomspace(lfs.min(), lfs.max() + 1, 7).astype(int))
        low_bins = []
        for lo, hi in zip(lbounds[:-1], lbounds[1:]):
            members = [x for x in low_exact if lo <= x[0] < hi]
            if not members:
                continue
            mass = np.asarray([x[2] for x in members], dtype=float)
            low_bins.append((float(np.exp(np.average(np.log([x[0] for x in members]), weights=mass))),
                             float(np.average([x[1] for x in members], weights=mass)),
                             float(np.average([x[4] for x in members], weights=mass)),
                             float(mass.sum())))
        lx = np.asarray([z[0] for z in low_bins]); lg = np.asarray([z[1] for z in low_bins]); lm = np.asarray([z[2] for z in low_bins]); lw = np.asarray([z[3] for z in low_bins])
        rec = result["branches"][branch]["low_window_comparison"]
        rec["n_bins"] = int(len(low_bins))
        rec["gap_beta_equal_bins"] = fit(lx, lg, np.ones_like(lx))["beta"]
        rec["M_beta_equal_bins"] = fit(lx, lm, np.ones_like(lx))["beta"]
        rec["gap_beta_token_mass_bins"] = fit(lx, lg, lw)["beta"]
        rec["M_beta_token_mass_bins"] = fit(lx, lm, lw)["beta"]
        # A direct independent M(f) fit over the same low-frequency window
        # used by the historical trigram cross-check.  The model gap fit and
        # this count fit still use different observations, so this remains a
        # cross-check rather than a causal proof.
    with OUT.open("w") as fh:
        json.dump(result, fh, indent=2)
    with OUT_CSV.open("w", newline="") as fh:
        fields = ["branch", "comparison", "beta_gap", "beta_M", "delta_beta", "r2_gap", "r2_M", "n_bins", "f_lo", "f_hi"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for branch, rec in result["branches"].items():
            eq_gap, eq_m = rec["gap_equal_bin"], rec["M_equal_bin"]
            low = rec["low_window_comparison"]
            writer.writerow({"branch": branch, "comparison": "same_7_bins", "beta_gap": eq_gap["beta"],
                             "beta_M": eq_m["beta"], "delta_beta": eq_gap["beta"] - eq_m["beta"],
                             "r2_gap": eq_gap["r2_log"], "r2_M": eq_m["r2_log"],
                             "n_bins": eq_gap["n"], "f_lo": rec["window"]["f_min"], "f_hi": rec["window"]["f_max"]})
            writer.writerow({"branch": branch, "comparison": "low_f_1_100_six_bins",
                             "beta_gap": low["gap_beta_equal_bins"], "beta_M": low["M_beta_equal_bins"],
                             "delta_beta": low["gap_beta_equal_bins"] - low["M_beta_equal_bins"],
                             "n_bins": low["n_bins"], "f_lo": 1, "f_hi": 100})
    for branch, rec in result["branches"].items():
        print(branch,
              f"gap_beta_equal={rec['gap_equal_bin']['beta']:.6f}",
              f"M_beta_equal={rec['M_equal_bin']['beta']:.6f}",
              f"gap_beta_mass={rec['gap_token_mass_weighted']['beta']:.6f}",
              f"M_beta_mass={rec['M_token_mass_weighted']['beta']:.6f}")
    print(OUT.relative_to(ROOT))
    print(OUT_CSV.relative_to(ROOT))


if __name__ == "__main__":
    main()
