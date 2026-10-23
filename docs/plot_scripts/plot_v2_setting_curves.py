#!/usr/bin/env python3
"""v2 setting curves: fixed-probe gap vs training context count, one figure per setting.

Input : docs/appendices/v2_order_sweep_20261017/exact_f_step2000.csv
        (per run x branch x integer context count f, step 2000; fixed-probe gap =
        val mean loss - train mean loss, both evaluated after the step-2000 update)
Method: f = 1..9 each plotted separately; f >= 10 pooled into log bins with
        token-weighted means.  A point is kept when both sides have >= MIN_TOK
        tokens.  Seeds are averaged where present (mean +- SD); the 2-8 joint
        settings are seed 42 only.

Outputs (docs/figs/sweeps_20261017/):
  fig_v2_setting_<k>.{svg,png}     k in {2, 2-3, ..., 2-3-4-5-6-7-8}   (7 figures)
  fig_v2_singles_exact_f.{svg,png} singles n=2..8: curves + alpha vs collision load
"""
import csv
import collections
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/Users/guoshaoyang/.verdent/skills/report-to-human/scripts")
import matplotlib.pyplot as plt  # noqa: E402
from panel_style import OKABE_ITO, C, save, apply  # noqa: E402

apply()

CSV = os.path.join(REPO, "docs", "appendices", "v2_order_sweep_20261017", "exact_f_step2000.csv")
OUT = os.path.join(REPO, "docs", "figs", "sweeps_20261017")
os.makedirs(OUT, exist_ok=True)

MIN_TOK = 100
EXACT = list(range(1, 10))
BINS = [(10, 19), (20, 49), (50, 99), (100, 199), (200, 499), (500, 999),
        (1000, 1999), (2000, 4999), (5000, 9999), (10000, 10 ** 9)]
R = 1048576
DISTINCT = {2: 3538293, 3: 18989467, 4: 34822438, 5: 42952218,
            6: 46350667, 7: 47745239, 8: 48392919}
SETTINGS = [(list(range(2, k + 1)), k) for k in range(2, 9)]


def branch_of(n):
    return {2: "bigram", 3: "trigram"}.get(n, f"{n}gram")


def arm_of(orders):
    return "n" + "-".join(str(n) for n in orders)


def load():
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(CSV)):
        key = (r["arm"], r["branch"])
        data[key][int(r["seed"])].append(dict(
            f=int(r["f"]), nt=int(r["n_train"]), nv=int(r["n_val"]),
            tm=float(r["train_mean"]), vm=float(r["val_mean"])))
    return data


def seed_points(rows):
    """Gap points for one seed: exact f in EXACT plus pooled bins (both sides >= MIN_TOK)."""
    pts = {}
    by_f = {r["f"]: r for r in rows}
    for f in EXACT:
        r = by_f.get(f)
        if r and min(r["nt"], r["nv"]) >= MIN_TOK:
            pts[("f", f)] = (float(f), r["vm"] - r["tm"])
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["f"] <= hi]
        nt = sum(r["nt"] for r in sel)
        nv = sum(r["nv"] for r in sel)
        if sel and min(nt, nv) >= MIN_TOK:
            tm = sum(r["tm"] * r["nt"] for r in sel) / nt
            vm = sum(r["vm"] * r["nv"] for r in sel) / nv
            x = math.sqrt(lo * min(hi, 10 ** 5))
            pts[("bin", lo)] = (x, vm - tm)
    return pts


def curve(data, arm, branch):
    """Mean +- SD over seeds for each x key. Returns sorted list of (x, mean, sd, nseeds)."""
    per_seed = data.get((arm, branch), {})
    acc = collections.defaultdict(list)
    for seed, rows in per_seed.items():
        for key, (x, g) in seed_points(rows).items():
            acc[key].append((x, g))
    out = []
    for key, vals in acc.items():
        xs = [v[0] for v in vals]
        gs = np.array([v[1] for v in vals])
        out.append((xs[0], float(gs.mean()),
                    float(gs.std(ddof=1)) if len(gs) > 1 else 0.0, len(gs)))
    return sorted(out)


def draw_setting(ax, data, orders):
    cols = OKABE_ITO + ["#999999"]
    for i, n in enumerate(orders):
        col = cols[i % len(cols)]
        c = curve(data, arm_of(orders), branch_of(n))
        if c:
            x, y, sd, ns = zip(*c)
            ax.errorbar(x, y, yerr=sd if max(ns) > 1 else None, fmt="o-", ms=3.8,
                        lw=1.4, capsize=2, color=col, label=f"order n={n}")
        b = curve(data, "nogram", branch_of(n))
        if b:
            x, y, _, _ = zip(*b)
            ax.plot(x, y, ls=":", lw=1.0, color=col, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = EXACT + [30, 100, 300, 1000, 3000, 10000, 30000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in EXACT] + ["30", "100", "300", "1k", "3k", "10k", "30k"])
    ax.minorticks_off()
    ax.axvline(9.5, color=C["soft"] if "soft" in C else "#cccccc", lw=0.9, zorder=0)
    ax.set_xlabel("training context count f  (1–9 separate; ≥10 pooled into log bins)")
    ax.set_ylabel("fixed-probe gap, val − train (nats)")
    ax.legend(fontsize=8, ncol=2, frameon=False)


def fig_setting(data, orders):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    draw_setting(ax, data, orders)
    lbl = "-".join(str(n) for n in orders[:1]) + ("" if len(orders) == 1 else f"-{orders[-1]}")
    ax.set_title(f"setting n = {lbl}: gap vs context frequency, step 2000", fontsize=10)
    ax.text(0.01, 0.02, "dotted: no-table baseline (nogram) per order", transform=ax.transAxes,
            fontsize=7, color="#666666")
    stem = os.path.join(OUT, f"fig_v2_setting_{lbl}")
    save(fig, stem)
    return stem


def fit_alpha(points):
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    m = ys > 0
    if m.sum() < 3:
        return float("nan"), float("nan")
    lx, ly = np.log(xs[m]), np.log(ys[m])
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return -b, r2


def fig_singles(data):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.4, 4.3), constrained_layout=True)
    cols = OKABE_ITO + ["#999999"]
    alphas, loads, sds = [], [], []
    for i, n in enumerate(range(2, 9)):
        col = cols[i % len(cols)]
        c = curve(data, f"n{n}", branch_of(n))
        if not c:
            continue
        x, y, sd, ns = zip(*c)
        ax_a.errorbar(x, y, yerr=sd if max(ns) > 1 else None, fmt="o-", ms=3.2, lw=1.2,
                      capsize=2, color=col, label=f"n={n}")
        per_seed = []
        for seed, rows in data[(f"n{n}", branch_of(n))].items():
            pts = list(seed_points(rows).values())
            a, r2 = fit_alpha(pts)
            per_seed.append((a, r2))
        a_arr = np.array([p[0] for p in per_seed])
        alphas.append(float(np.nanmean(a_arr)))
        sds.append(float(np.nanstd(a_arr, ddof=1)) if len(a_arr) > 1 else 0.0)
        loads.append(DISTINCT[n] / R)
        ax_b.annotate(f"n={n}", (loads[-1], alphas[-1]), textcoords="offset points",
                      xytext=(4, 4), fontsize=8)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ticks = EXACT + [30, 100, 300, 1000, 3000, 10000, 30000]
    ax_a.set_xticks(ticks)
    ax_a.set_xticklabels([str(t) for t in EXACT] + ["30", "100", "300", "1k", "3k", "10k", "30k"])
    ax_a.minorticks_off()
    ax_a.set_xlabel("training context count f  (1–9 separate; ≥10 pooled)")
    ax_a.set_ylabel("fixed-probe gap at order n (nats)")
    ax_a.set_title("(a) singles n=2..8, diagonal, 3 seeds", fontsize=10)
    ax_a.legend(fontsize=8, ncol=2, frameon=False)
    ax_b.errorbar(loads, alphas, yerr=sds, fmt="s-", ms=5, capsize=3, color="C3")
    ax_b.axhline(0, color="0.5", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("table load = distinct train contexts / R (R = 2^20)")
    ax_b.set_ylabel("exponent α (gap ~ f^−α)")
    ax_b.set_title("(b) exponent falls with collision load", fontsize=10)
    stem = os.path.join(OUT, "fig_v2_singles_exact_f")
    save(fig, stem)
    return stem, dict(zip(range(2, 9), zip(alphas, sds)))


if __name__ == "__main__":
    data = load()
    for orders, _ in SETTINGS:
        stem = fig_setting(data, orders)
        print("wrote", stem)
    stem, alpha = fig_singles(data)
    print("wrote", stem)
    for n, (a, s) in alpha.items():
        print(f"alpha n={n}: {a:.3f} +- {s:.3f}")
