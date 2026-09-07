#!/usr/bin/env python3
"""y/v scaling wave figures (§52): three-axis comparison input vs y vs v.

Inputs:
  - input arm (historical, both tables on): docs/report/versions/.../s1_table_size_points.csv
    and s1_epoch_length_points.csv
  - y/v arms (this wave, bigram single table): summary.json pulled under /tmp/yv_sum
    (ophis/ = tbl_y, g3601/ = tbl_v, g3602/ = ep + mlv5)

Outputs: docs/figs/main/fig_v5_yv_scaling_three_axis.png
"""
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "docs/report/versions/experiment-registry-20260827-before-v5-prune-assets/docs/appendices/s1_scaling_three_axis"
SUM = Path("/tmp/yv_sum")
OUT = ROOT / "docs/figs/main/fig_v5_yv_scaling_three_axis.png"

ARM_STYLE = {
    "input": dict(color="#1f77b4", marker="o", label="input (bi2: both tables on, vary bigram R)"),
    "y": dict(color="#2ca02c", marker="s", label="y (bigram single table)"),
    "v": dict(color="#d62728", marker="^", label="v (bigram single table)"),
}


def load_summaries(pattern):
    out = {}
    for p in glob.glob(str(pattern)):
        rid = Path(p).parent.name.replace("_fixed", "")
        out[rid] = json.load(open(p))
    return out


def tbl_points(summaries, pat=r"_R(\d+)$"):
    pts = []
    for rid, s in summaries.items():
        m = re.search(pat, rid)
        if m:
            pts.append((int(m.group(1)), s["final_gap"]))
    return np.array(sorted(pts))


def ep_points(summaries):
    pts = []
    for rid, s in summaries.items():
        m = re.search(r"_ep_bi_(\d+p\d*)xL4_3ep$", rid)
        if m:
            mult = float(m.group(1).replace("p", "."))
            if mult > 1.0:
                continue  # old >1x points carried the data-reuse bug; ep2 replaces them
            pts.append((mult, s["final_gap"]))
    return np.array(sorted(pts))


def ep2_points(summaries):
    """Corrected >1x epoch arms (ep2 wave, real multi-shard train sets)."""
    pts = {}
    for rid, s in summaries.items():
        m = re.search(r"_(input|y|v)_ep2_(\d+p\d+)x?L4_3ep$", rid)
        if m:
            arm = m.group(1)
            mult = float(m.group(2).replace("p", "."))
            pts.setdefault(arm, []).append((mult, s["final_gap"]))
    return {a: np.array(sorted(v)) for a, v in pts.items()}


def fit_window(x, y, lo, hi):
    m = (x >= lo) & (x <= hi) & (y > 0)
    if m.sum() < 3:
        return None
    sl, ic = np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)
    return sl, ic, m


def main():
    # ---------- data ----------
    input_tbl = {}
    with open(CSV_DIR / "s1_table_size_points.csv") as fh:
        for r in csv.DictReader(fh):
            if r["branch"] == "bigram":
                input_tbl[int(r["R"])] = float(r["final_gap"])
    input_tbl = np.array(sorted(input_tbl.items()))

    input_ep = []
    with open(CSV_DIR / "s1_epoch_length_points.csv") as fh:
        for r in csv.DictReader(fh):
            input_ep.append((float(r["epoch_multiplier_L4"]), float(r["final_gap"])))
    input_ep = np.array(sorted(input_ep))

    tbl = {
        "input": input_tbl,
        "y": tbl_points(load_summaries(SUM / "ophis/s1v5_128_y_tbl_*/summary.json")),
        "v": tbl_points(load_summaries(SUM / "g3601/s1v5_128_v_tbl_*/summary.json")
                        | load_summaries(SUM / "g3602/s1v5_128_v_tbl_*/summary.json")),
    }
    ep = {"input": input_ep[input_ep[:, 0] <= 1.0]}  # >1x CSV points carried the reuse bug
    for pos in ("y", "v"):
        ep[pos] = ep_points(load_summaries(SUM / f"g3602/s1v5_128_{pos}_ep_*/summary.json"))
    for arm, pts2 in ep2_points(load_summaries(SUM / "ep2/*/summary.json")).items():
        ep[arm] = np.vstack([ep[arm], pts2])

    ml = load_summaries(SUM / "g3602/mlv5_*/summary.json")
    ml_gap = {rid.replace("mlv5_", "").replace("_fd", ""): s["final_gap"]
              for rid, s in ml.items()}
    main_gap = {"y_{1}": 5.328084945678711, "v_{1}": 7.805806517601013}  # inj_fd main runs @2000

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    ax = axes[0]
    for arm, pts in tbl.items():
        st = ARM_STYLE[arm]
        ax.plot(pts[:, 0], pts[:, 1], marker=st["marker"], ms=4, lw=0,
                color=st["color"], label=st["label"], alpha=0.85)
        fit = fit_window(pts[:, 0], pts[:, 1], 1e4, 1.3e6)
        if fit:
            sl, ic, m = fit
            xx = np.array([1e4, 1.3e6])
            ax.plot(xx, 10 ** (sl * np.log10(xx) + ic), color=st["color"], lw=1.2,
                    alpha=0.8)
            ax.text(0.03, 0.97 - 0.07 * list(tbl).index(arm),
                    f"{arm}: slope={sl:.3f}", transform=ax.transAxes,
                    color=st["color"], fontsize=10, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical table rows R (bigram)")
    ax.set_ylabel("gap @1000 (val - train)")
    ax.set_title("(a) table-size scaling, log-log\nfit window R in [1e4, 1.3e6]")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for arm, pts in ep.items():
        st = ARM_STYLE[arm]
        ax.plot(pts[:, 0], pts[:, 1], marker=st["marker"], ms=5, lw=1,
                color=st["color"], label=arm, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("epoch length (multiple of L4 = 337 steps)")
    ax.set_ylabel("gap @ 3 passes (fixed pass count)")
    ax.set_title("(b) epoch-length scaling @ fixed 3 passes\n"
                 "bigram single table; >1x = ep2 wave (true multi-shard prefix\n"
                 "train sets, per-dose val); val-set switches at 1x->1.25x")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    groups = ["{1}", "{1,7}", "{1,3,5,7}"]
    xv = np.arange(3)
    w = 0.35
    yv = [main_gap["y_{1}"], ml_gap.get("y_L17"), ml_gap.get("y_L1357")]
    vv = [main_gap["v_{1}"], ml_gap.get("v_L17"), ml_gap.get("v_L1357")]
    ax.bar(xv - w / 2, yv, w, color=ARM_STYLE["y"]["color"], alpha=0.85, label="y")
    ax.bar(xv + w / 2, vv, w, color=ARM_STYLE["v"]["color"], alpha=0.85, label="v")
    for i, (a, b) in enumerate(zip(yv, vv)):
        ax.text(i - w / 2, a + 0.06, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.06, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(xv); ax.set_xticklabels(groups)
    ax.set_xlabel("inject layers (independent clean table per layer)")
    ax.set_ylabel("gap @2000 (both tables R=2^20)")
    ax.set_title("(c) multi-layer injection @2000\n{1} = main inj_fd runs")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("y/v scaling wave (§52): v5 minimal, RMSProp(0,.99) x128, backbone 6e-4, seed 42, 1x shard",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT, dpi=160)
    print(f"[done] {OUT}")

    for arm, pts in tbl.items():
        fit = fit_window(pts[:, 0], pts[:, 1], 1e4, 1.3e6)
        if fit:
            print(f"tbl {arm}: n={len(pts)} slope={fit[0]:.3f}")
    for arm, pts in ep.items():
        print(f"ep  {arm}: n={len(pts)} gap range [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print("ml:", {k: round(v, 3) for k, v in sorted(ml_gap.items())})


if __name__ == "__main__":
    main()
