#!/usr/bin/env python3
"""Paper Sec.3 scaling-law row: (a) frequency, (b) capacity, (c) corpus dose.

Three panels designed to sit side by side as subfigures (a)/(b)/(c) of one
LaTeX figure in overleaf-paper (2026-09-21 user decisions; 2026-09-22 panel (a)
switched to the ONLINE caliber per user request "都用online重画"):
  (a) bigram+trigram, log-log, native log-spaced freq bins of
      freq_bin_loss.jsonl (step-1000 record).  ONLINE gap per bucket:
      gap = fixed-val mean loss - current-batch online train mean loss
      (train side = the training batch in flight at step 1000, update not yet
      applied; val side = fixed val batches at the step-1000 model).  The
      "novel" bucket and empty buckets are excluded (no train loss / no data).
      Dashed lines = val-token-weighted log-log fits to these online buckets
      (coefficients printed to stdout; they replace the old fixed-probe
      CANON_FIT numbers, which no longer match this caliber).
  (b) capacity drawn LOG-LOG (user: "fig4还是得画双对数"), no fit line
      (user: "不用强行拟合"); non-positive gaps omitted from log display.
  (c) corpus dose with LINEAR y (user: "fig5需要y轴非对数") and x = training
      corpus tokens in millions (user: "横轴展示token的数量，单位是M").

Data (seed 42; input injection; table LR 128x backbone where a table exists):
  (a) docs/appendices/s1_scaling_three_axis/s1_frequency_online_freqbin.jsonl
      both tables enabled (2^20 rows each), step-1000 online freq-bin record
      (rerun of s1v5_128_frequency_main on ophis-gpu 2026-09-22, run dir
      data/runs_scaling/s1v5_128_frequency_main_fixed, code commit a4ed6ae;
      the original run dir was lost to cluster cleanup, user authorized rerun).
  (b) docs/appendices/s1_scaling_three_axis/s1_table_size_points.csv
      bigram-only / trigram-only online gap at step 1000, R = 1 .. 2.347e6
      (runs data/runs_scaling/s1v5_128_tbl_*_fixed).
  (c) docs/appendices/s1_scaling_three_axis/s1_dose_epoch_endpoints.csv
      trigram-only (2^20 rows), 20 corpus doses 0.125x..20x D1, gaps at the
      ends of epochs 2 and 3; x = epoch_batches * 72 * 2048 tokens / 1e6.

Reference (fixed-probe caliber, for stdout comparison only): the old panel (a)
fits from data/runs_scaling/s1v5_128_marm_both_fixed exact-F probe points
(see CANON_FIT below; also displayed in overleaf figures/fig_v5_good_turing_kernel.pdf).

Outputs (docs/figs/paper/):
  fig03a_frequency.pdf/.png
  fig03b_capacity.pdf/.png
  fig03c_corpus.pdf/.png
  fig345_row_preview.png        (side-by-side montage, review only)
Copy the three PDFs to overleaf-paper/figures/ with the same names.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

AX = ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
OUT = ROOT / "docs" / "figs" / "paper"
ONLINE_JSONL = AX / "s1_frequency_online_freqbin.jsonl"
ONLINE_STEP = 1000

PANEL_W, PANEL_H = 2.2, 1.95
TOKENS_PER_BATCH = 72 * 2048  # device batch 72 x seq 2048 prediction tokens

# Old fixed-probe canonical fits (paper text Sec. 4.1) -- kept ONLY for the
# stdout comparison printout; the drawn dashed lines are refit to the online
# buckets of the rerun (different caliber, so coefficients differ).
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


def parse_bucket_f(name):
    """Geometric-centre F for a log-spaced bucket label like '6-8', '2k-5k', '10k+'."""
    def val(tok):
        tok = tok.strip().lower().rstrip("+")
        if tok.endswith("k"):
            return float(tok[:-1]) * 1000
        return float(tok)
    parts = name.split("-")
    if len(parts) == 1:
        return val(parts[0])
    lo, hi = val(parts[0]), val(parts[1])
    return float(np.sqrt(lo * hi))


def online_rows(branch):
    """(F, online gap, val tokens) per native freq bucket at ONLINE_STEP."""
    rec = None
    for line in ONLINE_JSONL.open():
        r = json.loads(line)
        if r["step"] == ONLINE_STEP:
            rec = r
    if rec is None:
        raise SystemExit(f"step {ONLINE_STEP} not found in {ONLINE_JSONL}")
    rows = []
    names = [n for n in rec["train"][branch] if n != "novel"]
    for name in sorted(names, key=parse_bucket_f):
        if name == "novel":
            continue
        tr = rec["train"][branch][name]
        va = rec["val"][branch][name]
        if tr["token_count"] <= 0 or va["token_count"] <= 0:
            continue
        gap = va["mean_loss"] - tr["mean_loss"]
        if gap <= 0:  # non-positive gaps omitted from log display (as before)
            continue
        rows.append((parse_bucket_f(name), gap, float(va["token_count"])))
    return rows


def weighted_fit(rows):
    """Val-token-weighted log-log fit  gap = amp * F**slope  -> (amp, slope)."""
    lf = np.log([r[0] for r in rows])
    lg = np.log([r[1] for r in rows])
    w = np.sqrt([r[2] for r in rows])
    slope, intercept = np.polyfit(lf, lg, 1, w=w)
    return float(np.exp(intercept)), float(slope)


def draw_frequency(ax):
    fit_handles = []
    coefs = {}
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        rows = online_rows(branch)
        color = S.BRANCH_COLORS[branch]
        fx = np.array([r[0] for r in rows])
        gy = np.array([r[1] for r in rows])
        ax.plot(fx, gy, marker=marker, ls="none", ms=3.8, lw=0,
                color=color, zorder=3, label=branch)
        amp, slope = weighted_fit(rows)
        coefs[branch] = (amp, slope, len(rows))
        xs = np.geomspace(fx.min(), fx.max(), 80)
        ax.plot(xs, amp * xs ** slope, ls="--", lw=1.1, color=color, zorder=4)
    fit_handles.append(Line2D([], [], ls="--", lw=1.1, color="#444444",
                              label="online fit"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + fit_handles, loc="upper right",
              framealpha=0.9, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, None)
    ax.set_ylim(10 ** -1.8, 10 ** 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xlabel("context frequency $F$")
    ax.set_ylabel("online gap (nats)")
    for branch in ("bigram", "trigram"):
        amp, slope, n = coefs[branch]
        camp, cslope = CANON_FIT[branch]
        print(f"[fig03a] {branch}: online fit amp={amp:.3f} slope={slope:.3f} "
              f"(n={n} buckets) | old fixed-probe canon amp={camp:.3f} "
              f"slope={cslope:.3f}")
    return coefs


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
