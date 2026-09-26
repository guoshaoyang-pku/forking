#!/usr/bin/env python3
"""Paper Sec.3 scaling-law row: (a) frequency, (b) capacity, (c) corpus dose.

Three panels designed to sit side by side as subfigures (a)/(b)/(c) of one
LaTeX figure in overleaf-paper (2026-09-21 user decisions; panel (a) caliber
history: fixed-probe until 2026-09-25, then the ONLINE three-seed coverage
caliber, final; 2026-09-26: panel (a) restyled to the shared row canvas and
fonts, linear-regression fit kept, error bars dropped):
  (a) bigram+trigram ONLINE gap at step 1000, three-seed means (42/43/44)
      per native frequency bin, NO error bars (seed SD <= 0.07 nats);
      dashed lines = equal-bin OLS LINEAR regression of the gap on the
      measured novel fraction f (the caption's "scaled and shifted novel
      fractions"), drawn against F.  Data from the coverage audit packet
      docs/appendices/coverage_takeover_20260925/online_points.csv; the fit
      is cross-checked against online_fits.csv (mean_3, equal_bin).
      No legend (panel (b) carries the shared bigram/trigram legend).
      The historical FIXED-probe panel remains available as
      fig03a_frequency.* (draw_frequency) for reference only.
  (b) capacity drawn LOG-LOG (user: "fig4还是得画双对数"), no fit line
      (user: "不用强行拟合"); non-positive gaps omitted from log display;
      carries the bigram/trigram legend shared with (a).
  (c) corpus dose with LINEAR y (user: "fig5需要y轴非对数") and x = training
      corpus tokens in millions (user: "横轴展示token的数量，单位是M").

Data (seed 42; input injection; table LR 128x backbone where a table exists):
  (a) docs/appendices/s1_scaling_three_axis/s1_frequency_exact_points.csv
      both tables enabled (2^20 rows each), fixed-probe gap at step 1000
      (run data/runs_scaling/s1v5_128_frequency_main_fixed; layers with
      >= MIN_CONTEXTS shared contexts and positive gap).
  (b) docs/appendices/s1_scaling_three_axis/s1_table_size_points.csv
      bigram-only / trigram-only final gap at step 1000, R = 1 .. 2.347e6
      (runs data/runs_scaling/s1v5_128_tbl_*_fixed).
  (c) docs/appendices/s1_scaling_three_axis/s1_dose_epoch_endpoints.csv
      trigram-only (2^20 rows), 20 corpus doses 0.125x..20x D1, gaps at the
      ends of epochs 2 and 3; x = epoch_batches * 72 * 2048 tokens / 1e6.

The online freq-bin record (s1_frequency_online_freqbin.jsonl, rerun 2026-09-22)
is kept for reference; --online redraws (a) in that caliber to
fig03a_frequency_online.* without touching the paper panel.

Outputs (docs/figs/paper/):
  fig03a_frequency.pdf/.png/.svg        (legacy fixed-probe reference)
  fig03b_capacity.pdf/.png
  fig03b_capacity.svg
  fig03c_corpus.pdf/.png
  fig03c_corpus.svg
  fig345_row_preview.png        (side-by-side montage, review only)
Outputs (docs/figs/coverage_20260925/, the paper panel (a)):
  fig03a_frequency_coverage.pdf/.png/.svg
Copy the three paper PDFs (coverage fig03a + fig03b + fig03c) to
overleaf-paper/figures/ with the same names.
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
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

AX = ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
OUT = ROOT / "docs" / "figs" / "paper"
COV_AX = ROOT / "docs" / "appendices" / "coverage_takeover_20260925"
COV_OUT = ROOT / "docs" / "figs" / "coverage_20260925"
COV_STEP = 1000
COV_SEEDS = 3  # seeds 42/43/44 of the coverage audit packet
ONLINE_JSONL = AX / "s1_frequency_online_freqbin.jsonl"
ONLINE_STEP = 1000

PANEL_W, PANEL_H = 2.2, 1.95
TOKENS_PER_BATCH = 72 * 2048  # device batch 72 x seq 2048 prediction tokens

# Saturated, shared branch colors for panels (a)/(b).  These stay close to the
# project palette while remaining distinct after PDF/SVG embedding and
# grayscale-ish projector rendering.
BIGRAM_COLOR = "#087F7B"
TRIGRAM_COLOR = "#7A3EAA"
EPOCH2_COLOR = "#2A9D8F"
EPOCH3_COLOR = "#D1493F"

# Fixed-probe canonical fits (paper text Sec. 4.1, Eq. gap-fits); drawn as the
# dashed lines of panel (a).
CANON_FIT = {"bigram": (7.206, -0.233), "trigram": (3.759, -0.318)}
MIN_CONTEXTS = 32
N_BINS = 12
OUTLIER_LOG_RES = 0.095  # |ln(gap/fit)| above this AND ...
OUTLIER_STD = 0.40      # ... bin spread (nats) above this -> error bar


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


def _save_panel(fig, name, outdir=OUT):
    outdir.mkdir(parents=True, exist_ok=True)
    # fixed canvas (no bbox_inches) so the three panels share exact dimensions
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.svg")
    fig.savefig(outdir / f"{name}.png", dpi=300)
    plt.close(fig)
    print((outdir / f"{name}.pdf").relative_to(ROOT))


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
    train_bins = set(rec["train"]["bigram"])
    trigram_bins = set(rec["train"]["trigram"])
    val_bins = set(rec["val"][branch])
    if train_bins != trigram_bins or train_bins != val_bins:
        raise SystemExit(
            "bigram/trigram frequency bins differ: "
            f"bigram-only={sorted(train_bins - trigram_bins)}, "
            f"trigram-only={sorted(trigram_bins - train_bins)}, "
            f"train-vs-val={sorted(train_bins - val_bins)}"
        )
    rows = []
    names = sorted(train_bins - {"novel"}, key=parse_bucket_f)
    for name in names:
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


def freq_rows(branch):
    """(F, fixed-probe gap, shared token mass) per exact-F layer at step 1000."""
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
        out.append((fbar, gbar, std, len(sel)))
    return out


def draw_frequency(ax, show_legend=False):
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        bins = geometric_bins(freq_rows(branch))
        color = BIGRAM_COLOR if branch == "bigram" else TRIGRAM_COLOR
        amp, slope = CANON_FIT[branch]
        bx = np.array([b[0] for b in bins])
        bg = np.array([b[1] for b in bins])
        bs = np.array([b[2] for b in bins])
        res = np.log(bg / (amp * bx ** slope))
        outl = (np.abs(res) > OUTLIER_LOG_RES) & (bs > OUTLIER_STD)
        ax.plot(bx, bg, marker=marker, ls="none", ms=3.8, color=color,
                zorder=3, label=branch)
        if outl.any():
            ax.errorbar(bx[outl], bg[outl], yerr=bs[outl], fmt="none",
                        ecolor="#666666", elinewidth=0.8, capsize=1.8,
                        capthick=0.8, zorder=2)
        fx = np.geomspace(bx.min(), bx.max(), 80)
        ax.plot(fx, amp * fx ** slope, ls="--", lw=1.1, color=color, zorder=4)
        for x, g, r, o, n in zip(bx, bg, res, outl, [b[3] for b in bins]):
            print(f"[fig03a] {branch} F={x:6.1f} gap={g:.3f} ln(gap/fit)={r:+.3f} "
                  f"layers={n}{'  <- error bar' if o else ''}")
    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, None)
    ax.set_ylim(10 ** -0.3, 10 ** 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xlabel("context frequency $F$")
    ax.set_ylabel("probe gap (nats)")


def draw_frequency_online(ax):
    """Reference redraw of (a) in the online caliber (not the paper panel)."""
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        rows = online_rows(branch)
        color = BIGRAM_COLOR if branch == "bigram" else TRIGRAM_COLOR
        fx = np.array([r[0] for r in rows])
        gy = np.array([r[1] for r in rows])
        ax.plot(fx, gy, marker=marker, ls="none", ms=3.8, color=color, label=branch)
        amp, slope = weighted_fit(rows)
        xs = np.geomspace(fx.min(), fx.max(), 80)
        ax.plot(xs, amp * xs ** slope, ls="--", lw=1.1, color=color)
        print(f"[fig03a-online] {branch}: fit amp={amp:.3f} slope={slope:.3f} (n={len(rows)})")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.8, None); ax.set_ylim(10 ** -1.8, 10 ** 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.set_xlabel("context frequency $F$"); ax.set_ylabel("online gap (nats)")


def coverage_frequency_rows(branch):
    """(F, novel fraction f, three-seed mean online gap) per native bin.

    Online caliber of the coverage audit packet (docs/appendices/
    coverage_takeover_20260925/): post-update fixed-val CE minus pre-update
    current-batch CE, both tables enabled, step COV_STEP, native bins with
    >= 300 tokens per side (18 bins, F = 1..1000).  f is a corpus property,
    identical across seeds within a bin.
    """
    per_bin = {}
    for r in csv.DictReader((COV_AX / "online_points.csv").open()):
        if (r["branch"] != branch or int(r["step"]) != COV_STEP
                or r["arm"] != "both"):
            continue
        key = (int(r["F_lo"]), int(r["F_hi"]))
        bin_ = per_bin.setdefault(key, {"f": float(r["f"]), "gaps": []})
        assert abs(bin_["f"] - float(r["f"])) < 1e-12, key
        bin_["gaps"].append(float(r["gap"]))
    rows = []
    for (lo, hi) in sorted(per_bin):
        gaps = per_bin[(lo, hi)]["gaps"]
        assert len(gaps) == COV_SEEDS, (branch, lo, len(gaps))
        rows.append((float(np.sqrt(lo * hi)), per_bin[(lo, hi)]["f"],
                     float(np.mean(gaps))))
    # Cross-check the equal-bin OLS against the registered fit of the packet.
    reg = {r["branch"]: (float(r["a"]), float(r["b"]))
           for r in csv.DictReader((COV_AX / "online_fits.csv").open())
           if r["step"] == str(COV_STEP) and r["seed"] == "mean_3"
           and r["weighting"] == "equal_bin"}
    x = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    a, b = reg[branch]
    assert abs(slope - a) < 1e-6 and abs(intercept - b) < 1e-6, \
        (branch, slope, intercept, a, b)
    return rows


def draw_frequency_coverage(ax):
    """Paper Fig 5(a): three-seed mean online gaps, equal-bin OLS LINEAR
    regression gap = a*f + b on the measured novel fraction (dashed, drawn
    against F), no error bars, no legend (panel (b) carries the legend)."""
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        rows = coverage_frequency_rows(branch)
        F = np.array([r[0] for r in rows])
        f = np.array([r[1] for r in rows])
        g = np.array([r[2] for r in rows])
        slope, intercept = np.polyfit(f, g, 1)
        color = BIGRAM_COLOR if branch == "bigram" else TRIGRAM_COLOR
        ax.plot(F, g, marker=marker, ls="none", ms=3.8, color=color,
                zorder=3, label=branch)
        ax.plot(F, slope * f + intercept, ls="--", lw=1.1, color=color,
                zorder=4)
        print(f"[fig03a-coverage] {branch}: gap = {slope:.3f}*f {intercept:+.3f} "
              f"(equal-bin OLS, {len(rows)} bins, 3-seed means)")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1200)
    ax.set_ylim(-0.4, 7.8)
    ax.set_xlabel("context frequency $F$")
    ax.set_ylabel("online gap (nats)")


def draw_capacity(ax, show_legend=True):
    rows = list(csv.DictReader((AX / "s1_table_size_points.csv").open()))
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        pts = sorted((int(r["R"]), float(r["gap_1000"]))
                     for r in rows if r["branch"] == branch)
        pos = [(R, g) for R, g in pts if g > 0]
        color = BIGRAM_COLOR if branch == "bigram" else TRIGRAM_COLOR
        ax.plot([p[0] for p in pos], [p[1] for p in pos], marker=marker,
                ls="none", ms=3.8, color=color,
                label=branch)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 5e6)
    ax.set_ylim(0.011, 6)
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    if show_legend:
        ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")
    ax.set_xlabel("table rows $R$")
    ax.set_ylabel("gap (nats)")


def draw_corpus(ax):
    rows = list(csv.DictReader((AX / "s1_dose_epoch_endpoints.csv").open()))
    pts = sorted((int(r["epoch_batches"]) * TOKENS_PER_BATCH / 1e6,
                  float(r["gap_epoch_2"]), float(r["gap_epoch_3"]))
                 for r in rows)
    x = [p[0] for p in pts]
    for idx, (color, marker, label) in enumerate((
            (EPOCH2_COLOR, "s", "end of epoch 2 (trigram only)"),
            (EPOCH3_COLOR, "^", "end of epoch 3 (trigram only)"))):
        ax.plot(x, [p[idx + 1] for p in pts], marker=marker, ms=3.6,
                lw=0.9, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlim(4.5, 1600)
    ax.set_ylim(-0.2, 4.1)
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1000"])
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_xlabel("corpus tokens (M)")
    ax.set_ylabel("gap (nats)")


def make_panel(draw, name, outdir=OUT):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    draw(ax)
    fig.tight_layout(pad=0.4)
    _save_panel(fig, name, outdir)


def make_preview():
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.15))
    for ax, draw, tag in zip(axes, (draw_frequency_coverage, draw_capacity,
                                    draw_corpus), "abc"):
        draw(ax)
        ax.set_title(f"({tag})", fontsize=9, pad=2)
    fig.tight_layout(pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig345_row_preview.png", dpi=300)
    fig.savefig(OUT / "fig345_row_preview.svg")
    plt.close(fig)
    print((OUT / "fig345_row_preview.png").relative_to(ROOT))


def main():
    apply_panel_style()
    if "--online" in sys.argv:
        make_panel(draw_frequency_online, "fig03a_frequency_online")
        return
    make_panel(draw_frequency_coverage, "fig03a_frequency_coverage", COV_OUT)
    make_panel(draw_frequency, "fig03a_frequency")
    make_panel(draw_capacity, "fig03b_capacity")
    make_panel(draw_corpus, "fig03c_corpus")
    make_preview()


if __name__ == "__main__":
    main()
