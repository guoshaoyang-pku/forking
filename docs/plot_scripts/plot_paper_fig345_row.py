#!/usr/bin/env python3
"""Paper scaling row: frequency, capacity, and corpus size.

Panel (a): online gap, both tables (2^20 rows), input injection, table LR
128x, step 1000, seeds 42/43/44. Read coverage_takeover_20260925/online_points.csv;
plot all 18 native bins with 1 <= F <= 1000 using geometric midpoints and
three-seed arithmetic means. Dashed power laws use equal-bin log-log OLS.
The directly measured novel fractions use the identical fit protocol.
Panel (b): single-table capacity sweep, seed 42, online gap at step 1000,
log-log axes without a fitted line; carries the shared branch legend.
Panel (c): trigram-only corpus sweep, seed 42, online epoch-2/3 endpoints,
log x in millions of prediction tokens, linear y.

Current panel (a) is docs/figs/coverage_20260925/fig03a_frequency_coverage.*;
(b,c) and the review montage are in docs/figs/paper/.
--frequency-only regenerates only the current panel (a).
--online retains an older single-seed online reference; draw_frequency is
an unused historical fixed-probe renderer, not a source for the paper.
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

# Historical hard-coded curves, retained only for the unused legacy renderer.
# They are not recomputed fits and must not be cited as current results.
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
    svg = outdir / f"{name}.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
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


def powerlaw_fit(x, y):
    """Equal-bin OLS of log(y) on log(x); all supplied bins must be positive."""
    x, y = np.asarray(x), np.asarray(y)
    assert np.all(x > 0) and np.all(y > 0)
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    residual = ly - (slope * lx + intercept)
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(np.exp(intercept)), float(-slope), float(r2)


def draw_frequency_coverage(ax):
    """Current Fig 5(a): mean online gaps and equal-bin log-log fits.

    Both branches keep all 18 bins. No error bars; panel (b) carries the legend.
    The affine gap-versus-novelty fits remain in the appendix coverage plot.
    """
    for branch, marker in (("bigram", "o"), ("trigram", "^")):
        rows = coverage_frequency_rows(branch)
        F, f, g = np.array(rows).T
        assert len(rows) == 18 and F.min() == 1 and F.max() <= 1000
        amp, beta, r2 = powerlaw_fit(F, g)
        _, alpha, _ = powerlaw_fit(F, f)
        color = BIGRAM_COLOR if branch == "bigram" else TRIGRAM_COLOR
        ax.plot(F, g, marker=marker, ls="none", ms=3.8, color=color,
                zorder=3, label=branch)
        x = np.geomspace(F.min(), F.max(), 100)
        ax.plot(x, amp * x ** (-beta), ls="--", lw=1.1, color=color,
                zorder=2)
        print(f"[fig03a-powerlaw] {branch}: gap = {amp:.6f} F^-{beta:.6f}; "
              f"R2_log={r2:.6f}; novelty alpha={alpha:.6f}; "
              f"{len(rows)} bins, 3-seed means")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 1200)
    ax.set_ylim(0.035, 10)
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
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
            (EPOCH2_COLOR, "s", "epoch 2"),
            (EPOCH3_COLOR, "^", "epoch 3"))):
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
    svg = OUT / "fig345_row_preview.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print((OUT / "fig345_row_preview.png").relative_to(ROOT))


def main():
    apply_panel_style()
    if "--online" in sys.argv:
        make_panel(draw_frequency_online, "fig03a_frequency_online")
        return
    make_panel(draw_frequency_coverage, "fig03a_frequency_coverage", COV_OUT)
    if "--frequency-only" in sys.argv:
        return
    make_panel(draw_capacity, "fig03b_capacity")
    make_panel(draw_corpus, "fig03c_corpus")
    make_preview()


if __name__ == "__main__":
    main()
