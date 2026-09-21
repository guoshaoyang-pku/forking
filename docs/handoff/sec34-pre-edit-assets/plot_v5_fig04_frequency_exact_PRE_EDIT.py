#!/usr/bin/env python3
"""Fig 4 (v3): gap vs exact train hit-count f, log-log, 12 display bins.

Both branches in one figure (two panels sharing the y axis).  Display:
  * faint gray dots  = raw exact-f diagnostic points (positive gap,
    >= MIN_CONTEXTS shared contexts);
  * colored markers  = 12 geometric bins pooled by token mass (NOT connected);
  * thin gray yerr   = token-mass weighted std of the gap within each bin;
  * dashed line      = the *registered* power-law fit.  IMPORTANT: the
    exponents (-0.253 / -0.318) are the registry numbers from
    s1_scaling_fits.csv -- an equal-weight log-log regression over 7
    mass-pooled geometric bins (fit window [4, 4096]).  They are NOT refit
    here; the theory sections quote them.  The 12 display bins are a finer
    view of the same data; the fit line's amplitude is anchored to the
    display-bin means.

Outputs:
  * static windowed PNG+SVG  (y: 10^-0.3 .. 10^1)   fig_v5_s1_frequency_exact_f
  * interactive HTML with a 窗口/全图 toggle        fig_v5_s1_frequency_exact_interactive.html

Data: docs/appendices/s1_scaling_three_axis/s1_frequency_exact_points.csv
      docs/appendices/s1_scaling_three_axis/s1_scaling_fits.csv
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
PTS = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_frequency_exact_points.csv"
FITS = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_scaling_fits.csv"
OUT = ROOT / "docs" / "figs" / "main"

N_BINS = 12          # display bins (>= 10 per user request)
MIN_CONTEXTS = 32
Y_LO, Y_HI = 10 ** -0.3, 10 ** 1.0


def load_rows(branch):
    return [(float(r["f"]), float(r["gap"]), float(r["shared_token_mass"]))
            for r in csv.DictReader(PTS.open())
            if r["branch"] == branch and int(r["step"]) == 1000
            and float(r["gap"]) > 0 and int(r["shared_contexts"]) >= MIN_CONTEXTS]


def geometric_bins(rows, n_bins=N_BINS):
    fs = np.array([r[0] for r in rows])
    lo, hi = fs.min(), fs.max()
    edges = np.geomspace(lo, hi * 1.0001, n_bins + 1)
    bins = []
    for i in range(n_bins):
        sel = [r for r in rows if edges[i] <= r[0] < edges[i + 1]]
        if not sel:
            continue
        f = np.array([s[0] for s in sel])
        g = np.array([s[1] for s in sel])
        w = np.array([s[2] for s in sel])
        fbar = np.exp(np.average(np.log(f), weights=w))
        gbar = np.average(g, weights=w)
        # token-mass weighted std of the gap within the bin
        var = np.average((g - gbar) ** 2, weights=w)
        bins.append((fbar, gbar, np.sqrt(var), f.min(), f.max()))
    return bins


def registered_fits():
    return {r["branch"]: float(r["slope"]) for r in csv.DictReader(FITS.open())
            if r["family"] == "frequency_exact"}


def main():
    S.apply_style()
    fits = registered_fits()
    data = {b: load_rows(b) for b in ("bigram", "trigram")}
    bins = {b: geometric_bins(data[b]) for b in data}

    # ---------- static windowed figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, branch in zip(axes, ("bigram", "trigram")):
        color = S.BRANCH_COLORS[branch]
        f = np.array([r[0] for r in data[branch]])
        g = np.array([r[1] for r in data[branch]])
        ax.scatter(f, g, s=7, color="#9aa3ad", alpha=0.45, zorder=2,
                   label="exact-f points (positive gap)")
        bx = np.array([b[0] for b in bins[branch]])
        bg = np.array([b[1] for b in bins[branch]])
        bs = np.array([b[2] for b in bins[branch]])
        ax.errorbar(bx, bg, yerr=bs, fmt="o", color=color, ms=5, lw=0,
                    elinewidth=0.8, ecolor="#666666", capsize=2, capthick=0.8,
                    zorder=3, label=f"{N_BINS} geometric bins (mass-weighted)")
        slope = fits[branch]
        fx = np.geomspace(bx.min(), bx.max(), 80)
        amp = np.exp(np.mean(np.log(bg) - slope * np.log(bx)))
        ax.plot(fx, np.exp(np.log(amp) + slope * np.log(fx)), ls="--", lw=1.2,
                color=S.HOLDOUT_COLOR, zorder=4,
                label=f"registered fit: $G \\propto f^{{{slope:.3f}}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_xlabel("exact train hit-count per context f")
        ax.set_title(branch)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("gap (val - train probe) @ step 1000")
    fig.tight_layout()
    png, svg = S.save(fig, OUT, "fig_v5_s1_frequency_exact_f")
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))

    # ---------- interactive window/full toggle ----------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_g = np.concatenate([np.array([r[1] for r in data[b]]) for b in data])
    full_lo = float(np.log10(all_g.min())) - 0.08
    full_hi = float(np.log10(all_g.max())) + 0.08

    fig2 = make_subplots(cols=2, rows=1, shared_yaxes=True,
                         subplot_titles=("bigram", "trigram"),
                         horizontal_spacing=0.05)
    for col, branch in enumerate(("bigram", "trigram"), start=1):
        color = S.BRANCH_COLORS[branch]
        f = [r[0] for r in data[branch]]
        g = [r[1] for r in data[branch]]
        fig2.add_trace(go.Scatter(
            x=f, y=g, mode="markers", legendgroup="raw", name="exact-f points",
            showlegend=(col == 1),
            marker=dict(size=3.5, color="#9aa3ad", opacity=0.35),
            hovertemplate="f=%{x}<br>gap=%{y:.3f}<extra>exact-f</extra>"),
            row=1, col=col)
        bx = [b[0] for b in bins[branch]]
        bg = [b[1] for b in bins[branch]]
        bs = [b[2] for b in bins[branch]]
        fig2.add_trace(go.Scatter(
            x=bx, y=bg, mode="markers", legendgroup="bins",
            name=f"{N_BINS} geometric bins", showlegend=(col == 1),
            error_y=dict(type="data", array=bs, color="#666666",
                         thickness=1.0, width=4),
            marker=dict(size=8, color=color),
            hovertemplate="f≈%{x:.1f}<br>gap=%{y:.3f}<extra>bin (mass-weighted)</extra>"),
            row=1, col=col)
        slope = fits[branch]
        fx = np.geomspace(min(bx), max(bx), 80)
        amp = np.exp(np.mean(np.log(np.array(bg)) - slope * np.log(np.array(bx))))
        fig2.add_trace(go.Scatter(
            x=list(fx), y=list(np.exp(np.log(amp) + slope * np.log(fx))),
            mode="lines", legendgroup="fit", showlegend=(col == 1),
            name=f"registered fit (7-bin equal-weight)",
            line=dict(color=S.HOLDOUT_COLOR, dash="dash", width=1.4),
            hovertemplate=f"G ∝ f^{slope:.3f}<extra>fit</extra>"),
            row=1, col=col)

    window = [float(np.log10(Y_LO)), float(np.log10(Y_HI))]
    buttons = [
        dict(label="窗口视图 (y: 10⁻⁰·³–10)", method="relayout",
             args=[{"yaxis.range": window, "yaxis2.range": window}]),
        dict(label="全图视图", method="relayout",
             args=[{"yaxis.range": [full_lo, full_hi],
                    "yaxis2.range": [full_lo, full_hi]}]),
    ]
    fig2.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.02, xanchor="left",
                          y=1.22, yanchor="top", bgcolor="#fff",
                          bordercolor="#ccc")],
        annotations=[dict(text="y 轴范围", x=0.02, xref="paper",
                          xanchor="left", y=1.30, yref="paper",
                          showarrow=False, font=dict(size=12, color="#666"))],
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="-apple-system, 'Helvetica Neue', Arial, sans-serif",
                  size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.3),
        margin=dict(l=60, r=20, t=90, b=50), height=470,
    )
    axis_line = dict(showline=True, linecolor="#555", linewidth=1.0,
                     ticks="outside", gridcolor="#e6e6e6",
                     exponentformat="power", dtick=1)
    for ax in ("xaxis", "xaxis2"):
        fig2.update_layout({ax: dict(type="log",
                                     title="exact train hit-count per context f",
                                     **axis_line)})
    for ax in ("yaxis", "yaxis2"):
        fig2.update_layout({ax: dict(type="log", range=window, **axis_line)})
    fig2.update_layout(yaxis_title="gap (val − train probe) @ step 1000")

    html = OUT / "fig_v5_s1_frequency_exact_interactive.html"
    div = fig2.to_html(full_html=False, include_plotlyjs="plotly.min.js",
                       div_id="fig4x")
    html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{margin:0;background:#fff}</style></head>"
        f"<body>{div}</body></html>")
    print(html.relative_to(ROOT))


if __name__ == "__main__":
    main()
