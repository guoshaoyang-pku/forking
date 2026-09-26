#!/usr/bin/env python3
"""Plot the compact optimization-selection figure for Appendix C.

All plotted values come from the archived parameter-selection claims and
manifest.  The figure replaces the three dense LaTeX tables with four small
panels covering the backbone rate, table-rate multiplier, RMSProp beta2, and
warmup-start screens.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "docs" / "appendices" / "parameter_selection_20260925"
CLAIMS = PACKET / "claims.json"
MANIFEST = PACKET / "parameter_selection_manifest.json"
OUT = ROOT / "docs" / "figs" / "appendix_c_parameter_selection"

import sys

sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S


RAW = S.ARM_COLORS["input"]
NET = "#68737d"
TABLE = S.BRANCH_COLORS["bigram"]
EXTENDED = S.BRANCH_COLORS["trigram"]
SELECTED = "#1a7f37"
NO_WARMUP = S.ARM_COLORS["y"]


def load_data():
    if not CLAIMS.exists() or not MANIFEST.exists():
        raise FileNotFoundError(
            "Run the parameter-selection evidence packet first: "
            f"missing {CLAIMS} or {MANIFEST}"
        )
    return json.loads(CLAIMS.read_text()), json.loads(MANIFEST.read_text())


def config_for(manifest, run_id):
    return manifest["runs"][run_id.removesuffix("_fixed")]["config"]


def setup_style():
    S.apply_style()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
            "svg.hashsalt": "appendix_c_parameter_selection",
        }
    )


def panel_label(ax, label):
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )


def selected_line(ax, x, label=None):
    ax.axvline(x, color=SELECTED, lw=0.85, ls=":", zorder=1)
    if label:
        ax.text(
            x,
            0.97,
            label,
            transform=ax.get_xaxis_transform(),
            color=SELECTED,
            ha="center",
            va="top",
            fontsize=7.2,
        )


def plot_locked(ax, claims, manifest):
    rows = claims["locked"]
    xs = np.array([config_for(manifest, r["input"]["run_id"])["nanogpt_adam_lr"] for r in rows])
    raw = np.array([r["input"]["gap"] for r in rows])
    no = np.array([r["nogram"]["gap"] for r in rows])
    net = raw - no
    order = np.argsort(xs)
    xs, raw, net = xs[order], raw[order], net[order]
    ax.plot(xs, raw, "-o", color=RAW, label="raw gap", zorder=3)
    ax.plot(xs, net, "--o", color=NET, label="net gap", zorder=3)
    selected_line(ax, 6e-4, r"$6\times10^{-4}$")
    ax.set_xscale("log")
    ax.set_xticks([6e-5, 3e-4, 6e-4, 2e-3, 4e-3])
    ax.set_xticklabels([r"$6\cdot10^{-5}$", r"$3\cdot10^{-4}$", r"$6\cdot10^{-4}$", r"$2\cdot10^{-3}$", r"$4\cdot10^{-3}$"], rotation=25, ha="right")
    ax.set_ylim(-0.2, 6.4)
    ax.set_xlabel(r"backbone LR $\eta_B$")
    ax.set_ylabel("gap at step 2000 (nats)")
    ax.legend(loc="upper left", handlelength=1.5, ncol=2)


def plot_table_scale(ax, claims):
    rows = sorted(claims["scale"], key=lambda r: r["scale"])
    x = np.array([r["scale"] for r in rows], dtype=float)
    y = np.array([r["gap"] for r in rows])
    ext = sorted(claims["extended"], key=lambda r: r["scale"])
    xe = np.array([r["scale"] for r in ext], dtype=float)
    ye = np.array([r["gap"] for r in ext])
    ax.plot(x, y, "-o", color=TABLE, label="step 1000", zorder=3)
    ax.plot(xe, ye, "--^", color=EXTENDED, label="step 2000", zorder=3)
    selected_line(ax, 128, "128x")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 2, 8, 32, 128, 512])
    ax.set_xticklabels(["0.5", "2", "8", "32", "128", "512"])
    ax.set_ylim(0, 6.5)
    ax.set_xlabel(r"table LR multiplier $s_T$")
    ax.set_ylabel("gap (nats)")
    ax.legend(loc="upper left", handlelength=1.5)


def plot_beta2(ax, claims, manifest):
    candidates = [r for r in claims["support"] if "rms_b" in r["run_id"] and "s2p0" in r["run_id"]]
    # Include the selected beta2 run, which is also the scale=2 point.
    scale2 = next(r for r in claims["scale"] if r["scale"] == 2)
    candidates.append(scale2)
    by_beta = {}
    for row in candidates:
        beta = config_for(manifest, row["run_id"])["table_betas"][1]
        by_beta[round(beta, 4)] = row["gap"]
    betas = np.array(sorted(by_beta))
    xs = np.arange(len(betas))
    ys = np.array([by_beta[x] for x in betas])
    ax.plot(xs, ys, "-o", color=TABLE, zorder=3)
    selected_line(ax, int(np.where(np.isclose(betas, 0.99))[0][0]), ".99")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in betas])
    ax.set_ylim(1.05, 1.85)
    ax.set_xlabel(r"table RMSProp $\beta_2$")
    ax.set_ylabel("gap at step 1000 (nats)")


def plot_warmup(ax, claims, manifest):
    rows = claims["warmup"]
    starts = []
    gaps = []
    for row in rows:
        cfg = config_for(manifest, row["run_id"])
        starts.append(None if cfg["lr_schedule"] == "constant" else cfg["warmup_start_lr_mult"])
        gaps.append(row["gap"])
    order = sorted(range(len(starts)), key=lambda i: (starts[i] is None, starts[i] or 0))
    starts = [starts[i] for i in order]
    gaps = np.array([gaps[i] for i in order])
    labels = ["none" if s is None else f"{s:g}" for s in starts]
    x = np.arange(len(labels))
    colors = [NO_WARMUP if s is None else TABLE for s in starts]
    ax.plot(x[:-1], gaps[:-1], color=TABLE, lw=1.1, zorder=2)
    ax.scatter(x, gaps, c=colors, s=24, zorder=3)
    selected_line(ax, 1, ".25")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 1.72)
    ax.set_xlabel("warmup start multiplier")
    ax.set_ylabel("gap at step 1000 (nats)")
    ax.text(x[-1], gaps[-1] + 0.08, "no warmup", color=NO_WARMUP, ha="center", fontsize=7.2)


def main():
    claims, manifest = load_data()
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.55))
    panel_label(axes[0, 0], "(a)")
    panel_label(axes[0, 1], "(b)")
    panel_label(axes[1, 0], "(c)")
    panel_label(axes[1, 1], "(d)")
    plot_locked(axes[0, 0], claims, manifest)
    plot_table_scale(axes[0, 1], claims)
    plot_beta2(axes[1, 0], claims, manifest)
    plot_warmup(axes[1, 1], claims, manifest)
    for ax in axes.flat:
        ax.grid(True, which="major", alpha=0.2, lw=0.55)
        ax.tick_params(axis="both", which="major", pad=2)
    fig.tight_layout(w_pad=1.15, h_pad=1.35, rect=(0, 0, 1, 0.985))
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [
        ("pdf", {"metadata": {"CreationDate": datetime(2026, 9, 25, tzinfo=timezone.utc)}}),
        ("svg", {"metadata": {"Date": None}}),
        ("png", {"dpi": 300}),
    ]:
        fig.savefig(OUT / f"fig_appendix_c_parameter_selection.{suffix}", bbox_inches="tight", pad_inches=0.025, **kwargs)
    svg = OUT / "fig_appendix_c_parameter_selection.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(OUT / "fig_appendix_c_parameter_selection.pdf")


if __name__ == "__main__":
    main()
