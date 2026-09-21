#!/usr/bin/env python3
"""Publication-style continued-pretraining figures for the SFT section.

The source is the six dense freq=10 JSONL evidence runs. Each output figure
keeps one table policy fixed and combines ID, Magicoder, and OpenR1 panels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
from v5_style import apply_style, add_epoch_lines  # noqa: E402

SOURCE = ROOT / "docs" / "appendices" / "continue_pretrain_20260920_freq10"
OUT = ROOT / "docs" / "figs" / "continue_pretrain_20260920_freq10"
RUNS = (
    ("ID", "sft_id_tri_e5_freq10_{arm}_s42", "#2d6f9f"),
    ("Magicoder", "sft_ood_magicoder_e5_freq10_{arm}_s42", "#c4493d"),
    ("OpenR1", "sft_ood_openr1_10shards_e5_freq10_{arm}_s42", "#c58a0b"),
)
EXPECTED = list(range(10, 1661, 10)) + [1665]
LOSS_UPPER = {"Magicoder": 5.0, "OpenR1": 3.5}


def load(run_id: str) -> dict[str, np.ndarray]:
    path = SOURCE / run_id / "train_log.jsonl"
    rows = sorted((json.loads(x) for x in path.read_text().splitlines() if x.strip()), key=lambda x: x["step"])
    steps = [row["step"] for row in rows]
    if steps != EXPECTED:
        raise ValueError(f"unexpected cadence in {path}: {steps[:3]} ... {steps[-3:]}")
    return {key: np.asarray([row[key] for row in rows], dtype=float) for key in ("step", "train_loss", "val_loss", "gap")}


def bounds(values: np.ndarray, kind: str) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = max((hi - lo) * 0.10, 0.05)
    if kind == "gap":
        lo = min(lo - pad, -0.05)
        hi = max(hi + pad, 0.05)
    else:
        lo -= pad
        hi += pad
    return lo, hi


def plot_arm(arm: str) -> tuple[Path, Path]:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(5.075, 3.255), sharex="col", gridspec_kw={"height_ratios": [1.0, 0.86], "hspace": 0.18, "wspace": 0.28})
    fig.subplots_adjust(left=0.075, right=0.995, top=0.89, bottom=0.14)
    loaded = []
    for label, template, color in RUNS:
        run_id = template.format(arm=arm)
        loaded.append((label, run_id, color, load(run_id)))

    for col, (label, run_id, color, data) in enumerate(loaded):
        step = data["step"]
        ax_loss, ax_gap = axes[0, col], axes[1, col]
        ax_loss.plot(step, data["train_loss"], color=color, lw=1.35, label="train")
        ax_loss.plot(step, data["val_loss"], color=color, lw=1.15, ls="--", alpha=0.72, label="validation")
        ax_gap.plot(step, data["gap"], color=color, lw=1.35, label="gap")
        ax_gap.axhline(0.0, color="#80868b", lw=0.65, ls=":", zorder=0)
        for ax in (ax_loss, ax_gap):
            add_epoch_lines(ax, epoch_len=333, xmax=1665, color="#b8bec4")
            ax.set_xlim(0, 1665)
            ax.grid(True, axis="y")
            ax.tick_params(axis="both", pad=2)
        ax_loss.set_title(label, fontsize=10.5, pad=5, fontweight="medium")
        loss_lo, loss_hi = bounds(np.concatenate([data["train_loss"], data["val_loss"]]), "loss")
        if label in LOSS_UPPER:
            loss_hi = LOSS_UPPER[label]
        ax_loss.set_ylim(loss_lo, loss_hi)
        ax_gap.set_ylim(*bounds(data["gap"], "gap"))
        ax_loss.set_xticklabels([])
        ax_gap.set_xticks([0, 333, 666, 999, 1332, 1665])
        ax_gap.set_xticklabels(["0", "1", "2", "3", "4", "5"])
        ax_gap.tick_params(axis="x", labelsize=8)
        if col == 0:
            ax_loss.set_ylabel("loss (nats)", labelpad=3)
            ax_gap.set_ylabel("gap (nats)", labelpad=3)
        for spine in ("top", "right"):
            ax_loss.spines[spine].set_visible(False)
            ax_gap.spines[spine].set_visible(False)

    axes[1, 1].set_xlabel("replay epoch", labelpad=5)
    # Line semantics, sampling cadence, and epoch-boundary details belong in
    # the LaTeX caption. Keeping them out of the image leaves room for the
    # independent y scales and makes the asset reusable in the paper.
    png = OUT / f"fig_sft_{arm}_paper.png"
    pdf = OUT / f"fig_sft_{arm}_paper.pdf"
    svg = OUT / f"fig_sft_{arm}_paper.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ("trainable", "frozen"):
        png, pdf = plot_arm(arm)
        print(png)
        print(pdf)


if __name__ == "__main__":
    main()
