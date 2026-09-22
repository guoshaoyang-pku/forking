#!/usr/bin/env python3
"""Paper Sec.4.3 figure: random context loss masking, one continuous 3-epoch panel.

Experiment (collaborator's runs, both-table input model, seed 42): 2% of
training trigram contexts (set A) have their next-token loss excluded from
the epoch boundary onward; training continues on every other context for one
more epoch. B is a disjoint random unmasked peer set. Two paired runs: mask
from epoch 1 (segment 1->2) and mask from epoch 2 (segment 2->3), each with
its normal-training control from the same checkpoint. Plotted quantity is the
gap contribution C_S(t) = sum_{i in S} q_i [l_val,i(t) - l_tr,i(t)].

Data: docs/appendices/context_subspace_mask_20260921/
      fig_context_subspace_mask_extracted.csv -- curve coordinates recovered
      from the vector paths of the collaborator's two-panel figure
      overleaf-paper/figures/fig_context_subspace_mask.pdf (sha256 847f58bf...,
      Overleaf commit ea97c5c); calibration from the axis tick marks
      (x 1..2 / 2..3, y 0..0.03). Precision ~1e-4 nats. Replace with the raw
      run logs when the collaborator provides them.

Redesign: the two panels are spliced on a shared epoch axis. Orange = B
(unmasked peers), continuous across the boundary; blue dashed = A under
normal training (continuous); blue solid = A with loss masked from the
preceding epoch boundary, which leaves the dashed line at the boundary and
flattens. B's own normal-training control is omitted (it coincides with B
unmasked to within 1e-4 nats; see the CSV).

Output: docs/figs/paper/fig07_context_mask.pdf/.png. Copy the PDF to
overleaf-paper/figures/.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

SRC = ROOT / "docs" / "appendices" / "context_subspace_mask_20260921" / "fig_context_subspace_mask_extracted.csv"
OUT = ROOT / "docs" / "figs" / "paper"
BLUE, ORANGE = "#2d6f9f", "#d5622b"


def load():
    d = defaultdict(list)
    for r in csv.DictReader(SRC.open()):
        d[(r["panel"], r["series"])].append((float(r["epoch"]), float(r["gap_contribution"])))
    return d


def concat(d, series):
    return [p for panel in ("a", "b") for p in d[(panel, series)]]


def main():
    S.apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "legend.fontsize": 8})
    d = load()
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for e in (2,):
        ax.axvline(e, color="#888888", lw=0.7, ls=":")
    xs, ys = zip(*concat(d, "B_unmasked"))
    ax.plot(xs, ys, color=ORANGE, lw=1.6, label="B: unmasked peers")
    xs, ys = zip(*concat(d, "A_control"))
    ax.plot(xs, ys, color=BLUE, lw=1.2, ls="--", label="A: normal training")
    for i, panel in enumerate(("a", "b")):
        xs, ys = zip(*d[(panel, "A_masked")])
        ax.plot(xs, ys, color=BLUE, lw=1.6,
                label="A: loss masked from epoch boundary" if i == 0 else None)
        ax.plot(xs[0], ys[0], marker="o", ms=4, color=BLUE, mfc="white", zorder=5)
    ax.set_xlim(1, 3)
    ax.set_xticks([1, 1.5, 2, 2.5, 3])
    ax.set_xticklabels(["1", "1.5", "2", "2.5", "3"])
    ax.set_ylim(0, 0.033)
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig07_context_mask.pdf")
    fig.savefig(OUT / "fig07_context_mask.png", dpi=200)
    print((OUT / "fig07_context_mask.pdf").relative_to(ROOT))


if __name__ == "__main__":
    main()
