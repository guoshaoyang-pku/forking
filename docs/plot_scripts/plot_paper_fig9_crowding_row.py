#!/usr/bin/env python3
"""Paper Sec.4.2 crowding-out row: three panels on the SAME population.

Population: validation positions whose bigram context (t-1, t) occurs in
the training shard but whose true continuation t+1 never followed that
context in training ("seen context, novel continuation"; n = 158,790).
Arms: no-gram control vs input-route bigram+trigram model ("+both"), seed
42, table LR 128x, R = 2^20 clean tables, checkpoints at steps 337 / 674 /
1000 (epoch boundaries 1, 2, ~3).

  (a) probability mass the model puts on continuations that DID follow the
      context in training (top-32 mass; "crowding"): +both piles mass onto
      memorised continuations while the control barely moves.
  (b) cross-entropy of the true (novel) continuation: pushed up every pass
      for +both, pulled down for the control.
  (c) excess CE (+both minus control) of the novel continuation versus the
      context's training frequency bin, one line per epoch: damage grows
      with replay and decays with F -- the frequency law of Sec.3 at the
      token level.

Data: docs/appendices/marm_logits_20260916/marm_subspace_mass.json
      (copied from ophis-gpu data/runs_scaling/logits_rank/, produced by
      code/tools/logits_subspace_stats.py on eval_logits_rank.py dumps of
      runs s1v5_128_marm_{nogram,both}_{e1ckpt,e2ckpt,ckpt}_fixed).

Outputs (docs/figs/paper/): fig09a_crowd_mass.pdf, fig09b_crowd_ce.pdf,
fig09c_crowd_excess_vsF.pdf (+ .png) and fig09_crowding_preview.png.
Copy the PDFs to overleaf-paper/figures/ with the same names.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

SRC = ROOT / "docs" / "appendices" / "marm_logits_20260916" / "marm_subspace_mass.json"
OUT = ROOT / "docs" / "figs" / "paper"

PANEL_W, PANEL_H = 2.2, 1.95
BRANCH = "bigram"
EPOCHS = ("e1", "e2", "e3")
EPOCH_X = (1, 2, 3)
F_BINS = ("1", "2-3", "4-31", "32-511", "512+")
ARMS = (("nogram", S.ARM_COLORS["nogram"], "s", "no-gram"),
        ("both", S.ARM_COLORS["input"], "o", "+bigram+trigram"))
EPOCH_COLORS = {"e1": "#9ecae1", "e2": "#4292c6", "e3": "#08306b"}


def apply_panel_style():
    S.apply_style()
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
    })


def load():
    return json.loads(SRC.read_text())["arms"]


def cell(arms, arm, ep, grp):
    return arms[arm][ep]["val"][BRANCH][grp]


def _save_panel(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print((OUT / f"{name}.pdf").relative_to(ROOT))


def draw_mass(ax, arms):
    for arm, color, marker, label in ARMS:
        ys = [cell(arms, arm, ep, "ctxseen|novel")["seen_mass"] for ep in EPOCHS]
        ax.plot(EPOCH_X, ys, marker=marker, ms=4, lw=1.2, color=color, label=label)
    ax.set_xticks(EPOCH_X)
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mass on train-seen\ncontinuations")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")


def draw_ce(ax, arms):
    for arm, color, marker, label in ARMS:
        ys = [cell(arms, arm, ep, "ctxseen|novel")["ce_mean"] for ep in EPOCHS]
        ax.plot(EPOCH_X, ys, marker=marker, ms=4, lw=1.2, color=color, label=label)
    ax.set_xticks(EPOCH_X)
    ax.set_xlim(0.75, 3.25)
    ax.set_ylim(5, 9.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("CE of novel\ncontinuation (nats)")


def draw_excess(ax, arms):
    xs = list(range(len(F_BINS)))
    for ep, x_ep in zip(EPOCHS, EPOCH_X):
        ys = [cell(arms, "both", ep, f"ctx{b}|novel")["ce_mean"]
              - cell(arms, "nogram", ep, f"ctx{b}|novel")["ce_mean"] for b in F_BINS]
        ax.plot(xs, ys, marker="o", ms=3.6, lw=1.2, color=EPOCH_COLORS[ep],
                label=f"epoch {x_ep}")
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(F_BINS)
    ax.set_ylim(-0.3, 6.5)
    ax.set_xlabel("context frequency $F$")
    ax.set_ylabel("excess CE of novel\ncontinuation (nats)")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")


def make_panel(draw, arms, name):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    draw(ax, arms)
    fig.tight_layout(pad=0.4)
    _save_panel(fig, name)


def make_preview(arms):
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.15))
    for ax, draw, tag in zip(axes, (draw_mass, draw_ce, draw_excess), "abc"):
        draw(ax, arms)
        ax.set_title(f"({tag})", fontsize=9, pad=2)
    fig.tight_layout(pad=0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig09_crowding_preview.png", dpi=200)
    plt.close(fig)
    print((OUT / "fig09_crowding_preview.png").relative_to(ROOT))


def print_numbers(arms):
    g = "ctxseen|novel"
    print(f"population n = {cell(arms, 'both', 'e1', g)['n']:,}")
    for arm, *_ in ARMS:
        print(arm, "seen_mass", [round(cell(arms, arm, ep, g)["seen_mass"], 3) for ep in EPOCHS],
              "ce", [round(cell(arms, arm, ep, g)["ce_mean"], 2) for ep in EPOCHS],
              "p_true", [round(cell(arms, arm, ep, g)["true_prob_mean"], 3) for ep in EPOCHS])
    for ep in EPOCHS:
        print(ep, "excess by F", [round(cell(arms, "both", ep, f"ctx{b}|novel")["ce_mean"]
                                        - cell(arms, "nogram", ep, f"ctx{b}|novel")["ce_mean"], 2)
                                  for b in F_BINS])


def main():
    apply_panel_style()
    arms = load()
    print_numbers(arms)
    make_panel(draw_mass, arms, "fig09a_crowd_mass")
    make_panel(draw_ce, arms, "fig09b_crowd_ce")
    make_panel(draw_excess, arms, "fig09c_crowd_excess_vsF")
    make_preview(arms)


if __name__ == "__main__":
    main()
