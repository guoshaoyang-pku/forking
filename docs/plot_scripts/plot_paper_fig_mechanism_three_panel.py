#!/usr/bin/env python3
"""Three-panel mechanism figure for ICLR paper Section 4.3.

Output: docs/figs/paper/fig_mechanism_three_panel.{pdf,png}

Panels (left to right, single row, total width = 396 pt = ICLR \\linewidth):
  (a) Continuation mass: train-seen mass (left axis, %) and val-only mass
      (right axis, %) over epochs 1-4, for ngram/nogram x masked/unmasked.
  (b) Gap dynamics: fixed-probe gap contribution C_S (nats) vs epoch,
      four series (normal training, unmasked peers, mask@ep2, mask@ep3).
  (c) Seen vs novel validation CE: five epochs, both-table and no-gram,
      train-seen vs novel continuations of train-seen bigram contexts.

Data sources (all artifact-traceable, no hand-entered numbers):
  Panel (a) train-seen mass: appendix_f_mask_plotted_values.json -> seen_mass_panel
  Panel (a) val-only mass: context_kl/kl_restricted_split_e1_l4.json -> z_valn_*
  Panel (b): runs_scaling/s1v5_128_marm_both_ctxmask_{ctrl,e1,e2}_l4_fixed/context_gap.jsonl
  Panel (c): fig09_crowd_ce_seen_novel_extracted.csv (vector-path extraction
             from paper/figures/fig09_crowd_ce_seen_novel.pdf, sha256 in README)

Style: v5_style.py conventions, base font 6.5-7 pt at 396 pt display width.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

OUT = ROOT / "docs" / "figs" / "paper"
APPENDIX = ROOT / "docs" / "appendices" / "context_subspace_mask_20260921"
CROWD_APPENDIX = ROOT / "docs" / "appendices" / "fig09_crowd_ce_extracted_20260926"

# Colours matching v5_style + existing fig09_dynamics palette
BLUE = S.ARM_COLORS["input"]       # #2d6f9f  ngram / normal-training
GREY = S.ARM_COLORS["nogram"]      # #686d73  no-gram
ORANGE = "#d5622b"                 # unmasked peers (matches fig09_dynamics)
PURPLE = "#7b4aa2"                 # mask-at-epoch-3

EPOCH_STEPS = 337


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_panel_a():
    """Return (epochs, train_seen_dict, val_only_dict).

    train_seen_dict: {label: [ep1, ep2, ep3, ep4]}
    val_only_dict:   {label: [ep1, ep2, ep3, ep4]}
    label in {both_masked, both_unmasked, nogram_masked, nogram_unmasked}
    """
    manifest = _load_json(APPENDIX / "appendix_f_mask_plotted_values.json")
    ts = manifest["seen_mass_panel"]["values"]

    split = _load_json(APPENDIX / "context_kl" / "kl_restricted_split_e1_l4.json")
    pairs = split["pairs"]

    epochs = [1, 2, 3, 4]
    train_seen = {k: [ts[k][str(e)] for e in epochs] for k in ts}

    # Val-only mass: z_valn_ref_mean at epoch 1 (reference), z_valn_new_mean at 2-4
    val_only = {}
    for arm, prefix in [("both", "both"), ("nogram", "nogram")]:
        for cohort, suffix in [("A", "masked"), ("B", "unmasked")]:
            label = f"{prefix}_{suffix}"
            e2 = pairs[f"{arm}_e2"]["sets"][cohort]
            vals = [e2["z_valn_ref_mean"]]
            for ep in [2, 3, 4]:
                vals.append(pairs[f"{arm}_e{ep}"]["sets"][cohort]["z_valn_new_mean"])
            val_only[label] = vals

    return epochs, train_seen, val_only


def load_panel_b():
    """Cohort mean fixed-probe gap, val_mean - tr_mean (nats), vs epoch.

    Returns list of (label, style_dict, epochs_array, gap_array).
    """
    runs_dir = APPENDIX / "runs_scaling"
    specs = [
        ("ctrl", ("val_mean_A", "tr_mean_A"),
         dict(color=BLUE, lw=0.8, ls="--"), "normal"),
        ("ctrl", ("val_mean_B", "tr_mean_B"),
         dict(color=ORANGE, lw=0.8), "peers"),
        ("e1", ("val_mean_A", "tr_mean_A"),
         dict(color=BLUE, lw=0.8), "mask ep 2"),
        ("e2", ("val_mean_A", "tr_mean_A"),
         dict(color=PURPLE, lw=0.8), "mask ep 3"),
    ]
    result = []
    for run_infix, (vk, tk), style, label in specs:
        pth = runs_dir / f"s1v5_128_marm_both_ctxmask_{run_infix}_l4_fixed" / "context_gap.jsonl"
        rows = [json.loads(l) for l in pth.open()]
        rows.sort(key=lambda r: r["step"])
        xs, ys = [], []
        for r in rows:
            v, t = r.get(vk), r.get(tk)
            if v is None or t is None or v != v or t != t:
                continue
            xs.append(r["step"] / EPOCH_STEPS)
            ys.append(v - t)
        result.append((label, style, np.array(xs), np.array(ys)))
    return result


def load_panel_c():
    """Return (epochs, {series_label: [ce_values]})."""
    csv_path = CROWD_APPENDIX / "fig09_crowd_ce_seen_novel_extracted.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    series = {
        "ngram, seen":  [float(r["ngram_seen"]) for r in rows],
        "ngram, novel": [float(r["ngram_novel"]) for r in rows],
        "no-gram, seen":  [float(r["nogram_seen"]) for r in rows],
        "no-gram, novel": [float(r["nogram_novel"]) for r in rows],
    }
    return epochs, series


LW = 0.8
MS = 2.6
MEW = 0.6


def _line(ax, xs, ys, color, ls, marker, filled):
    ax.plot(xs, ys, color=color, ls=ls, lw=LW, marker=marker, ms=MS,
            mfc=(color if filled else "white"), mec=color, mew=MEW, zorder=3)


def draw_panel_a(ax_t, ax_b, epochs, train_seen, val_only):
    """Two compact stacked rows sharing the epoch axis."""
    from matplotlib.lines import Line2D
    specs = [
        ("both_unmasked",  BLUE, "-",  True,  "n-gram unmasked"),
        ("both_masked",    BLUE, "--", False, "n-gram masked"),
        ("nogram_unmasked", GREY, "-",  True,  "no-gram unmasked"),
        ("nogram_masked",   GREY, "--", False, "no-gram masked"),
    ]
    legend_elements = []
    for key, color, ls, filled, legend_label in specs:
        _line(ax_t, epochs, train_seen[key], color, ls, "o", filled)
        _line(ax_b, epochs, val_only[key], color, ls, "s", filled)
        legend_elements.append(Line2D([0], [0], color=color, ls=ls, lw=LW,
                                       marker="o", ms=MS,
                                       mfc=(color if filled else "white"),
                                       mec=color, label=legend_label))
    for ax in (ax_t, ax_b):
        ax.set_xlim(0.7, 4.3)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, pad=1.5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}"))
    ax_t.set_ylabel("train-seen (%)")
    ax_t.set_ylim(0.12, 0.75)
    ax_t.tick_params(labelbottom=False)
    ax_t.legend(handles=legend_elements, loc="lower left",
                bbox_to_anchor=(-0.16, 1.06), fontsize=5.5, frameon=False,
                handlelength=1.5, labelspacing=0.15, borderpad=0.0,
                ncol=2, columnspacing=1.0, handletextpad=0.35)
    ax_b.set_xlabel("epoch")
    ax_b.set_ylabel("val-only (%)")
    ax_b.set_ylim(0.0, 0.070)


def draw_panel_b(ax, series_list):
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean gap (nats)")
    for label, style, xs, ys in series_list:
        ax.plot(xs, ys, label=label, zorder=3, **style)
    for b in range(1, 4):
        ax.axvline(b, color="#888888", lw=0.4, ls=":", zorder=1)
    ax.legend(loc="upper left", fontsize=5.5, frameon=False,
              handlelength=1.5, labelspacing=0.15, borderpad=0.0, ncol=2,
              columnspacing=0.9, handletextpad=0.35)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)


def draw_panel_c(ax, epochs, series):
    ax.set_xlim(0.7, 5.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation CE (nats)")
    specs = [
        ("no-gram, novel", GREY, "--", "o", True),
        ("no-gram, seen",  GREY, "-",  "o", False),
        ("ngram, novel",   "#d5622b", "--", "s", True),
        ("ngram, seen",    BLUE, "-",  "s", False),
    ]
    for key, color, ls, marker, filled in specs:
        ax.plot(epochs, series[key], color=color, ls=ls, lw=LW, marker=marker,
                ms=MS, mfc=(color if filled else "white"), mec=color, mew=MEW,
                zorder=3, label=key)
    ax.legend(loc="upper left", fontsize=5.5, frameon=False,
              handlelength=1.5, labelspacing=0.15, borderpad=0.0,
              handletextpad=0.35)
    ax.set_ylim(2, 12)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)


def main():
    S.apply_style()
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
    })

    epochs_a, train_seen, val_only = load_panel_a()
    series_b = load_panel_b()
    epochs_c, series_c = load_panel_c()

    # Native widths match the LaTeX subfigure fractions of \linewidth
    # (396 pt): (a) 0.36, (b) 0.34, (c) 0.28, so no rescaling is needed.
    H = 1.70
    sizes = {"a": (0.36 * 396 / 72.0, H),
             "b": (0.34 * 396 / 72.0, H),
             "c": (0.28 * 396 / 72.0, H)}

    OUT.mkdir(parents=True, exist_ok=True)
    made = {}

    fig = plt.figure(figsize=sizes["a"])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.62)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_t)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.755, bottom=0.19)
    draw_panel_a(ax_t, ax_b, epochs_a, train_seen, val_only)
    made["a"] = OUT / "fig_mechanism_a.pdf"
    fig.savefig(made["a"]); fig.savefig(made["a"].with_suffix(".png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=sizes["b"])
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.185, right=0.985, top=0.985, bottom=0.19)
    draw_panel_b(ax, series_b)
    made["b"] = OUT / "fig_mechanism_b.pdf"
    fig.savefig(made["b"]); fig.savefig(made["b"].with_suffix(".png"), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=sizes["c"])
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.225, right=0.985, top=0.985, bottom=0.19)
    draw_panel_c(ax, epochs_c, series_c)
    made["c"] = OUT / "fig_mechanism_c.pdf"
    fig.savefig(made["c"]); fig.savefig(made["c"].with_suffix(".png"), dpi=300)
    plt.close(fig)

    for k, v in made.items():
        print(f"{k}: {v.relative_to(ROOT)}")

    manifest = {
        "scope": "Section 4.3 context-subspace evidence, three LaTeX subfigures",
        "latex_widths": {"a": 0.36, "b": 0.34, "c": 0.28},
        "native_size_pt": {k: [round(w * 72, 1), round(h * 72, 1)]
                           for k, (w, h) in sizes.items()},
        "base_font_pt": 7,
        "script": "docs/plot_scripts/plot_paper_fig_mechanism_three_panel.py",
        "panel_a": {
            "train_seen_source": "docs/appendices/context_subspace_mask_20260921/appendix_f_mask_plotted_values.json:seen_mass_panel.values",
            "train_seen_sha256": _sha256(APPENDIX / "appendix_f_mask_plotted_values.json"),
            "val_only_source": "docs/appendices/context_subspace_mask_20260921/context_kl/kl_restricted_split_e1_l4.json:z_valn_*",
            "val_only_sha256": _sha256(APPENDIX / "context_kl" / "kl_restricted_split_e1_l4.json"),
            "metric": "mean full-vocabulary probability mass on train-seen (top) and validation-only (bottom) continuations at fixed validation positions",
        },
        "panel_b": {
            "source_runs": ["s1v5_128_marm_both_ctxmask_ctrl_l4_fixed",
                            "s1v5_128_marm_both_ctxmask_e1_l4_fixed",
                            "s1v5_128_marm_both_ctxmask_e2_l4_fixed"],
            "source_sha256": {r: _sha256(APPENDIX / "runs_scaling" /
                              f"s1v5_128_marm_both_ctxmask_{r}_l4_fixed" /
                              "context_gap.jsonl")
                              for r in ("ctrl", "e1", "e2")},
            "metric": "cohort mean fixed-probe gap val_mean_S - tr_mean_S (nats); "
                      "the share-weighted contribution C_S of Appendix Eq. "
                      "(context-gap-contribution) multiplies this by each "
                      "context's validation-position share and is therefore "
                      "~70x smaller",
        },
        "panel_c": {
            "extracted_csv": "docs/appendices/fig09_crowd_ce_extracted_20260926/fig09_crowd_ce_seen_novel_extracted.csv",
            "extracted_csv_sha256": _sha256(CROWD_APPENDIX / "fig09_crowd_ce_seen_novel_extracted.csv"),
            "metric": "position-weighted validation cross-entropy (nats) for train-seen bigram contexts split by continuation type",
        },
    }
    mp = APPENDIX / "fig_mechanism_three_panel_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {mp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
