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
    """Return list of (label, style_dict, epochs_array, cs_array)."""
    runs_dir = APPENDIX / "runs_scaling"
    specs = [
        ("ctrl", "C_A", dict(color=BLUE, lw=1.0, ls="--"), "normal training"),
        ("ctrl", "C_B", dict(color=ORANGE, lw=1.4), "unmasked peers"),
        ("e1",   "C_A", dict(color=BLUE, lw=1.4), "mask at epoch 2"),
        ("e2",   "C_A", dict(color=PURPLE, lw=1.4), "mask at epoch 3"),
    ]
    result = []
    for run_infix, key, style, label in specs:
        p = runs_dir / f"s1v5_128_marm_both_ctxmask_{run_infix}_l4_fixed" / "context_gap.jsonl"
        rows = [json.loads(l) for l in p.open()]
        rows.sort(key=lambda r: r["step"])
        xs, ys = [], []
        for r in rows:
            v = r.get(key)
            if v is None or v != v:
                continue
            xs.append(r["step"] / EPOCH_STEPS)
            ys.append(v)
        result.append((label, style, np.array(xs), np.array(ys)))
    return result


def load_panel_c():
    """Return (epochs, {series_label: [ce_values]})."""
    csv_path = CROWD_APPENDIX / "fig09_crowd_ce_seen_novel_extracted.csv"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    epochs = [int(r["epoch"]) for r in rows]
    series = {
        "ngram, seen":  [float(r["ngram_seen"]) for r in rows],
        "ngram, novel": [float(r["ngram_novel"]) for r in rows],
        "no-gram, seen":  [float(r["nogram_seen"]) for r in rows],
        "no-gram, novel": [float(r["nogram_novel"]) for r in rows],
    }
    return epochs, series


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_panel_a(ax_t, ax_b, epochs, train_seen, val_only):
    """Continuation mass as two stacked rows sharing the epoch axis.

    Top row: mean probability mass on train-seen continuations (%).
    Bottom row: mean probability mass on validation-only continuations (%).
    """
    series_specs = [
        ("both_unmasked",  BLUE, "-",  True,  "N-gram unmasked"),
        ("both_masked",    BLUE, "--", False, "N-gram masked"),
        ("nogram_unmasked", GREY, "-",  True,  "No-gram unmasked"),
        ("nogram_masked",   GREY, "--", False, "No-gram masked"),
    ]

    from matplotlib.lines import Line2D
    legend_elements = []
    for key, color, ls, filled, legend_label in series_specs:
        mfc = color if filled else "white"
        for ax, values, marker in ((ax_t, train_seen[key], "o"),
                                   (ax_b, val_only[key], "s")):
            ax.plot(epochs, values, color=color, ls=ls, lw=1.3,
                    marker=marker, ms=3.5, mfc=mfc, mec=color, mew=0.8,
                    zorder=3)
        legend_elements.append(Line2D([0], [0], color=color, ls=ls, lw=1.3,
                                       marker="o", ms=3.5, mfc=mfc, mec=color,
                                       label=legend_label))

    for ax in (ax_t, ax_b):
        ax.set_xlim(0.7, 4.3)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}"))

    ax_t.set_ylabel("train-seen mass (%)")
    ax_t.set_ylim(0.12, 0.75)
    ax_t.tick_params(labelbottom=False)
    ax_t.legend(handles=legend_elements, loc="lower left",
                bbox_to_anchor=(0.10, 1.02), fontsize=5.0, framealpha=0.0,
                edgecolor="none", handlelength=1.6, labelspacing=0.2,
                borderpad=0.0, ncol=2, columnspacing=1.2, handletextpad=0.4)

    ax_b.set_xlabel("epoch")
    ax_b.set_ylabel("val-only mass (%)")
    ax_b.set_ylim(0.0, 0.070)


def draw_panel_b(ax, series_list):
    """Gap dynamics: fixed-probe C_S vs epoch."""
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")

    for label, style, xs, ys in series_list:
        ax.plot(xs, ys, **style, label=label, zorder=3)

    for b in range(1, 4):
        ax.axvline(b, color="#888888", lw=0.5, ls=":", zorder=1)

    ax.legend(loc="upper left", fontsize=5.5, framealpha=0.9,
              edgecolor="none", handlelength=1.8, labelspacing=0.3,
              borderpad=0.3)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_panel_c(ax, epochs, series):
    """Seen vs novel validation CE."""
    ax.set_xlim(0.7, 5.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation CE (nats)")

    specs = [
        ("no-gram, novel", GREY, "--", "o", True,  "no-gram, novel"),
        ("no-gram, seen",  GREY, "-",  "o", False, "no-gram, seen"),
        ("ngram, novel",   "#d5622b", "--", "s", True,  "ngram, novel"),
        ("ngram, seen",    BLUE, "-",  "s", False, "ngram, seen"),
    ]

    for key, color, ls, marker, filled, label in specs:
        mfc = color if filled else "white"
        ax.plot(epochs, series[key], color=color, ls=ls, lw=1.3,
                marker=marker, ms=3.5, mfc=mfc, mec=color, mew=0.8,
                label=label, zorder=3)

    ax.legend(loc="upper left", fontsize=5.5, framealpha=0.9,
              edgecolor="none", handlelength=2.0, labelspacing=0.3,
              borderpad=0.3)
    ax.set_ylim(2, 12)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    S.apply_style()

    # Override font sizes for 396 pt target width (~5.5 in)
    # At 396 pt display width, 6.5 pt base font is legible
    plt.rcParams.update({
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
    })

    # Load all data
    epochs_a, train_seen, val_only = load_panel_a()
    series_b = load_panel_b()
    epochs_c, series_c = load_panel_c()

    # Figure layout: panel (a) is two stacked rows sharing the epoch axis;
    # panels (b) and (c) span both rows. Total width 396 pt = 5.5 in so that
    # \includegraphics[width=\linewidth] needs no rescaling.
    fig_w = 5.5   # 396 pt
    fig_h = 2.20
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.05, 0.85],
                          wspace=0.62, hspace=0.42, height_ratios=[1.05, 0.95])
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[1, 0], sharex=ax_a1)
    axes = [ax_a1, fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]

    for ax, letter in zip([ax_a1] + axes[1:], ["(a)", "(b)", "(c)"]):
        ax.set_title(letter, fontsize=7, fontweight="bold", loc="left", pad=3)

    draw_panel_a(ax_a1, ax_a2, epochs_a, train_seen, val_only)
    draw_panel_b(axes[1], series_b)
    draw_panel_c(axes[2], epochs_c, series_c)

    fig.tight_layout(pad=0.3)

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "fig_mechanism_three_panel.pdf"
    png_path = OUT / "fig_mechanism_three_panel.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"PDF: {pdf_path.relative_to(ROOT)}")
    print(f"PNG: {png_path.relative_to(ROOT)}")

    # Measure actual output size
    try:
        from pypdf import PdfReader
        r = PdfReader(str(pdf_path))
        mb = r.pages[0].mediabox
        w_pt = float(mb.width)
        h_pt = float(mb.height)
    except ImportError:
        # Fallback: use subprocess to get page size via gs or pdfinfo
        import subprocess
        result = subprocess.run(
            ["/usr/local/bin/gs", "-dNOPAUSE", "-dBATCH", "-dSAFER",
             "-sDEVICE=bbox", str(pdf_path)],
            capture_output=True, text=True
        )
        # Parse bbox output or use known figsize
        w_pt = fig_w * 72  # convert inches to points
        h_pt = fig_h * 72
        # Try pdfinfo as alternative
        try:
            result2 = subprocess.run(
                ["pdfinfo", str(pdf_path)], capture_output=True, text=True
            )
            for line in result2.stdout.split('\n'):
                if 'Page size' in line:
                    parts = line.split()
                    w_pt = float(parts[2])
                    h_pt = float(parts[4])
                    break
        except Exception:
            pass

    print(f"Output size: {w_pt:.1f} x {h_pt:.1f} pt")
    print(f"At 396 pt display width, base font = 6.5 pt (set in rcParams)")
    print(f"Panel width: {w_pt/3:.1f} pt each")

    # Write manifest
    write_manifest(pdf_path, png_path, w_pt, h_pt)


def write_manifest(pdf_path, png_path, w_pt, h_pt):
    """Write manifest JSON next to appendix_f_mask_plotted_values.json."""
    manifest = {
        "scope": "Section 4.3 three-panel mechanism figure for ICLR main text",
        "figure": {
            "pdf": str(pdf_path.relative_to(ROOT)),
            "png": str(png_path.relative_to(ROOT)),
            "sha256_pdf": _sha256(pdf_path),
            "size_pt": {"width": w_pt, "height": h_pt},
            "target_display_width_pt": 396,
            "base_font_pt": 6.5,
        },
        "script": "docs/plot_scripts/plot_paper_fig_mechanism_three_panel.py",
        "panel_a_continuation_mass": {
            "train_seen_source": {
                "path": "docs/appendices/context_subspace_mask_20260921/appendix_f_mask_plotted_values.json",
                "sha256": _sha256(APPENDIX / "appendix_f_mask_plotted_values.json"),
                "key": "seen_mass_panel.values",
            },
            "val_only_source": {
                "path": "docs/appendices/context_subspace_mask_20260921/context_kl/kl_restricted_split_e1_l4.json",
                "sha256": _sha256(APPENDIX / "context_kl" / "kl_restricted_split_e1_l4.json"),
                "keys": "z_valn_ref_mean (epoch 1), z_valn_new_mean (epochs 2-4)",
            },
            "source_runs": [
                "s1v5_128_marm_both_ctxmask_e1_l4_fixed",
                "s1v5_128_marm_nogram_ctxmask_e1_l4_fixed",
            ],
            "metric": "mean full-vocabulary probability mass on train-seen continuations (left axis) and validation-only continuations (right axis) at fixed validation positions",
            "epochs": [1, 2, 3, 4],
            "plotted_values": {
                "train_seen": load_panel_a()[1],
                "val_only": load_panel_a()[2],
            },
        },
        "panel_b_gap_dynamics": {
            "source_runs": {
                "ctrl": "s1v5_128_marm_both_ctxmask_ctrl_l4_fixed",
                "e1": "s1v5_128_marm_both_ctxmask_e1_l4_fixed",
                "e2": "s1v5_128_marm_both_ctxmask_e2_l4_fixed",
            },
            "source_files": [
                {
                    "path": f"docs/appendices/context_subspace_mask_20260921/runs_scaling/s1v5_128_marm_both_ctxmask_{r}_l4_fixed/context_gap.jsonl",
                    "sha256": _sha256(APPENDIX / "runs_scaling" / f"s1v5_128_marm_both_ctxmask_{r}_l4_fixed" / "context_gap.jsonl"),
                }
                for r in ["ctrl", "e1", "e2"]
            ],
            "metric": "fixed-probe gap contribution C_S = sum_i q_i [l_val,i - l_tr,i] (nats)",
            "series": ["normal training (C_A ctrl)", "unmasked peers (C_B ctrl)",
                       "mask at epoch 2 (C_A e1)", "mask at epoch 3 (C_A e2)"],
        },
        "panel_c_seen_vs_novel_ce": {
            "source": {
                "extracted_csv": "docs/appendices/fig09_crowd_ce_extracted_20260926/fig09_crowd_ce_seen_novel_extracted.csv",
                "original_pdf": "paper/figures/fig09_crowd_ce_seen_novel.pdf",
                "original_pdf_sha256": "d62d7544440a7fc07c2652f87353892d9870bf77bca32c5860e5d8f794cb17ba",
                "extraction_method": "pypdf content-stream parsing, calibrated against axis ticks",
            },
            "metric": "position-weighted validation cross-entropy (nats) for train-seen bigram contexts split by continuation type",
            "checkpoints": "steps 337, 674, 1011, 1348, 1685 (epochs 1-5, seed 42)",
            "position_counts": {
                "seen_continuation": 405058,
                "novel_continuation": 158790,
                "excluded_unseen_context": 25400,
            },
        },
    }

    manifest_path = APPENDIX / "fig_mechanism_three_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
