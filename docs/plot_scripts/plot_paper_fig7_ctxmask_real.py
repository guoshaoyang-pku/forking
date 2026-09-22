#!/usr/bin/env python3
"""Paper Sec.4.3 figures from the ctxmask A/B runs (experiment-log §62).

Data (ophis-gpu, /data1/guoshaoyang/ngram-gap-lab/data/runs_scaling/,
fetched to a local dir by --runs_root):
  s1v5_128_marm_both_ctxmask_{ctrl,e1,e2}_fixed/context_gap.jsonl
    per-10-step rows with Cpos_A/Cpos_B = val_share * (val_mean - tr_online),
    the paper's gap contribution C_S(t) for context set S (2% of distinct
    trigram types, seed 42; A = loss-masked from the boundary, B = peers).

Fig 7 (fig07_context_mask.pdf): single panel over epochs 0..4, online
caliber (Cpos).  Fig 7 fixed-probe caliber (fig07_context_mask_fixed.pdf):
same layout but plotting C_A/C_B, the per-context val-share-weighted gap
contribution against the FIXED train probe (train.py evaluate_context_gap);
masked A freezes at its pre-mask plateau (no replay-boundary spike), control
keeps stepping up each epoch.  ctrl supplies
the continuous B (unmasked) and A-normal-training (dashed) curves; e1 and e2
are plotted as separate full masked trials, including their later-epoch
aftermaths.  2026-09-22 (15:39) user redesign: each masked trial runs to the
SAME late horizon (4 epochs, run_ids *_l4, launcher
code/cluster/run_ctxmask_ab_long.sh) so the epoch-1-masked trial's aftermath
stays visible through epochs 2-4 instead of being cut at the epoch-2 boundary;
the plot prefers *_l4 dirs and falls back to the short §62 dirs.  Style:
epoch ticks are INTEGERS only, the whole figure is exported at 0.5
linewidth-equivalent (4.4 in for the project's 8.8 in text width), PNG dpi
300.  The two masked trials are drawn separately: the epoch-1-masked trial
continues through later epochs, rather than being replaced by the epoch-2
trial at epoch 2.

Fig 7* (fig07star_context_kl.pdf): per-arm KL drift of masked-context
next-token distributions (eval_context_kl.py output JSON), both-table vs
no-gram (and bigram/trigram) for the masked epoch vs the preceding normal
epoch and the normal control.

Usage:
  python docs/plot_scripts/plot_paper_fig7_ctxmask_real.py \
      --runs_root /path/to/fetched/runs_scaling \
      --kl_json docs/appendices/context_subspace_mask_20260921/context_kl.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
import v5_style as S

OUT = ROOT / "docs" / "figs" / "paper"
BLUE, ORANGE = "#2d6f9f", "#d5622b"
EPOCH_STEPS = 337


def load_ctx(runs_root: Path, run: str):
    rows = []
    for line in (runs_root / f"{run}_fixed" / "context_gap.jsonl").open():
        r = json.loads(line)
        rows.append(r)
    rows.sort(key=lambda r: r["step"])
    return rows


def load_ctx_pref(runs_root: Path, base: str):
    """Prefer the 4-epoch *_l4 rerun, fall back to the short §62 run."""
    for suffix in ("_l4", ""):
        p = runs_root / f"{base}{suffix}_fixed" / "context_gap.jsonl"
        if p.exists():
            return load_ctx(runs_root, base + suffix)
    raise SystemExit(f"no context_gap.jsonl for {base}[_l4] under {runs_root}")


def _perstep_series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None or v != v:
            continue
        xs.append(r["step"] / EPOCH_STEPS)
        ys.append(v)
    return np.array(xs), np.array(ys)


def fig7_perstep(runs_root: Path, infix: str = "ctxmask"):
    """Per-step Fig 7 in both calibers, extended through 4 epochs (l4 runs).

    Top panel (CANONICAL, fixed caliber): C_S = sum over contexts in the
    fixed-probe ∩ fixed-val join of q_i [l_val,i - l_tr,i].  The probe is a
    fixed set of 4 train batches, so there is no measurement-position
    artifact; the ctrl staircase (jump at each boundary when the table rows
    are re-fit, decay within the epoch from backbone drift) is real
    dynamics.  Masked A stays flat at its pre-mask level.

    Bottom panel (online caliber): Cpos_S = val_share * (val_mean -
    tr_online_mean).  Per-step shape carries the shard-position artifact
    (early-shard positions have no backbone recency -> high tr at epoch
    starts); read epoch end vs epoch end or use the epoch-mean figure.
    """
    ctrl = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_ctrl")
    e1 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e1")
    e2 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e2")
    max_epoch = max(int(np.ceil(r["step"] / EPOCH_STEPS)) for r in ctrl)

    specs = [  # (rows, fixed_key, online_key, style, label)
        (ctrl, "C_B", "Cpos_B", dict(color=ORANGE, lw=1.4), "unmasked peers"),
        (ctrl, "C_A", "Cpos_A", dict(color=BLUE, lw=1.0, ls="--"), "normal training"),
        (e1, "C_A", "Cpos_A", dict(color=BLUE, lw=1.4), "masked from epoch 2"),
        (e2, "C_A", "Cpos_A", dict(color="#7b4aa2", lw=1.4), "masked from epoch 3"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(4.4, 4.8), sharex=True,
                             gridspec_kw=dict(hspace=0.32))
    for ax, key_idx, title in ((axes[0], 1, "fixed-set caliber (canonical)"),
                               (axes[1], 2, "online caliber (position artifact; "
                                            "compare epoch end to epoch end)")):
        for rows, fkey, okey, st, lab in specs:
            xs, ys = _perstep_series(rows, fkey if key_idx == 1 else okey)
            ax.plot(xs, ys, **st, label=lab if key_idx == 1 else None)
        for b in range(1, max_epoch):
            ax.axvline(b, color="#888888", lw=0.7, ls=":")
        ax.set_xlim(0, max_epoch)
        ax.set_ylabel("gap contribution (nats)")
        ax.set_title(title, fontsize=8, loc="left", color="#444444")
    axes[0].legend(loc="upper left", framealpha=0.9, edgecolor="none",
                   handlelength=1.8, fontsize=7.5)
    axes[1].set_xticks(range(max_epoch + 1))
    axes[1].set_xlabel("epoch")
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if infix == "ctxmask" else f"_{infix}"
    fig.savefig(OUT / f"fig07_context_mask_perstep{tag}.pdf")
    fig.savefig(OUT / f"fig07_context_mask_perstep{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask_perstep{tag}.pdf").relative_to(ROOT))

    # standalone canonical fixed-caliber panel (for paper embedding)
    fig2, ax = plt.subplots(figsize=(4.4, 2.5))
    for rows, fkey, _okey, st, lab in specs:
        xs, ys = _perstep_series(rows, fkey)
        ax.plot(xs, ys, **st, label=lab)
    for b in range(1, max_epoch):
        ax.axvline(b, color="#888888", lw=0.7, ls=":")
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(range(max_epoch + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none",
              handlelength=1.8, fontsize=7.5)
    fig2.tight_layout(pad=0.4)
    fig2.savefig(OUT / f"fig07_context_mask_fixed_perstep{tag}.pdf")
    fig2.savefig(OUT / f"fig07_context_mask_fixed_perstep{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask_fixed_perstep{tag}.pdf").relative_to(ROOT))


def epoch_means(rows, key_tr, key_val, share_key, n_min=20000):
    """Per-epoch mean of share * (val_mean - tr_online) for one context set.

    Full online windows only (tr_online_n > n_min): the extra boundary evals
    create 2-5-step windows that sit on early-shard positions and would bias
    the epoch mean.  Returns {epoch: cpos_mean}.
    """
    out = {}
    for r in rows:
        if r.get("tr_online_n_A", 0) < n_min:
            continue
        tr, va = r.get(key_tr), r.get(key_val)
        if tr is None or va is None or tr != tr or va != va:
            continue
        ep = int(np.ceil(r["step"] / EPOCH_STEPS))
        out.setdefault(ep, []).append(r[share_key] * (va - tr))
    return {ep: float(np.mean(v)) for ep, v in out.items()}


def fig7_epochmean(runs_root: Path, infix: str = "ctxmask"):
    """Position-artifact-free Fig 7: epoch-mean online caliber.

    With fixed replay the per-eval online tr_A slides across shard positions
    within an epoch (late-shard positions enjoy backbone recency), producing
    a sawtooth that is a measurement-position effect, not time evolution.
    Epoch means remove it; masked-A then stays flat across epochs while
    normal-training A keeps growing (§62 2026-09-22 correction).
    """
    ctrl = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_ctrl")
    e1 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e1")
    e2 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e2")

    ctrl_A = epoch_means(ctrl, "tr_online_A", "val_mean_A", "val_share_A")
    ctrl_B = epoch_means(ctrl, "tr_online_B", "val_mean_B", "val_share_B")
    e1_A = epoch_means(e1, "tr_online_A", "val_mean_A", "val_share_A")
    e2_A = epoch_means(e2, "tr_online_A", "val_mean_A", "val_share_A")
    max_epoch = max(ctrl_A)

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.axvline(1.5, color="#888888", lw=0.7, ls=":")
    ax.axvline(2.5, color="#888888", lw=0.7, ls=":")

    def line(d, **kw):
        xs = sorted(d)
        ax.plot(xs, [d[e] for e in xs], marker="o", ms=4, **kw)

    line(ctrl_B, color=ORANGE, lw=1.4, label="unmasked peers")
    line(ctrl_A, color=BLUE, lw=1.0, ls="--", label="normal training")
    line({e: v for e, v in e1_A.items() if e >= 2},
         color=BLUE, lw=1.4, label="masked from epoch 2")
    line({e: v for e, v in e2_A.items() if e >= 3},
         color="#7b4aa2", lw=1.4, label="masked from epoch 3")
    ax.set_xlim(0.7, max_epoch + 0.3)
    ax.set_xticks(range(1, max_epoch + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none",
              handlelength=1.8, fontsize=7.5)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if infix == "ctxmask" else f"_{infix}"
    fig.savefig(OUT / f"fig07_context_mask_epochmean{tag}.pdf")
    fig.savefig(OUT / f"fig07_context_mask_epochmean{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask_epochmean{tag}.pdf").relative_to(ROOT))
    for name, d in (("ctrl_A", ctrl_A), ("ctrl_B", ctrl_B),
                    ("e1_A", e1_A), ("e2_A", e2_A)):
        print(f"  {name}: " + "  ".join(f"ep{e}={v:.4f}" for e, v in sorted(d.items())))


def fig7(runs_root: Path, infix: str = "ctxmask"):
    ctrl = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_ctrl")
    e1 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e1")
    e2 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e2")
    max_epoch = max(int(np.ceil(r["step"] / EPOCH_STEPS))
                    for rows in (ctrl, e1, e2) for r in rows)

    def xy(rows, key, step_min=0):
        pts = [(r["step"] / EPOCH_STEPS, r[key]) for r in rows
               if r["step"] > step_min and r.get(key) is not None
               and r[key] == r[key]]
        return zip(*pts) if pts else ([], [])

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    # 2026-09-22 user: show epoch 1 as well (both masks start at epoch
    # boundaries 1 and 2; epoch 1 is the unmasked baseline segment)
    ax.axvline(1, color="#888888", lw=0.7, ls=":")
    ax.axvline(2, color="#888888", lw=0.7, ls=":")
    x, y = xy(ctrl, "Cpos_B")
    ax.plot(x, y, color=ORANGE, lw=1.4, label="unmasked peers")
    x, y = xy(ctrl, "Cpos_A")
    ax.plot(x, y, color=BLUE, lw=1.0, ls="--", label="normal training")
    x, y = xy(e1, "Cpos_A")
    ax.plot(x, y, color=BLUE, lw=1.4, label="masked from epoch 2")
    x, y = xy(e2, "Cpos_A")
    ax.plot(x, y, color="#7b4aa2", lw=1.4, label="masked from epoch 2")
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(range(max_epoch + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", handlelength=1.8)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if infix == "ctxmask" else f"_{infix}"
    fig.savefig(OUT / f"fig07_context_mask{tag}.pdf")
    fig.savefig(OUT / f"fig07_context_mask{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask{tag}.pdf").relative_to(ROOT))


def fig7_fixed(runs_root: Path, infix: str = "ctxmask"):
    """Fixed-probe caliber: C_A/C_B = sum_i q_i [l_val,i - l_tr,i], l_tr from
    the fixed train probe.  Same three-run overlay as fig7."""
    ctrl = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_ctrl")
    e1 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e1")
    e2 = load_ctx_pref(runs_root, f"s1v5_128_marm_both_{infix}_e2")
    max_epoch = max(int(np.ceil(r["step"] / EPOCH_STEPS))
                    for rows in (ctrl, e1, e2) for r in rows)

    def xy(rows, key):
        pts = [(r["step"] / EPOCH_STEPS, r[key]) for r in rows
               if r.get(key) is not None and r[key] == r[key]]
        return zip(*pts) if pts else ([], [])

    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.axvline(1, color="#888888", lw=0.7, ls=":")
    ax.axvline(2, color="#888888", lw=0.7, ls=":")
    x, y = xy(ctrl, "C_B")
    ax.plot(x, y, color=ORANGE, lw=1.4, label="unmasked peers")
    x, y = xy(ctrl, "C_A")
    ax.plot(x, y, color=BLUE, lw=1.0, ls="--", label="normal training")
    x, y = xy(e1, "C_A")
    ax.plot(x, y, color=BLUE, lw=1.4, label="masked from epoch 2")
    x, y = xy(e2, "C_A")
    ax.plot(x, y, color="#7b4aa2", lw=1.4, label="masked from epoch 2")
    ax.set_xlim(0, max_epoch)
    ax.set_xticks(range(max_epoch + 1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap contribution, fixed probe (nats)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", handlelength=1.8)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if infix == "ctxmask" else f"_{infix}"
    fig.savefig(OUT / f"fig07_context_mask_fixed{tag}.pdf")
    fig.savefig(OUT / f"fig07_context_mask_fixed{tag}.png", dpi=300)
    print((OUT / f"fig07_context_mask_fixed{tag}.pdf").relative_to(ROOT))


def fig7star(kl_json: Path, arms=None, tag=""):
    """Fig 7*: drift of the masked contexts' next-token distributions.

    Three KL probes on set A (mean full-vocab KL over fixed-val positions):
      prev_normal  = epoch-1 boundary -> epoch-2 boundary (a normal epoch);
      masked       = epoch-2 boundary -> end of the epoch-3-masked run;
      normal_ctrl  = epoch-2 boundary -> epoch-3 end of the unmasked control.
    Honest reading (2026-09-22): masking lowers A-drift for every arm
    (2% of gradient updates removed), and the both-table arm drifts least in
    absolute terms -- but the masked-epoch KL is NOT "almost zero" in any arm
    (residual cross-context backbone drift; it shifts train and val together
    and therefore does not enter the gap, cf. fig7).
    """
    d = json.loads(kl_json.read_text())["pairs"]
    arms = arms or ["nogram", "bigram", "trigram", "both"]
    labels = {"prev_normal": "epoch 2\n(normal)",
              "masked": "epoch 3\n(masked)",
              "normal_ctrl": "epoch 3\n(control)"}
    colors = {"nogram": S.ARM_COLORS["nogram"], "bigram": S.BRANCH_COLORS["bigram"],
              "trigram": S.BRANCH_COLORS["trigram"], "both": S.ARM_COLORS["input"]}
    names = {"nogram": "no-gram", "bigram": "bigram only",
             "trigram": "trigram only", "both": "both tables"}
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    n = len(arms)
    series = []
    for i, arm in enumerate(arms):
        if f"{arm}_masked" not in d:
            continue
        xs = np.arange(3) + (i - (n - 1) / 2) * 0.08
        ys = [d[f"{arm}_{k}"]["sets"]["A"]["kl_ref_new_mean"] for k in labels]
        ax.plot(xs, ys, marker="o", ms=4.5, lw=1.4, color=colors[arm],
                label=names[arm])
        series.append((arm, xs, ys))
    for k in range(3):
        ranked = sorted(series, key=lambda t: -t[2][k])
        for rank, (arm, xs, ys) in enumerate(ranked):
            dx = (list(colors).index(arm) - (n - 1) / 2) * 3
            dy = 6 if rank == 0 else -13
            ax.annotate(f"{ys[k]:.2f}", (xs[k], ys[k]),
                        textcoords="offset points", xytext=(dx, dy),
                        ha="center", fontsize=6.5, color=colors[arm])
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(labels.values()), fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    y_hi = max(y for _, _, ys in series for y in ys)
    ax.set_ylim(top=y_hi * 1.14)
    ax.set_ylabel("KL$(p_{ref} \\| p_{end})$ on A (nats)")
    ax.legend(framealpha=0.9, edgecolor="none", fontsize=7.5,
              loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / f"fig07star_context_kl{tag}.pdf")
    fig.savefig(OUT / f"fig07star_context_kl{tag}.png", dpi=300)
    print((OUT / f"fig07star_context_kl{tag}.pdf").relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path)
    ap.add_argument("--kl_json", type=Path)
    ap.add_argument("--infix", default="ctxmask",
                    help="run-id infix for fig7 (ctxmask | rowmask)")
    ap.add_argument("--arms", default="nogram,bigram,trigram,both",
                    help="comma list of arms for fig7* (subset OK)")
    args = ap.parse_args()
    S.apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8})
    if args.runs_root:
        fig7(args.runs_root, args.infix)
        fig7_epochmean(args.runs_root, args.infix)
        fig7_perstep(args.runs_root, args.infix)
        fig7_fixed(args.runs_root, args.infix)
    if args.kl_json:
        arms = [a for a in args.arms.split(",") if a]
        tag = "" if args.infix == "ctxmask" else f"_{args.infix}"
        fig7star(args.kl_json, arms, tag)


if __name__ == "__main__":
    main()
