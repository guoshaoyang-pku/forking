#!/usr/bin/env python3
"""Divergence root-cause plot from stability.jsonl (train.py --stability_interval).

For each _stab run: train loss, max attention logit per layer, attention entropy
(most collapsed head), residual RMS per layer, output-logit logsumexp / max,
pre-clip grad norms, q/k weight norms. Exact arms vs their nogram control,
x-axis = step; the train-loss onset (30-row smoothed train loss > running min + 0.10)
is marked. Output: docs/figs/v27_divergence_20261023/fig_stab.{png,svg} + onset table.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab"
H = f"{ROOT}/.tmp/harvest/v27_a100"
OUT = f"{ROOT}/docs/figs/v27_divergence_20261023"
os.makedirs(OUT, exist_ok=True)


def load(rid):
    fs = sorted(glob.glob(f"{H}/*/{rid}/stability.jsonl"), key=lambda f: -os.path.getsize(f))
    if not fs:
        return None
    rows = []
    for line in open(fs[0]):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def onset(rows, key="train_loss", W=30, thr=0.10):
    v = [r[key] for r in rows]
    s = [r["step"] for r in rows]
    mn = 9e9
    for i in range(len(v) - W + 1):
        m = sum(v[i:i + W]) / W
        mn = min(mn, m)
        if m > mn + thr:
            return s[i + W - 1]
    return None


def series(rows, key):
    return np.array([r["step"] for r in rows if key in r]), np.array([r[key] for r in rows if key in r])


def main(pairs):
    panels = [("train_loss", "train loss"), ("att_max", "max attention logit (per layer)"),
              ("att_entmin", "entropy of most-collapsed head"), ("x_rms", "residual RMS at block input"),
              ("lse_mean", "output logsumexp (mean)"), ("logit_max", "max output logit"),
              ("g_backbone", "backbone grad norm (pre-clip)"), ("w_q", "‖W_q‖ per layer")]
    fig, axs = plt.subplots(len(pairs), len(panels), figsize=(4.0 * len(panels), 3.0 * len(pairs)), squeeze=False)
    report = []
    for row, (ex, ng, title) in enumerate(pairs):
        E, N = load(ex), load(ng)
        if not E:
            continue
        on = onset(E)
        report.append((title, on, E[-1]["step"]))
        nl = len([k for k in E[-1] if k.startswith("att_max")])
        for col, (key, lab) in enumerate(panels):
            ax = axs[row, col]
            for rows, ls, a in ((E, "-", 1.0), (N, ":", 0.8)):
                if not rows:
                    continue
                if key in rows[-1]:
                    x, y = series(rows, key)
                    ax.plot(x, y, ls, color="#d97757" if rows is E else "#6a9bcc", lw=1.0, alpha=a)
                else:
                    cm = plt.get_cmap("viridis")
                    for li in range(nl):
                        k = f"{key}{li}"
                        if k in rows[-1]:
                            x, y = series(rows, k)
                            ax.plot(x, y, ls, color=cm(li / max(1, nl - 1)), lw=0.9, alpha=a,
                                    label=f"L{li}" if (rows is E and row == 0) else None)
            if on:
                ax.axvline(on, color="#b54a32", lw=0.8, ls="--")
            if key in ("g_backbone",):
                ax.set_yscale("log")
            ax.grid(True, color="#e8e6dc", lw=0.5)
            if row == 0:
                ax.set_title(lab, fontsize=9, loc="left")
                if key == "att_max":
                    ax.legend(fontsize=6, frameon=False)
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
    fig.suptitle("solid = OE arm, dotted = nogram control; red dashed = train-loss onset", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/fig_stab.{ext}", dpi=110)
    for t, on, last in report:
        print(f"{t:36} onset {on} (last step {last})")


if __name__ == "__main__":
    P = [("v27_17m_exact_r136k_dn20_tls1_s43_e3_stab_fixed", "v27_17m_nogram_r136k_dn20_tls1_s43_e3_stab_fixed", "17M R/N=8 s43"),
         ("v27_17m_exact_r136k_dn20_tls1_s42_e3_stab_fixed", "v27_17m_nogram_r136k_dn20_tls1_s43_e3_stab_fixed", "17M R/N=8 s42"),
         ("v27_17m_exact_r35k_dn20_tls1_s42_e3_stab_fixed", "v27_17m_nogram_r136k_dn20_tls1_s43_e3_stab_fixed", "17M R/N=2 s42"),
         ("v27_35m_exact_r35k_dn20_tls1_s42_e3_stab_fixed", "v27_35m_nogram_r280k_dn20_tls0.5_s42_e3_stab_fixed", "35M R/N=1 s42"),
         ("v27_50m_exact_r100k_dn20_tls1_s43_e3_stab_fixed", "v27_50m_nogram_r400k_dn20_tls1_s43_e3_stab_fixed", "50M R/N=2 s43")]
    main(P)
