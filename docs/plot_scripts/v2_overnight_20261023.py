#!/usr/bin/env python3
"""Overnight report figures (2026-10-23): N-ladder at D/N=20 R/N=8, clip rescue, distill readouts.

Sources: .tmp/harvest/v27_a100/paired.csv + raw train_log.jsonl (v27_analyze.py conventions:
E-end 60-row windows, same-seed nogram pairing, mK, + dVal = OE wins); student logs in
.tmp/harvest/student/{pro5k,a100}/<run>/train_log.jsonl (harvested separately).
Outputs docs/figs/v2_overnight_20261023/{fig_ladder,fig_clip,fig_distill}.{png,svg}
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab"
H = f"{ROOT}/.tmp/harvest/v27_a100"
SH = f"{ROOT}/.tmp/harvest/student"
OUT = f"{ROOT}/docs/figs/v2_overnight_20261023"
os.makedirs(OUT, exist_ok=True)
C = dict(ink="#141413", muted="#5e5d59", grid="#e8e6dc", orange="#d97757", blue="#6a9bcc",
         green="#788c5d", red="#b54a32", mid="#b0aea5", fig="#c46686")
NN = {"35m": 35, "50m": 50, "104m": 104, "200m": 200}


def style(ax):
    ax.grid(True, color=C["grid"], lw=0.6)
    for s in ax.spines.values():
        s.set_color(C["mid"])
    ax.tick_params(colors=C["muted"])


def rows_of(rid):
    fs = sorted(glob.glob(f"{H}/*/{rid}/train_log.jsonl"), key=lambda f: -os.path.getsize(f))
    if not fs:
        return []
    out = []
    for line in open(fs[0]):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("val_loss") is not None:
            out.append(r)
    return out


def fig_ladder():
    P = list(csv.DictReader(open(f"{H}/paired.csv")))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for e, ax in zip((1, 2, 3), axs):
        style(ax)
        for model in ("35m", "50m", "104m", "200m"):
            pts = [r for r in P if r["model"] == model and abs(float(r["dn"]) - 20) < 1e-6
                   and abs(float(r["rn"]) - 8) < 0.05 and r["tag"] in ("", "_p27b") and r.get(f"dval{e}")
                   and r["diverged"] != "True"]
            for r in pts:
                hollow = r["dup_pool"] == "True"
                ax.plot(NN[model], float(r[f"dval{e}"]), "o", ms=8, color=C["orange"],
                        mfc="none" if hollow else C["orange"])
                ax.plot(NN[model], float(r[f"dgap{e}"]), "s", ms=6, color=C["blue"],
                        mfc="none" if hollow else C["blue"])
        ax.set_xscale("log")
        ax.set_xticks([35, 50, 104, 200]); ax.set_xticklabels(["35M", "50M", "104M", "200M"])
        ax.minorticks_off()
        ax.axhline(0, color=C["ink"], lw=0.7)
        ax.set_title(f"end of pass {e}", loc="left", fontsize=11)
        ax.set_xlabel("model parameters N")
    axs[0].set_ylabel("mK")
    axs[0].plot([], [], "o", color=C["orange"], label="dVal (+ = OE wins)")
    axs[0].plot([], [], "s", color=C["blue"], label="dGap (+ = OE adds forking)")
    axs[0].plot([], [], "o", color=C["mid"], mfc="none", label="hollow = pool had the duplicate shards")
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("D/N = 20, R/N = 8, table LR 1x (35M: 0.5x); one point per seed", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/fig_ladder.{ext}", dpi=150)
    plt.close(fig)


def fig_clip():
    cases = [("35M R/N=0.26 D/N=20", "v27_35m_exact_r9k_dn20_tls0.5_s42_e3"),
             ("35M R/N=1 D/N=20", "v27_35m_exact_r35k_dn20_tls0.5_s42_e3"),
             ("35M R/N=8 D/N=28 s43", "v27_35m_exact_r280k_dn28_tls0.5_s43_e3"),
             ("50M R/N=2 D/N=20 s43", "v27_50m_exact_r100k_dn20_tls1_s43_e3"),
             ("50M R/N=8 D/N=28", "v27_50m_exact_r400k_dn28_tls1_s42_e3"),
             ("50M R/N=8 D/N=20 s43 (healthy control)", "v27_50m_exact_r400k_dn20_tls1_s43_e3")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.4), sharey=False)
    for (label, base), ax in zip(cases, axs.flat):
        style(ax)
        for suf, col, lab in (("_fixed", C["red"], "no clip"), ("_clip1.0_fixed", C["green"], "backbone clip 1.0")):
            R = rows_of(base + suf)
            if not R:
                continue
            s = np.array([r["step"] for r in R]); v = np.array([r["val_loss"] for r in R])
            k = 20
            if len(v) > k:
                v = np.convolve(v, np.ones(k) / k, mode="valid"); s = s[k - 1:]
            ax.plot(s, v, color=col, lw=1.4, label=lab)
        ax.set_ylim(2.55, 3.6)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("step")
    axs[0, 0].set_ylabel("val loss"); axs[1, 0].set_ylabel("val loss")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/fig_clip.{ext}", dpi=150)
    plt.close(fig)


def stu_rows(path):
    out = []
    for line in open(path):
        if line.startswith('{"step'):
            out.append(json.loads(line))
    return out


def fig_distill():
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax in axs:
        style(ax)
    # wave 1 (pro5k): val delta vs nogram
    base = f"{SH}/pro5k"
    if os.path.exists(f"{base}/stu_nogram_s42_e3/train_log.jsonl"):
        nv = {r["step"]: r["val_loss"] for r in stu_rows(f"{base}/stu_nogram_s42_e3/train_log.jsonl") if "val_loss" in r}
        arms = [("stu_random_1_8_tls1_s42_e3", "random init, trainable", C["green"], "-"),
                ("stu_trainable_1_8_tls1_s42_e3", "distilled 1-8, trainable", C["orange"], "-"),
                ("stu_frozen_1_8_tls1_s42_e3", "distilled 1-8, frozen", C["orange"], "--"),
                ("stu_trainable_1_8_ctr_r072_s42_e3", "distilled 1-8 centred, trainable", C["blue"], "-"),
                ("stu_trainable_9_16_ctr_r072_s42_e3", "distilled 9-16 centred", C["fig"], "-"),
                ("stu_trainable_17_24_ctr_r072_s42_e3", "distilled 17-24 centred", C["red"], "-"),
                ("stu_trainable_25_32_ctr_r072_s42_e3", "distilled 25-32 centred", C["muted"], "-")]
        for run, lab, col, ls in arms:
            p = f"{base}/{run}/train_log.jsonl"
            if not os.path.exists(p):
                continue
            av = {r["step"]: r["val_loss"] for r in stu_rows(p) if "val_loss" in r}
            ks = sorted(set(av) & set(nv))
            if ks:
                axs[0].plot([k + 1 for k in ks], [(nv[k] - av[k]) * 1000 for k in ks], ls, color=col, marker="o", ms=3, label=lab)
        axs[0].axvline(7355, color=C["mid"], lw=0.8, ls=":")
        axs[0].text(7355, axs[0].get_ylim()[1] if axs[0].lines else 0, " end of pass 1", fontsize=8, color=C["muted"], va="top")
    axs[0].axhline(0, color=C["ink"], lw=0.7)
    axs[0].set_xlabel("step"); axs[0].set_ylabel("dVal vs nogram (mK, + = OE wins)")
    axs[0].set_title("student 240M, distilled vs random tables (val, every 500 steps)", loc="left", fontsize=10)
    axs[0].legend(fontsize=7, frameon=False)
    # wave 2 (A100): train-loss delta in 100-step windows (early, before the first val)
    base2 = f"{SH}/a100"
    np2 = f"{base2}/stu2_nogram_s42_e3/train_log.jsonl"
    if os.path.exists(np2):
        nt = {r["step"]: r["train_loss"] for r in stu_rows(np2)}
        arms2 = [("stu2_random_1_8_s42_e3", "random, trainable", C["green"], "-"),
                 ("stu2_randfrozen_1_8_s42_e3", "random, frozen", C["green"], "--"),
                 ("stu2_frozen_1_8_s42_e3", "distilled, frozen (raw)", C["orange"], "--"),
                 ("stu2_frozen_1_8_ctr_s42_e3", "distilled, frozen, centred", C["blue"], "--"),
                 ("stu2_trainable_1_8_ctr_s42_e3", "distilled, trainable, centred", C["blue"], "-"),
                 ("stu2_random_1_8_rms0036_s42_e3", "random, small init", C["mid"], "-"),
                 ("stu2_frozen_9_16_ctr_s42_e3", "distilled 9-16, frozen, centred", C["fig"], "--")]
        for run, lab, col, ls in arms2:
            p = f"{base2}/{run}/train_log.jsonl"
            if not os.path.exists(p):
                continue
            at = {r["step"]: r["train_loss"] for r in stu_rows(p)}
            ks = sorted(set(at) & set(nt))
            W = 100
            xs, ys = [], []
            for a in range(0, (max(ks) + 1 if ks else 0) - W + 1, W):
                win = [k for k in range(a, a + W) if k in at and k in nt]
                if len(win) == W:
                    xs.append(a + W); ys.append(np.mean([nt[k] - at[k] for k in win]) * 1000)
            axs[1].plot(xs, ys, ls, color=col, label=lab)
    axs[1].axhline(0, color=C["ink"], lw=0.7)
    axs[1].set_xlabel("step"); axs[1].set_ylabel("dTrain vs nogram (mK, 100-step windows)")
    axs[1].set_title("wave 2 controls (A100): is the deficit the mean offset?", loc="left", fontsize=10)
    axs[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/fig_distill.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_ladder()
    fig_clip()
    fig_distill()
    print("ok", OUT)
