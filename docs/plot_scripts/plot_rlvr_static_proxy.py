#!/usr/bin/env python3
"""Plot RLVR-like static verifier-selected continued-pretraining runs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "plot_scripts"))
from v5_style import apply_style, add_epoch_lines
SRC = ROOT / "docs" / "appendices" / "rlvr_static_proxy_20260921" / "runs"
OUT = ROOT / "docs" / "figs" / "rlvr_static_proxy_20260921"
RUNS = [("RLVR-MATH", "math"), ("RLVR-MIXED", "mixed")]
def load(tag, arm):
    rid = f"sft_rlvr_{tag}_e5_{arm}_s42"
    rows = [json.loads(x) for x in (SRC / rid / "train_log.jsonl").read_text().splitlines() if x.strip()]
    rows.sort(key=lambda r:r["step"])
    return {k: np.array([r[k] for r in rows], float) for k in ("step","train_loss","val_loss","gap")}
def plot(arm):
    apply_style()
    fig, ax = plt.subplots(2,2,figsize=(4.69,2.975),sharex="col",gridspec_kw={"height_ratios":[1,.82],"hspace":.18,"wspace":.27})
    fig.subplots_adjust(left=.085,right=.995,top=.90,bottom=.15)
    for j,(title,tag) in enumerate(RUNS):
        d=load(tag,arm); x=d["step"]
        ax[0,j].plot(x,d["train_loss"],lw=1.25,color="#2d6f9f")
        ax[0,j].plot(x,d["val_loss"],lw=1.15,ls="--",color="#c4493d",alpha=.82)
        ax[1,j].plot(x,d["gap"],lw=1.25,color="#4f8f62")
        ax[1,j].axhline(0,color="#80868b",lw=.6,ls=":")
        for a in ax[:,j]:
            add_epoch_lines(a,333,1665,color="#b8bec4"); a.set_xlim(0,1665); a.grid(True,axis="y")
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        ax[0,j].set_title(title,fontsize=10.5,pad=4,fontweight="medium")
        all_loss=np.r_[d["train_loss"],d["val_loss"]]; lo,hi=float(all_loss.min()),float(all_loss.max()); pad=max(.1*(hi-lo),.05)
        ax[0,j].set_ylim(lo-pad,hi+pad)
        gl,gh=float(d["gap"].min()),float(d["gap"].max()); gp=max(.1*(gh-gl),.05); ax[1,j].set_ylim(gl-gp,gh+gp)
        ax[0,j].set_xticklabels([]); ax[1,j].set_xticks([0,333,666,999,1332,1665]); ax[1,j].set_xticklabels(["0","1","2","3","4","5"],fontsize=8)
    ax[0,0].set_ylabel("loss (nats)"); ax[1,0].set_ylabel("gap (nats)"); ax[1,0].set_xlabel("replay epoch"); ax[1,1].set_xlabel("replay epoch")
    # caption carries line semantics: blue solid=train, red dashed=fixed val, green=gap.
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f"fig_rlvr_{arm}.pdf",bbox_inches="tight"); fig.savefig(OUT/f"fig_rlvr_{arm}.png",dpi=300,bbox_inches="tight"); plt.close(fig)
def plot_math_comparison():
    apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(4.69, 2.975), sharex="col",
                           gridspec_kw={"height_ratios":[1,.82], "hspace":.18, "wspace":.27})
    fig.subplots_adjust(left=.085, right=.995, top=.90, bottom=.15)
    for j, arm in enumerate(("trainable", "frozen")):
        d = load("math", arm); x = d["step"]
        ax[0,j].plot(x, d["train_loss"], lw=1.25, color="#2d6f9f")
        ax[0,j].plot(x, d["val_loss"], lw=1.15, ls="--", color="#c4493d", alpha=.82)
        ax[1,j].plot(x, d["gap"], lw=1.25, color="#4f8f62")
        ax[1,j].axhline(0, color="#80868b", lw=.6, ls=":")
        for a in ax[:,j]:
            add_epoch_lines(a, 333, 1665, color="#b8bec4")
            a.set_xlim(0, 1665); a.grid(True, axis="y")
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        ax[0,j].set_title("table trainable" if arm == "trainable" else "table frozen",
                          fontsize=10.5, pad=4, fontweight="medium")
        all_loss = np.r_[d["train_loss"], d["val_loss"]]
        lo, hi = float(all_loss.min()), float(all_loss.max()); pad = max(.1*(hi-lo), .05)
        ax[0,j].set_ylim(lo-pad, hi+pad)
        gl, gh = float(d["gap"].min()), float(d["gap"].max()); gp = max(.1*(gh-gl), .05)
        ax[1,j].set_ylim(gl-gp, gh+gp)
        ax[0,j].set_xticklabels([])
        ax[1,j].set_xticks([0,333,666,999,1332,1665]); ax[1,j].set_xticklabels(["0","1","2","3","4","5"], fontsize=8)
    ax[0,0].set_ylabel("loss (nats)"); ax[1,0].set_ylabel("gap (nats)")
    ax[1,0].set_xlabel("replay epoch"); ax[1,1].set_xlabel("replay epoch")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_rlvr_math.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_rlvr_math.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def plot_mixed_comparison():
    apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(4.69, 2.975), sharex="col",
                           gridspec_kw={"height_ratios":[1,.82], "hspace":.18, "wspace":.27})
    fig.subplots_adjust(left=.085, right=.995, top=.90, bottom=.15)
    for j, arm in enumerate(("trainable", "frozen")):
        d = load("mixed", arm); x = d["step"]
        ax[0,j].plot(x, d["train_loss"], lw=1.25, color="#2d6f9f")
        ax[0,j].plot(x, d["val_loss"], lw=1.15, ls="--", color="#c4493d", alpha=.82)
        ax[1,j].plot(x, d["gap"], lw=1.25, color="#4f8f62")
        ax[1,j].axhline(0, color="#80868b", lw=.6, ls=":")
        for a in ax[:,j]:
            add_epoch_lines(a, 333, 1665, color="#b8bec4")
            a.set_xlim(0, 1665); a.grid(True, axis="y")
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        ax[0,j].set_title("table trainable" if arm == "trainable" else "table frozen",
                          fontsize=10.5, pad=4, fontweight="medium")
        all_loss = np.r_[d["train_loss"], d["val_loss"]]
        lo, hi = float(all_loss.min()), float(all_loss.max()); pad = max(.1*(hi-lo), .05)
        ax[0,j].set_ylim(lo-pad, hi+pad)
        gl, gh = float(d["gap"].min()), float(d["gap"].max()); gp = max(.1*(gh-gl), .05)
        ax[1,j].set_ylim(gl-gp, gh+gp)
        ax[0,j].set_xticklabels([])
        ax[1,j].set_xticks([0,333,666,999,1332,1665]); ax[1,j].set_xticklabels(["0","1","2","3","4","5"], fontsize=8)
    ax[0,0].set_ylabel("loss (nats)"); ax[1,0].set_ylabel("gap (nats)")
    ax[1,0].set_xlabel("replay epoch"); ax[1,1].set_xlabel("replay epoch")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_rlvr_mixed.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_rlvr_mixed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
for arm in ("trainable","frozen"): plot(arm)
plot_math_comparison()
plot_mixed_comparison()
