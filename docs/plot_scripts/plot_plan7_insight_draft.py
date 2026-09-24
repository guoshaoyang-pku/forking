"""Draft insight figure for Sec. 4.2-4.3 (context subspaces + crowding-out). DRAFT for redesign.

Evidence: docs/appendices/subspace_probe_20260925/ep5ckpt_v5_{nogram,trigram,both}_s42.json
(experiment-log sec66; runs ep5ckpt_v5_*_s42 on 360-2, seed 42, epochs 1-5).
Output: docs/figs/plan7_insight_draft/insight_draft_v1.{pdf,png}
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = os.path.join(ROOT, "docs/appendices/subspace_probe_20260925")
OUT = os.path.join(ROOT, "docs/figs/plan7_insight_draft")
CKS = ["epoch1_model.pt", "epoch2_model.pt", "epoch3_model.pt", "epoch4_model.pt", "final_model.pt"]
C_NG, C_BOTH, C_TRI = "#7a7a7a", "#1f6fb4", "#8e5bb5"

R = {r: json.load(open(os.path.join(EV, f"ep5ckpt_v5_{r}_s42.json"))) for r in ("nogram", "trigram", "both")}
n_ctx = len(json.load(open(os.path.join(EV, "probe_tokens.json")))["dogcat_candidates"])


def ctx_mean(run, key):
    out = []
    for ck in CKS:
        rows = R[run]["ckpts"][ck]["context_only"][2:]
        out.append(sum(r[key] for r in rows) / len(rows))
    return out


def transfer(run, variant):
    return [R[run]["ckpts"][ck]["influence"]["variants"][variant]["transfer_q_dog"] for ck in CKS]


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, p, q, color, lw, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def schematic(ax, with_table):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.02, 0.70, 0.26, 0.12, '"that is a"', "#fde9e7", fs=8)
    box(ax, 0.02, 0.22, 0.26, 0.12, '"there is a"', "#e8f1fb", fs=8)
    box(ax, 0.40, 0.40, 0.22, 0.24, "shared\nbackbone", "#f2f2f2", fs=8)
    if with_table:
        box(ax, 0.30, 0.77, 0.12, 0.09, "row A", "#fff4cc", fs=7)
        box(ax, 0.30, 0.15, 0.12, 0.09, "row B", "#fff4cc", fs=7)
    for y, lab in ((0.78, "dog"), (0.62, "cat"), (0.46, "rabbit")):
        box(ax, 0.78, y - 0.05, 0.18, 0.09, lab, "#ffffff", fs=7.5)
    ax.text(0.87, 0.90, 'p( · | "there is a")', ha="center", fontsize=7.5)
    arrow(ax, (0.15, 0.93), (0.15, 0.83), "#c0392b", 1.4)
    ax.text(0.15, 0.95, 'replayed: "that is a → dog"', ha="center", fontsize=7, color="#c0392b")
    if with_table:
        arrow(ax, (0.28, 0.76), (0.31, 0.81), "#c0392b", 2.2)
        arrow(ax, (0.28, 0.74), (0.43, 0.60), "#c0392b", 1.4)
        arrow(ax, (0.62, 0.52), (0.78, 0.78), C_BOTH, 1.5)
        arrow(ax, (0.62, 0.50), (0.78, 0.62), C_BOTH, 0.6, ls="--")
        ax.text(0.36, 0.05, "own row per context: most of the update\nstays inside that context's subspace",
                ha="center", fontsize=7)
        ax.text(0.70, 0.30, "T ≈ 0.5\ncat +0.06", ha="center", fontsize=7.5, color=C_BOTH)
    else:
        arrow(ax, (0.28, 0.74), (0.43, 0.60), "#c0392b", 2.2)
        arrow(ax, (0.62, 0.52), (0.78, 0.78), C_NG, 2.4)
        arrow(ax, (0.62, 0.50), (0.78, 0.62), C_NG, 1.6)
        arrow(ax, (0.62, 0.48), (0.78, 0.46), C_NG, 1.2)
        ax.text(0.36, 0.05, "one shared pathway: evidence pools across\ncontexts and across similar continuations",
                ha="center", fontsize=7)
        ax.text(0.70, 0.30, "T ≈ 0.7–1.0\ncat +0.12", ha="center", fontsize=7.5, color=C_NG)


fig = plt.figure(figsize=(13.2, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 1.0, 1.0], wspace=0.28)
ax = fig.add_subplot(gs[0, 0]); schematic(ax, False); ax.set_title("(a) no n-gram: shared coordinates", fontsize=9)
ax = fig.add_subplot(gs[0, 1]); schematic(ax, True); ax.set_title("(b) with n-gram: private rows", fontsize=9)

ep = [1, 2, 3, 4, 5]
ax = fig.add_subplot(gs[0, 2])
ax.plot(ep, ctx_mean("both", "logp_dog"), "-o", color=C_BOTH, ms=3, label="n-gram, seen (dog)")
ax.plot(ep, ctx_mean("both", "logp_cat"), "--o", color=C_BOTH, ms=3, mfc="white", label="n-gram, unseen (cat)")
ax.plot(ep, ctx_mean("nogram", "logp_dog"), "-s", color=C_NG, ms=3, label="no-gram, seen (dog)")
ax.plot(ep, ctx_mean("nogram", "logp_cat"), "--s", color=C_NG, ms=3, mfc="white", label="no-gram, unseen (cat)")
ax.set_xlabel("replay epoch", fontsize=8)
ax.set_ylabel("mean log p (context only)", fontsize=8)
ax.set_title(f"(c) inside a context: {n_ctx} rare contexts\ntrain saw only ' dog'", fontsize=9)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.3, frameon=False, loc="lower left")

ax = fig.add_subplot(gs[0, 3])
for run, c, lab in (("nogram", C_NG, "no-gram"), ("trigram", C_TRI, "trigram"), ("both", C_BOTH, "both tables")):
    s, g = transfer(run, "sign"), transfer(run, "grad")
    ax.fill_between(ep, s, g, color=c, alpha=0.15, lw=0)
    ax.plot(ep, s, "-o", color=c, ms=3, label=lab)
ax.set_ylim(0.3, 1.05)
ax.set_xlabel("replay epoch", fontsize=8)
ax.set_ylabel("transfer T", fontsize=8)
ax.set_title("(d) across contexts: update on\n'that is a→dog' moves 'there is a→dog'", fontsize=9)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, frameon=False, loc="lower left")
fig.text(0.5, -0.04, "DRAFT (seed 42, one minimal pair for T; band = gradient vs sign-update estimate). "
         "Evidence: experiment-log §66, docs/appendices/subspace_probe_20260925.", ha="center", fontsize=7, color="#666666")
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"insight_draft_v1.{ext}"), bbox_inches="tight", dpi=200)
print("saved", OUT)
