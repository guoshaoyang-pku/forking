#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measurement-audit figure for docs/experiment-log.md §129.

Three panels, all from harvested artifacts (never hand-copied numbers):

  A  val-loss trajectory (24 equal bins per run) for the 50m / 104m / 159m
     exact-vs-nogram pairs. Quasi-stationary runs improve monotonically; the
     159m pair dips and then rises, which is what makes 159m point readings
     regime-dependent.
  B  terminal-row artifact `window - single` own gap per done arm, grouped by
     data pool. The offset repeats inside a pool (exact and nogram alike) and
     changes sign across pools, so it is a fixed final-batch difficulty offset,
     not sampling noise.
  C  104m R ladder on the window basis: E1 gap is nogram-identical, E2 shows the
     onset, E3 is monotone in R while the E3 val benefit peaks at R/N ~ 8.

Inputs
------
    python3 docs/plot_scripts/v26dn_traj_dump.py  <runs_fixed> > traj.tsv
    python3 docs/plot_scripts/v26dn_harvest.py    <runs_fixed> --tsv > harvest.tsv
    python3 docs/plot_scripts/v26dn_measurement_audit_20261021.py \
        --traj /tmp/traj360.tsv /tmp/trajoph.tsv \
        --harvest /tmp/h360w.tsv /tmp/hophw.tsv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figs" / "v26_measurement_audit_20261021"
OUT.mkdir(parents=True, exist_ok=True)

C_EXACT = "#1f77b4"
C_NOGRAM = "#8f969d"
C_DIVERGED = "#d62728"
POOL_COLOR = {"35m": "#b8860b", "50m": "#1f77b4", "104m": "#2e8b57", "159m": "#c1440e"}

# Panel A: representative pairs (run_id prefix, label, colour key)
PAIRS = [
    ("v26dn_50m_exact_r800k_d1b_tls1_s42_e3_fd", "50m exact r800k", "50m", C_EXACT, "-"),
    ("v26dn_50m_nogram_r400k_d1b_tls1_s42_e3_fd", "50m nogram", "50m", C_NOGRAM, "--"),
    ("v26dn_104m_exact_r832k_d2p1b_tls1_s42_e3_fd", "104m exact r832k", "104m", "#2e8b57", "-"),
    ("v26dn_104m_nogram_r832k_d2p1b_tls1_s42_e3_fd", "104m nogram", "104m", "#9cc3a8", "--"),
    ("v26dn_159m_exact_r1272k_d3p2b_tls1_s42_e3_fd", "159m exact r1272k", "159m", "#c1440e", "-"),
    ("v26dn_159m_nogram_r1272k_d3p2b_tls1_s42_e3_fd", "159m nogram", "159m", "#e0a08a", "--"),
]


def read_traj(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            rd = csv.DictReader(f, delimiter="\t")
            for row in rd:
                nbins = sum(1 for k in row if k.startswith("v") and k[1:].isdigit())
                out[row["run_id"]] = [float(row[f"v{i}"]) for i in range(nbins)]
    return out


def read_harvest(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                out[row["run_id"]] = row
    return out


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def panel_a(ax, traj):
    handles = []
    for rid, label, pool, colour, ls in PAIRS:
        v = traj.get(rid)
        if not v:
            print(f"  ! missing traj for {rid}")
            continue
        x = [i / (len(v) - 1) for i in range(len(v))]
        (ln,) = ax.plot(x, v, color=colour, lw=1.9, ls=ls, marker="o", ms=2.6, label=label)
        handles.append(ln)
    ax.set_xlabel("run progress (0 = start, 1 = end)", fontsize=9)
    ax.set_ylabel("fixed val loss (24-bin mean)", fontsize=9)
    ax.set_title("A · val 轨迹：50m/104m 单调，159m 先跌后涨（瞬变）", fontsize=10.5, loc="left")
    ax.grid(color="#dfe3e8", lw=0.6)
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)
    # annotate the 159m excursion
    v159 = traj.get(PAIRS[4][0])
    if v159:
        lo = min(range(len(v159)), key=lambda i: v159[i])
        hi = max(range(lo, len(v159)), key=lambda i: v159[lo] - v159[i]) if False else None
        after = max(v159[lo:])
        hi_i = lo + v159[lo:].index(after)
        ax.annotate("", xy=(hi_i / (len(v159) - 1), after), xytext=(lo / (len(v159) - 1), v159[lo]),
                    arrowprops=dict(arrowstyle="->", color="#c1440e", lw=1.3))
        ax.text(0.52, min(v159) - 0.05, f"跌 {v159[lo]:.3f} → 涨回 {after:.3f}（+{after - v159[lo]:.3f}）",
                color="#c1440e", fontsize=8)


def panel_b(ax, harvest):
    items = []
    for rid, r in harvest.items():
        if r.get("status") != "done":
            continue
        if rid.endswith("_e5_fd"):
            continue
        g, wg = fnum(r.get("e3_gap")), fnum(r.get("e3_win_gap"))
        if g is None or wg is None:
            continue
        model = rid.split("_")[1]
        e1v, e3v = fnum(r.get("e1_val")), fnum(r.get("e3_val"))
        div = e1v is not None and e3v is not None and e3v > e1v + 0.5
        items.append((model, rid, wg - g, div))
    order = ["35m", "50m", "104m", "159m"]
    items.sort(key=lambda t: (order.index(t[0]) if t[0] in order else 99, t[2]))
    ys = range(len(items))
    for y, (model, rid, d, div) in zip(ys, items):
        ax.barh(y, d, color=C_DIVERGED if div else POOL_COLOR.get(model, "#888"),
                alpha=0.55 if div else 0.95, height=0.72)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r[0]} {r[1].split('_', 2)[2][:22]}" for r in items], fontsize=6)
    ax.set_xlabel("window − single  own gap  (E3)", fontsize=9)
    ax.set_title("B · 末行伪影按数据池分档（池内 exact/nogram 同值 → 非噪声）", fontsize=10.5, loc="left")
    ax.grid(axis="x", color="#dfe3e8", lw=0.6)
    ax.tick_params(labelsize=8)
    for model in order:
        sub = [i for i, r in enumerate(items) if r[0] == model]
        if sub:
            ax.text(0.085, (sub[0] + sub[-1]) / 2, model, color=POOL_COLOR[model],
                    fontsize=9, fontweight="bold", va="center")
    ax.text(0.0, -0.16, "红 = val 发散臂（§129d 判定，其 own gap 不可解释）", transform=ax.transAxes,
            fontsize=7.5, color=C_DIVERGED)


def panel_c(ax, harvest):
    base = harvest.get("v26dn_104m_nogram_r832k_d2p1b_tls1_s42_e3_fd")
    rows = [
        ("nogram", "v26dn_104m_nogram_r832k_d2p1b_tls1_s42_e3_fd", None),
        ("R/N=3.8\nr400k", "v26dn_104m_exact_r400k_d2p1b_tls1_s42_e3_fd", base),
        ("R/N=8\nr832k", "v26dn_104m_exact_r832k_d2p1b_tls1_s42_e3_fd", base),
        ("R/N=15.4\nr1600k", "v26dn_104m_exact_r1600k_d2p1b_tls1_s42_e3_fd", base),
    ]
    eps = [1, 2, 3]
    colours = ["#8f969d", "#1f77b4", "#2e8b57", "#c1440e"]
    width = 0.26
    for k, (lab, rid, b) in enumerate(rows):
        r = harvest.get(rid)
        if not r:
            print(f"  ! missing harvest row {rid}")
            continue
        xs = [k + (e - 2) * width for e in eps]
        ys = [fnum(r.get(f"e{e}_win_gap")) for e in eps]
        ax.bar(xs, ys, width=width * 0.92, color=colours[k], alpha=0.9, label=lab)
        for x, y in zip(xs, ys):
            if y is not None:
                ax.text(x, y + (0.004 if y >= 0 else -0.011), f"{y:+.3f}", ha="center",
                        fontsize=7, color="#333333")
    ax.axhline(-0.0340, color=C_NOGRAM, lw=1.1, ls=":")
    ax.text(3.02, -0.0340, "nogram E3 水平", fontsize=7, color="#60666d", va="center")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("own gap = val − train（末 600 步窗口均值）", fontsize=9)
    ax.set_title("C · 104m：E1 与 nogram 不可区分、E2 起阶梯、E3 随 R 单调", fontsize=10.5, loc="left")
    ax.grid(axis="y", color="#dfe3e8", lw=0.6)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-0.55, 4.0)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    # E3 val benefit annotation (wrapped so it stays inside the canvas)
    bx = harvest.get("v26dn_104m_nogram_r832k_d2p1b_tls1_s42_e3_fd")
    notes = []
    for lab, rid, b in rows[1:]:
        r = harvest.get(rid)
        if not r or not bx:
            continue
        dv = fnum(r.get("e3_val")) - fnum(bx.get("e3_val"))
        notes.append(f"{lab.split(chr(10))[0]} Δval {dv:+.4f}")
    ax.text(0.5, -0.235, "E3 Δval vs nogram\n" + "   ".join(notes[:2]) + "\n" + "   ".join(notes[2:]),
            transform=ax.transAxes, fontsize=7.5, color="#333333", ha="center", va="top")


def main():
    argv = sys.argv[1:]
    traj_paths, harv_paths = [], []
    for flag, dest in (("--traj", traj_paths), ("--harvest", harv_paths)):
        while flag in argv:
            i = argv.index(flag)
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                dest.append(argv[j])
                j += 1
            del argv[i:j]
    traj = read_traj(traj_paths)
    harv = read_harvest(harv_paths)
    print(f"traj runs: {len(traj)}   harvest runs: {len(harv)}")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6))
    panel_a(axes[0], traj)
    panel_b(axes[1], harv)
    panel_c(axes[2], harv)
    fig.suptitle("v26dn 度量审计（docs/experiment-log.md §129）：末行伪影按池分档 · 159m 瞬变 · 104m 窗口阶梯",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    for ext in ("svg", "png"):
        path = OUT / f"fig_measurement_audit.{ext}"
        fig.savefig(path, dpi=170 if ext == "png" else None)
        print("saved", path)


if __name__ == "__main__":
    main()
