#!/usr/bin/env python3
"""Why the 17M rung of the v26dn iso-ratio ladder cannot be rescued -- 2026-10-22.

Question (user 2026-10-22 16:28): can the 17M rung be fixed?  We fixed several
infra bugs before (misR tables, /tmp, launcher overwrite, val leak), so is the
17M failure an infra artifact?

Data: train_log.jsonl + table_norm.jsonl of five 360-2 arms, pulled verbatim into
`docs/figs/dn20_17m_diagnostic_20261022/series.csv` (arm, step, train, val, rms2,
rms5; table_norm sampled every 50 steps, matched to the nearest logged step).

  A  val loss vs step: the four 17M arms (one R/N=16 cell passed once in the 17M
     grid, the three R/N=8 arms did not) plus the healthy 50M r400k reference.
     The 17M arms descend normally to ~3.2-3.3, then climb back to the init level
     (~7.4) -- train and val together, so the gap stays ~-0.08 and the failure is
     invisible in the gap.
  B  table RMS: the 17M table state stays *calmer* (2gram 0.044 -> 0.062, monotone)
     than the healthy 50M arm (0.036 -> 0.077, and still rising at the end).  The
     table is not what blows up; the backbone is.
  C  dose ladder (R/N=8): mean table-row displacement over E=3 at tls=1, from the
     assignment npz counts (sum(counts)/rows x streamed tokens / 1B pool x table
     lr).  Dose rises monotonically with N (1.29 -> 2.20) while survival rises too
     (0/2 -> 2/2), so the failing rung is the LEAST-dosed one: no tls-side setting
     could ever have fixed 17M, which is why the recorded tls sweep
     (0.125/0.25/0.375/0.5/0.625/0.75/1) found no window.

Usage: python3 docs/plot_scripts/dn20_17m_diagnostic_20261022.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figs", "dn20_17m_diagnostic_20261022")
CSV_PATH = os.path.join(FIG_DIR, "series.csv")
RED = "#b3261e"
GREY = "#8a8a8a"

# label -> (line colour, line style, linewidth, drawn in panel A / B)
STYLE = {
    "17m_r136k_tls1": ("#b3261e", "-", 2.0),
    "17m_r272k_tls1": ("#e08a2e", "-", 1.8),
    "17m_r136k_tls0.125": ("#8a6f9e", "-", 1.5),
    "17m_nogram": ("#2f6f4f", "-", 2.0),
    "50m_r400k_tls1": ("#4a6fa5", "-", 2.0),
}
SHOW_RMS = ("17m_r136k_tls1", "17m_r272k_tls1", "17m_r136k_tls0.125", "50m_r400k_tls1")

# (N in M, dose at tls=1, survived, launched) for R/N = 8; doses from the
# assignment counts (sum/counts per row x streamed tokens / 1B-pool x table lr).
LADDER = [
    (17, 1.29, 0, 2),
    (35, 1.51, 2, 3),
    (50, 1.66, 2, 2),
    (104, 1.84, 2, 2),
    (159, 1.97, None, None),   # excluded: val leak / per-epoch plunge
    (200, 2.20, None, None),   # in flight
]


def load() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with open(CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["arm"], []).append(row)
    return out


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16.4, 4.9))

    # ---- A: val loss trajectories -----------------------------------------
    for arm, (colour, ls, lw) in STYLE.items():
        rows = data[arm]
        xs = [int(r["step"]) for r in rows]
        ys = [float(r["val_loss"]) for r in rows]
        label = {
            "17m_r136k_tls1": "17M r136k tls1 (R/N=8, 崩)",
            "17m_r272k_tls1": "17M r272k tls1 (R/N=16, 轻崩)",
            "17m_r136k_tls0.125": "17M r136k tls0.125 (剂量最低)",
            "17m_nogram": "17M nogram（同模型无表）",
            "50m_r400k_tls1": "50M r400k tls1（健康参照）",
        }[arm]
        ax_a.plot(xs, ys, color=colour, ls=ls, lw=lw, label=label)
    # train loss of the worst arm: same rise -> optimisation runaway, not forking
    worst = data["17m_r136k_tls1"]
    ax_a.plot([int(r["step"]) for r in worst], [float(r["train_loss"]) for r in worst],
              color=RED, ls=":", lw=1.4, label="17M r136k tls1 train（同升）")
    ax_a.axhline(7.9, color=GREY, lw=0.9, ls="--")
    ax_a.text(12, 7.95, "随机初始化水平 ~7.9", fontsize=8, color=GREY)
    ax_a.axvspan(6600, 6915, color=RED, alpha=0.07)
    ax_a.annotate("最后 ~300 步\n退回初始化", xy=(6750, 7.35), xytext=(1500, 6.6), fontsize=8.7,
                  color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax_a.set_xscale("log")
    ax_a.set_xlabel("step（17M 一轮 2305 步；50M 一轮 6780 步）")
    ax_a.set_ylabel("val_loss")
    ax_a.set_title("A · 17M 先正常下降、再整体反冲\n（train 与 val 同升 → 优化崩塌，不是 forking）", fontsize=10.4)
    ax_a.legend(fontsize=8.2, frameon=False, loc="lower left")
    ax_a.grid(alpha=0.25)

    # ---- B: table RMS ------------------------------------------------------
    for arm in SHOW_RMS:
        rows = [r for r in data[arm] if r["rms2"]]
        if not rows:
            continue
        colour = STYLE[arm][0]
        xs = [int(r["step"]) for r in rows]
        ax_b.plot(xs, [float(r["rms2"]) for r in rows], color=colour, lw=1.9,
                  label={"17m_r136k_tls1": "17M r136k tls1（崩）",
                         "17m_r272k_tls1": "17M r272k tls1（崩）",
                         "17m_r136k_tls0.125": "17M r136k tls0.125",
                         "50m_r400k_tls1": "50M r400k tls1（健康）"}[arm])
        ax_b.plot(xs, [float(r["rms5"]) for r in rows], color=colour, lw=1.1, ls="--", alpha=0.75)
    ax_b.axvspan(6600, 6915, color=RED, alpha=0.07)
    ax_b.text(200, 0.0805, "实线 2-gram／虚线 5-gram 行 RMS", fontsize=8.2, color="#444")
    ax_b.annotate("崩的这臂表状态反而更平\n(0.044→0.062，单调无爆炸)", xy=(6600, 0.060), xytext=(105, 0.0205),
                  fontsize=8.7, color="#1f4e79", arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.0))
    ax_b.set_xscale("log")
    ax_b.set_xlabel("step")
    ax_b.set_ylabel("table row RMS")
    ax_b.set_title("B · 崩的不是表：17M 的表 RMS 全程平缓、比 50M 健康臂还低", fontsize=10.4)
    ax_b.legend(fontsize=8.2, frameon=False, loc="lower right")
    ax_b.grid(alpha=0.25)

    # ---- C: dose ladder vs survival ---------------------------------------
    ns = [n for n, _d, _s, _t in LADDER]
    doses = [d for _n, d, _s, _t in LADDER]
    ax_c.plot(ns, doses, "o-", color="#1f4e79", lw=2.2, ms=7, label="每行位移剂量（tls=1，E3）")
    for n, d, _s, _t in LADDER:
        ax_c.annotate(f"{d:.2f}", (n, d), textcoords="offset points", xytext=(0, 8),
                      ha="center", fontsize=8.6, color="#1f4e79")
    ax_c.set_xscale("log")
    ax_c.set_xticks([17, 35, 50, 104, 159, 200])
    ax_c.set_xticklabels(["17M", "35M", "50M", "104M", "159M", "200M"])
    ax_c.minorticks_off()
    ax_c.set_ylim(1.0, 2.6)
    ax_c.set_xlabel("模型 N（D=20N，R=8N，iso-ratio 梯子）")
    ax_c.set_ylabel("剂量（lr 单位，越大＝表被推得越远）", color="#1f4e79")
    ax_c.axvspan(15, 26, color="#b3261e", alpha=0.10)
    ax_c.annotate("存活 0/2", xy=(17, 1.29), xytext=(16.5, 1.06), fontsize=8.6, color=RED)
    ax2 = ax_c.twinx()
    surv_x = [n for n, _d, s, _t in LADDER if s is not None]
    surv_y = [s / t for _n, _d, s, t in LADDER if s is not None]
    ax2.plot(surv_x, surv_y, "s--", color="#2f6f4f", lw=1.7, ms=6, label="存活率（臂在窗内／已发）")
    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel("存活率", color="#2f6f4f")
    ax2.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax_c.set_title("C · 剂量随 N 单调升、存活率也单调升\n→ 崩的那档剂量最低，tls 侧本来就不可能救", fontsize=10.4)
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, fontsize=8.4, frameon=False, loc="lower right")
    ax_c.grid(alpha=0.25)

    for ax in (ax_a, ax_b):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("v26dn 17M 档为什么修不掉（seed 42，同机配对 nogram；数据＝360-2 runs_fixed 的 "
                 "train_log/table_norm 原样）", fontsize=12.3, y=1.02)
    fig.tight_layout()
    for ext in ("svg", "png"):
        path = os.path.abspath(os.path.join(FIG_DIR, f"fig_17m_diagnostic.{ext}"))
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print("wrote", path)


if __name__ == "__main__":
    main()
