#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§130/§131 figure: who controls the forking gap — R/N (level) and table LR (scale).

Four panels, all numbers parsed from harvested TSV artifacts (never hand-copied):

  A  E3 window gap vs R/N at tls = 1, one series per model size. 50m and 104m
     land on the same curve (R/N = 8: +0.0365 / +0.0356; R/N ~ 16: +0.0880 /
     +0.0889), i.e. at fixed table LR the gap is set by R/N (equivalently R/D
     here, because D/N is held at ~20) and not by the backbone size.
  B  E3 window gap vs table LR on three ladders. Halving tls multiplies the gap
     by 0.41-0.56 (power exponent 0.87-1.28).
  C  E3 val benefit vs R/N. dval is *not* R/N-invariant (50m peaks at R/N 8-16,
     104m peaks lower and saturates at R/N ~ 8). It is flat in tls only for
     R/N <= 16; at R/N = 32 the default tls1 collapses to 4-16 mK while tls0.5 /
     0.25 hold 31-43 mK, i.e. lowering the table LR *returns* the benefit that
     forking ate (log §131).
  D  dval by epoch at R/N = 32 (solid, tls 1 / 0.5 / 0.25) against R/N = 16
     tls1 (grey dashed). tls1 starts highest (98.7 mK at E1) and ends lowest
     (13.9 mK at E3); tls0.25 starts lowest (58.0) and ends highest (42.2) —
     the ordering inverts, so the low-tls arm is not merely delayed.

Divergent / non-stationary arms are excluded (log §129(g)). Every reading is the
mean of the epoch's last 60 logged rows (`*_win` columns of v26dn_harvest.py).

Inputs
------
    python3 docs/plot_scripts/v26dn_harvest.py <runs_fixed> --tsv --window 60 \
        > harvest_<machine>.tsv
    python3 docs/plot_scripts/v26dn_tls_knob_20261021.py \
        --harvest /tmp/h360w_now.tsv /tmp/hophw_now.tsv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figs" / "v26_tls_knob_20261021"
OUT.mkdir(parents=True, exist_ok=True)

C_50M = "#1f77b4"
C_104M = "#2e8b57"
C_NOGRAM = "#8f969d"
C_TLS = {1.0: "#d62728", 0.5: "#e08214", 0.25: "#1a7f4b", 0.125: "#5b6bbf"}

# run_id -> model size, table rows, table LR, seed. `_fixed` is already stripped.
PAT = re.compile(
    r"^v26dn_(?P<model>\d+m)_(?P<kind>exact|nogram)_r(?P<rows>\d+)k_d(?P<data>[\dp]+)b"
    r"_tls(?P<tls>[\d.]+)(?:_s(?P<seed>\d+))?(_e(?P<epochs>\d+))?_fd$"
)
WIN_COLS = {
    "e1": ("e1_win_val", "e1_win_train", "e1_win_gap"),
    "e2": ("e2_win_val", "e2_win_train", "e2_win_gap"),
    "e3": ("e3_win_val", "e3_win_train", "e3_win_gap"),
}
POOL_SPLIT = {"35m": 35000.0, "50m": 50000.0, "104m": 104000.0, "159m": 159000.0}


def read_harvest(paths: list[str]) -> dict[str, dict]:
    """Merge harvest TSVs by run_id, preferring finished runs with more steps."""
    rows: dict[str, dict] = {}
    for path in paths:
        with open(path) as handle:
            for rec in csv.DictReader(handle, delimiter="\t"):
                rid = rec["run_id"]
                prev = rows.get(rid)
                key = (rec["status"] == "done", int(rec["last_step"] or 0))
                if prev is not None:
                    prev_key = (prev["status"] == "done", int(prev["last_step"] or 0))
                    if key <= prev_key:
                        continue
                rows[rid] = rec
    return rows


def parse(rid: str) -> dict | None:
    m = PAT.match(rid)
    if not m:
        return None
    d = m.groupdict()
    model = d["model"]
    return {
        "rid": rid,
        "model": model,
        "kind": d["kind"],
        "rows": float(d["rows"]) * 1000.0,
        "r_over_n": float(d["rows"]) * 1000.0 / POOL_SPLIT[model],
        "tls": float(d["tls"]),
        "seed": int(d["seed"] or 42),
        "epochs": int(d["epochs"] or 3),
        "status": None,
        "final_val": None,
    }


def num(rec: dict, col: str) -> float | None:
    value = rec.get(col, "")
    return None if value in ("", None) else float(value)


def collect(rows: dict[str, dict]) -> dict[str, list[dict]]:
    """Return finished (status=done) arms keyed by model size, healthy only."""
    out: dict[str, list[dict]] = {"50m": [], "104m": []}
    for rid, rec in rows.items():
        arm = parse(rid)
        if arm is None or arm["model"] not in out:
            continue
        arm["status"] = rec["status"]
        if rec["status"] != "done" or rec["last_step"] == "":
            continue
        final_val = num(rec, "final_val")
        if final_val is None:
            continue
        # Divergence guard (log §129(g)): a healthy arm ends well under 3.0 on
        # this corpus; anything above is a non-stationary run whose gap is not
        # interpretable.
        if final_val > 3.0:
            continue
        arm["final_val"] = final_val
        arm["final_train"] = num(rec, "final_train")
        for ep, (vc, tc, gc) in WIN_COLS.items():
            arm[f"{ep}_val"] = num(rec, vc)
            arm[f"{ep}_train"] = num(rec, tc)
            arm[f"{ep}_gap"] = num(rec, gc)
        out[arm["model"]].append(arm)
    return out


def by_tls(arms: list[dict], tls: float, seed: int | None = None) -> list[dict]:
    picked = [a for a in arms if a["tls"] == tls and (seed is None or a["seed"] == seed)]
    return sorted(picked, key=lambda a: a["r_over_n"])


def baseline(arms: list[dict], seed: int = 42) -> dict | None:
    for a in arms:
        if a["kind"] == "nogram" and a["seed"] == seed:
            return a
    return None


def main(argv: list[str]) -> int:
    if "--harvest" not in argv:
        print(__doc__)
        return 2
    idx = argv.index("--harvest")
    paths = [a for a in argv[idx + 1 :] if not a.startswith("--")]
    rows = read_harvest(paths)
    arms = collect(rows)
    for model, lst in arms.items():
        print(f"{model}: {len(lst)} finished healthy arms")

    base = {m: baseline(lst) for m, lst in arms.items()}
    for m, b in base.items():
        print(f"  nogram {m} s42: E3 win gap {b['e3_gap']:+.4f}  final val {b['final_val']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), dpi=160)
    axes = list(axes.flat)
    for ax in axes:
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)

    # ---- Panel A: gap vs R/N at tls = 1 (invariance across model size) ----
    ax = axes[0]
    for model, color, marker in (("50m", C_50M, "o"), ("104m", C_104M, "s")):
        pts = [a for a in by_tls(arms[model], 1.0) if a.get("e3_gap") is not None]
        pts = [a for a in pts if a["kind"] == "exact"]
        xs = [a["r_over_n"] for a in pts]
        ys = [a["e3_gap"] for a in pts]
        ax.plot(xs, ys, marker=marker, color=color, lw=2.0, ms=6.5, mfc="white", mew=1.8,
                label=f"{model} exact tls1")
        for a in pts:
            if a["r_over_n"] in (8.0,) or abs(a["r_over_n"] - 15.4) < 0.6 or a["r_over_n"] == 16.0:
                # 50m and 104m land on top of each other here: stagger in x and y.
                xoff, yoff = (-16, 10) if model == "50m" else (16, -16)
                ax.annotate(f"{a['e3_gap']:+.4f}", (a["r_over_n"], a["e3_gap"]),
                            textcoords="offset points", xytext=(xoff, yoff), ha="center",
                            fontsize=8, color=color)
    ax.axhline(base["50m"]["e3_gap"], color=C_NOGRAM, ls="--", lw=1.2,
               label=f"nogram 50m/104m ({base['50m']['e3_gap']:+.4f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_xlabel("R / N   （N = 参数量(M) × 10³，本梯子内 R/D 同步）")
    ax.set_ylabel("E3 own gap（末 600 步窗口，val − train）")
    ax.set_title("A  固定 tls：gap 只由 R/N 定\n50m 与 104m 在 R/N=8 / ≈16 处逐点重合", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # ---- Panel B: gap vs table LR on the three ladders ----
    ax = axes[1]
    ladders = [
        ("50m", 8.0, C_50M, "o", "50m R/N=8"),
        ("50m", 16.0, "#7fb3d5", "o", "50m R/N=16"),
        ("50m", 32.0, "#2c5f8a", "o", "50m R/N=32"),
        ("104m", 8.0, C_104M, "s", "104m R/N=8"),
    ]
    ratios = []
    seed_ratios = []
    for model, rn, color, marker, label in ladders:
        picked = [a for a in arms[model] if a["kind"] == "exact" and abs(a["r_over_n"] - rn) < 0.06
                  and a.get("e3_gap") is not None]
        picked.sort(key=lambda a: -a["tls"])
        if len(picked) < 2:
            continue
        xs = [a["tls"] for a in picked]
        ys = [a["e3_gap"] for a in picked]
        ls = "-" if model == "50m" else "--"
        ax.plot(xs, ys, marker=marker, color=color, lw=2.0, ms=6.5, mfc="white", mew=1.8,
                ls=ls, label=label)
        for hi, lo in zip(picked, picked[1:]):
            if lo["e3_gap"] <= 1e-9:
                continue
            bucket = ratios if lo["tls"] != hi["tls"] else seed_ratios
            bucket.append((label, hi["tls"], lo["tls"], lo["e3_gap"] / hi["e3_gap"]))
    for model, b in base.items():
        ax.axhline(b["e3_gap"], color=C_NOGRAM, ls=":", lw=1.1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1.0, 0.5, 0.25, 0.125])
    ax.set_xticklabels(["1×", "0.5×", "0.25×", "0.125×"])
    ax.set_xlabel("table LR 倍数（tls）")
    ax.set_ylabel("E3 own gap（末 600 步窗口）")
    ax.set_title("B  固定 R/N：gap ≈ 正比于 tls\n每减半 ×0.41–0.56（虚线=104m）", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.text(0.97, 0.06, "nogram 带（虚线）", transform=ax.transAxes, ha="right", fontsize=7.5,
            color="#60666d")

    # ---- Panel C: E3 val benefit vs R/N ----
    ax = axes[2]
    b43_104 = baseline(arms["104m"], 43)
    for model, color, marker in (("50m", C_50M, "o"), ("104m", C_104M, "s")):
        b42 = baseline(arms[model], 42)
        for tls, alpha, ls in ((1.0, 1.0, "-"), (0.5, 0.75, "--"), (0.25, 0.55, ":")):
            pts = [a for a in by_tls(arms[model], tls, seed=42)
                   if a["kind"] == "exact" and a.get("final_val") is not None]
            if not pts:
                continue
            xs = [a["r_over_n"] for a in pts]
            ys = [a["final_val"] - b42["final_val"] for a in pts]
            ax.plot(xs, ys, marker=marker, color=color, lw=2.0, ms=6, alpha=alpha, ls=ls,
                    mfc="white", mew=1.6, label=f"{model} tls{tls:g}")
            if model == "104m" and tls == 1.0 and b43_104 is not None:
                for a in pts:
                    if a["seed"] == 42:
                        lo = a["final_val"] - b43_104["final_val"]
                        ax.plot([a["r_over_n"]] * 2, sorted([a["final_val"] - b42["final_val"], lo]),
                                color=color, lw=1.2, alpha=0.6, zorder=1)
                        ax.plot([a["r_over_n"]], [lo], marker="_", color=color, ms=10, alpha=0.6)
    ax.axhline(0.0, color="#8f969d", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_xlabel("R / N")
    ax.set_ylabel("E3 Δval（同 seed nogram 终值 − 本臂终值）")
    ax.set_title("C  Δval：R/N ≤ 16 对 tls 不敏感；R/N=32 降 tls 把被吃掉的收益还回来\n104m 竖线 = s42/s43 nogram 配对区间", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    # ---- Panel D: Δval by epoch — at R/N = 32 the ordering inverts ----
    ax = axes[3]
    e3_dvals = []
    for rn, tls_list, ls, lw, alpha in ((32.0, (1.0, 0.5, 0.25), "-", 2.2, 1.0),
                                        (16.0, (1.0,), "--", 1.6, 0.55)):
        for tls in tls_list:
            for a in by_tls(arms["50m"], tls, seed=42):
                if abs(a["r_over_n"] - rn) > 1e-9:
                    continue
                ys = []
                for ep in ("e1", "e2", "e3"):
                    av, bv = a.get(f"{ep}_val"), base["50m"].get(f"{ep}_val")
                    ys.append(np.nan if av is None or bv is None else bv - av)
                if np.isnan(ys).all():
                    continue
                ax.plot([1, 2, 3], ys, marker="o", ls=ls, lw=lw, ms=6.5, color=C_TLS[tls],
                        mfc="white", mew=1.7, alpha=alpha,
                        label=f"R/N={rn:g} tls{tls:g}")
                if rn == 32.0:
                    ax.annotate(f"{ys[2]*1000:+.1f}", (3, ys[2]), textcoords="offset points",
                                xytext=(8, -2), fontsize=8, color=C_TLS[tls])
    ax.axhline(0.0, color="#8f969d", lw=1.0)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["E1", "E2", "E3"])
    ax.set_xlim(0.85, 3.5)
    ax.set_xlabel("epoch（每 epoch = 1B unique 数据的一遍）")
    ax.set_ylabel("Δval（同 seed nogram − 本臂，末 600 步窗口）")
    ax.set_title("D  Δval 随 epoch：R/N=32 高 tls 起最高、末最低（次序反转）\n实线 = R/N 32 三条 tls，灰虚线 = R/N 16 tls1 对照",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    fig.suptitle("tls 旋钮：gap 由 R/N 定档（A）且正比于 table LR（B）；"
                 "Δval 在 R/N ≤ 16 上不动（C），在 R/N=32 上被 tls1 吃掉、由低 tls 归还（C/D）",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("svg", "png"):
        path = OUT / f"fig_tls_knob.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("saved", path)

    print("\nper-halving gap ratios (label, tls_hi, tls_lo, ratio):")
    for label, t1, t2, r in ratios:
        print(f"  {label:<14} {t1:g}->{t2:g}: {r:.3f}")
    print("same-tls repeats (seed / duplicate-arm control):")
    for label, t1, t2, r in seed_ratios:
        print(f"  {label:<14} {t1:g}->{t2:g}: {r:.3f}")

    print("\nE3 Δval vs tls, 50m exact arms (paired same-seed nogram, mK):")
    print(f"  {'R/N':>5} {'tls':>6} {'seed':>4} {'E3 dval':>9} {'E3 gap':>9}")
    for a in sorted(arms["50m"], key=lambda x: (x["r_over_n"], x["tls"], x["seed"])):
        if a["kind"] != "exact":
            continue
        b = baseline(arms["50m"], a["seed"])
        if b is None or a.get("e3_val") is None or b.get("e3_val") is None:
            continue
        print(f"  {a['r_over_n']:>5.1f} {a['tls']:>6g} {a['seed']:>4} "
              f"{(b['e3_val'] - a['e3_val']) * 1000:>+9.1f} {a['e3_gap']:>+9.4f}")

    print("\nΔval by epoch, R/N = 16 / 32 (paired nogram, mK):")
    for a in sorted(arms["50m"], key=lambda x: (x["r_over_n"], x["tls"], x["seed"])):
        if a["kind"] != "exact" or a["r_over_n"] not in (16.0, 32.0):
            continue
        b = baseline(arms["50m"], a["seed"])
        if b is None:
            continue
        cells = []
        for ep in ("e1", "e2", "e3"):
            av, bv = a.get(f"{ep}_val"), b.get(f"{ep}_val")
            cells.append("    ---" if av is None or bv is None else f"{(bv - av) * 1000:>+7.1f}")
        print(f"  R/N={a['r_over_n']:>4.1f} tls={a['tls']:<5g} s{a['seed']} E1/E2/E3: " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
