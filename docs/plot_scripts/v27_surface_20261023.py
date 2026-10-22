#!/usr/bin/env python3
"""v27 scaling surface: dGap / dTrain / dVal over (D/N, R/N) at E3 end (and E1/E2).

Source: .tmp/harvest/v27_a100/paired.csv (code/tools/v27_analyze.py; E-end 60-row windows,
same-seed nogram pairing, mK, positive dVal = OE wins, dVal = dTrain - dGap).
Excludes: diverged, e1_dirty, dup_pool (v27a pools > 2.385B; drawn hollow).
Fits (clean points with dGap > 3 mK):
  (1) per (model,tls) at R/N=8: dGap(D/N) power vs exponential
  (2) 50M tls1: log dGap = c + a log(R/N) + b log(D/N)    (2-var power law)
  (3) tpp test: if dGap were a function of tpp = D/R only, a = -b.
Outputs docs/figs/v27_surface_20261023/{fig_surface.png,svg,fits.json}
"""
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab"
SRC = f"{ROOT}/.tmp/harvest/v27_a100/paired.csv"
OUT = f"{ROOT}/docs/figs/v27_surface_20261023"
os.makedirs(OUT, exist_ok=True)

C = {"ink": "#191919", "muted": "#6b6a64", "grid": "#e8e6dc", "clay": "#d97757",
     "sky": "#6a9bcc", "olive": "#788c5d", "fig": "#c46686", "slate": "#5e5d59", "red": "#bf4d43"}


def load():
    rows = list(csv.DictReader(open(SRC)))
    out = []
    for r in rows:
        if r["tag"] not in ("", "_p27b"):
            continue
        r2 = dict(model=r["model"], tls=r["tls"], dn=float(r["dn"]), rn=round(float(r["rn"]), 2),
                  seed=int(r["seed"]), div=r["diverged"] == "True", dirty=r["e1_dirty"] == "True",
                  dup=r["dup_pool"] == "True", done=r["done"] == "True")
        for e in (1, 2, 3):
            for k in ("dtrain", "dgap", "dval"):
                v = r.get(f"{k}{e}")
                r2[f"{k}{e}"] = float(v) if v not in (None, "") else None
        out.append(r2)
    return out


def agg(rows, key, e=3, clean=True):
    g = defaultdict(list)
    for r in rows:
        if r[f"dval{e}"] is None:
            continue
        if clean and (r["div"] or r["dirty"] or r["dup"]):
            continue
        g[key(r)].append(r)
    res = {}
    for k, lst in g.items():
        res[k] = {f: float(np.mean([x[f"{f}{e}"] for x in lst])) for f in ("dtrain", "dgap", "dval")}
        res[k]["n"] = len(lst)
        res[k]["dval_sd"] = float(np.std([x[f"dval{e}"] for x in lst])) if len(lst) > 1 else None
    return res


def fit_pow(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = y > 3
    b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    pred = np.exp(a) * x[m] ** b
    return dict(A=float(np.exp(a)), b=float(b), rmse=float(np.sqrt(np.mean((pred - y[m]) ** 2))), n=int(m.sum()))


def fit_exp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = y > 3
    k, a = np.polyfit(x[m], np.log(y[m]), 1)
    pred = np.exp(a) * np.exp(k * x[m])
    return dict(A=float(np.exp(a)), k=float(k), rmse=float(np.sqrt(np.mean((pred - y[m]) ** 2))), n=int(m.sum()))


def main():
    rows = load()
    fits = {}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax in axs.flat:
        ax.grid(True, color=C["grid"], lw=0.6)
        for s in ax.spines.values():
            s.set_color(C["muted"])

    # ---- (a) dGap vs D/N at R/N=8, E1/E2/E3
    ax = axs[0, 0]
    for (model, tls, col) in (("35m", "0.5", C["sky"]), ("50m", "1", C["clay"])):
        for e, ls, a_ in ((1, ":", 0.5), (2, "--", 0.7), (3, "-", 1.0)):
            S = agg(rows, lambda r: (r["model"], r["tls"], r["rn"], r["dn"]), e=e)
            pts = sorted((k[3], v) for k, v in S.items() if k[:3] == (model, tls, 8.0))
            if not pts:
                continue
            x = [p[0] for p in pts]; y = [p[1]["dgap"] for p in pts]
            ax.plot(x, y, ls, color=col, alpha=a_, marker="o", ms=4,
                    label=f"{model.upper()} tls{tls} E{e}")
            if e == 3:
                fp, fe = fit_pow(x, y), fit_exp(x, y)
                fits[f"dgap_vs_dn_{model}_tls{tls}_rn8_E3"] = {"power": fp, "exp": fe, "x": x, "y": y}
        # dup (hollow)
        S = agg([r for r in rows if r["dup"] and not r["div"]], lambda r: (r["model"], r["tls"], r["rn"], r["dn"]), clean=False)
        for k, v in S.items():
            if k[:3] == (model, tls, 8.0):
                ax.plot(k[3], v["dgap"], "o", mfc="none", mec=col, ms=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D/N (unique tokens per param)"); ax.set_ylabel("dGap (mK)  ↑ = more forking")
    ax.set_title("(a) forking added by OE vs data, R/N=8", loc="left", fontsize=11)
    ax.legend(fontsize=7, ncol=2, frameon=False)

    # ---- (b) dGap vs R/N at fixed D/N (E3)
    ax = axs[0, 1]
    S = agg(rows, lambda r: (r["model"], r["tls"], r["dn"], r["rn"]))
    combos = [("50m", "1", 2, C["clay"], "-"), ("50m", "1", 5, C["clay"], "--"), ("50m", "1", 20, C["clay"], ":"),
              ("35m", "0.5", 5, C["sky"], "--"), ("35m", "0.5", 20, C["sky"], ":"),
              ("35m", "1", 2, C["olive"], "-"), ("35m", "1", 5, C["olive"], "--")]
    for model, tls, dn, col, ls in combos:
        pts = sorted((k[3], v) for k, v in S.items() if k[:3] == (model, tls, float(dn)))
        if len(pts) < 2:
            continue
        x = [p[0] for p in pts]; y = [max(p[1]["dgap"], 0.5) for p in pts]
        ax.plot(x, y, ls, color=col, marker="o", ms=4, label=f"{model.upper()} tls{tls} D/N={dn}")
        fits[f"dgap_vs_rn_{model}_tls{tls}_dn{dn}_E3"] = fit_pow(x, [p[1]["dgap"] for p in pts])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("R/N (rows per order per 1k params)"); ax.set_ylabel("dGap (mK)")
    ax.set_title("(b) forking added vs table size", loc="left", fontsize=11)
    ax.legend(fontsize=7, frameon=False)

    # ---- (c) dTrain vs R/N
    ax = axs[0, 2]
    for model, tls, dn, col, ls in combos:
        pts = sorted((k[3], v) for k, v in S.items() if k[:3] == (model, tls, float(dn)))
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1]["dtrain"] for p in pts], ls, color=col, marker="o", ms=4,
                label=f"{model.upper()} tls{tls} D/N={dn}")
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("R/N"); ax.set_ylabel("dTrain (mK)  ↑ = OE fits train better")
    ax.set_title("(c) train-side gain vs table size", loc="left", fontsize=11)
    ax.legend(fontsize=7, frameon=False)

    # ---- (d) dVal vs R/N (acceptance)
    ax = axs[1, 0]
    for model, tls, dn, col, ls in combos:
        pts = sorted((k[3], v) for k, v in S.items() if k[:3] == (model, tls, float(dn)))
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1]["dval"] for p in pts], ls, color=col, marker="o", ms=4,
                label=f"{model.upper()} tls{tls} D/N={dn}")
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(-80, 60)
    ax.set_xlabel("R/N"); ax.set_ylabel("dVal E3 (mK)  ↑ = OE wins (acceptance)")
    ax.set_title("(d) net E3 gain vs table size", loc="left", fontsize=11)
    ax.legend(fontsize=7, frameon=False)

    # ---- (e) dVal vs D/N at R/N=8
    ax = axs[1, 1]
    S2 = agg(rows, lambda r: (r["model"], r["tls"], r["rn"], r["dn"]))
    for (model, tls, col) in (("35m", "0.5", C["sky"]), ("50m", "1", C["clay"])):
        pts = sorted((k[3], v) for k, v in S2.items() if k[:3] == (model, tls, 8.0))
        ax.plot([p[0] for p in pts], [p[1]["dval"] for p in pts], "-", color=col, marker="o", ms=5,
                label=f"{model.upper()} tls{tls} dVal")
        ax.plot([p[0] for p in pts], [p[1]["dtrain"] for p in pts], ":", color=col, marker=".", ms=4,
                label=f"{model.upper()} dTrain")
        ax.plot([p[0] for p in pts], [p[1]["dgap"] for p in pts], "--", color=col, marker=".", ms=4,
                label=f"{model.upper()} dGap")
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("D/N"); ax.set_ylabel("mK")
    ax.set_title("(e) E3 decomposition vs data, R/N=8", loc="left", fontsize=11)
    ax.legend(fontsize=7, ncol=2, frameon=False)

    # ---- (f) tpp collapse test: dGap vs tpp
    ax = axs[1, 2]
    for (model, tls, col) in (("35m", "0.5", C["sky"]), ("35m", "1", C["olive"]), ("50m", "1", C["clay"])):
        pts = [(k[2] * 1000 / k[3], v["dgap"], k[2]) for k, v in S.items() if k[:2] == (model, tls) and v["dgap"] > 0.5]
        for t, g, dn in pts:
            ax.plot(t, g, "o", color=col, ms=3 + 1.5 * math.log2(dn), alpha=0.8)
        ax.plot([], [], "o", color=col, label=f"{model.upper()} tls{tls} (size ∝ log D/N)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tpp = D/R (unique tokens per table row)"); ax.set_ylabel("dGap (mK)")
    ax.set_title("(f) does tpp alone collapse dGap?", loc="left", fontsize=11)
    ax.legend(fontsize=7, frameon=False)

    # ---- 2-var power-law fit, 50M tls1 and 35M tls0.5
    for model, tls in (("50m", "1"), ("35m", "0.5"), ("35m", "1")):
        X, Y = [], []
        for k, v in S.items():
            if k[:2] == (model, tls) and v["dgap"] > 3:
                X.append([1.0, math.log(k[3]), math.log(k[2])]); Y.append(math.log(v["dgap"]))
        if len(X) >= 4:
            X, Y = np.array(X), np.array(Y)
            coef, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
            pred = X @ coef
            fits[f"dgap_2var_{model}_tls{tls}_E3"] = dict(
                A=float(math.exp(coef[0])), a_RN=float(coef[1]), b_DN=float(coef[2]),
                rms_log=float(np.sqrt(np.mean((pred - Y) ** 2))), n=len(Y),
                tpp_only_would_need="a_RN = -b_DN")
        Xt, Yt = [], []
        for k, v in S.items():
            if k[:2] == (model, tls) and v["dtrain"] > 3:
                Xt.append([1.0, math.log(k[3]), math.log(k[2])]); Yt.append(math.log(v["dtrain"]))
        if len(Xt) >= 4:
            Xt, Yt = np.array(Xt), np.array(Yt)
            coef, *_ = np.linalg.lstsq(Xt, Yt, rcond=None)
            fits[f"dtrain_2var_{model}_tls{tls}_E3_positive"] = dict(
                A=float(math.exp(coef[0])), a_RN=float(coef[1]), b_DN=float(coef[2]), n=len(Yt))

    fig.suptitle("v27 clean pool · E3 end · same-seed nogram pairing · hollow = v27a dup pool", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{OUT}/fig_surface.png", dpi=150)
    fig.savefig(f"{OUT}/fig_surface.svg")
    json.dump(fits, open(f"{OUT}/fits.json", "w"), indent=1)
    for k, v in fits.items():
        if isinstance(v, dict) and "power" in v:
            print(k, "power b=%.3f rmse=%.1f | exp k=%.4f rmse=%.1f (n=%d)" % (
                v["power"]["b"], v["power"]["rmse"], v["exp"]["k"], v["exp"]["rmse"], v["power"]["n"]))
        else:
            print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
