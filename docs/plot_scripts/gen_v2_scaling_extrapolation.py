#!/usr/bin/env python3
"""v2 第一条线：forking gap 的 scaling 关系与外推（2026-10-17）。

数据（本地 harvest，均为 data/runs_fixed 的 summary/train_log 副本）：
  output/harvest_e3/*.json                      §78 E=3/E2/E6 epoch 固定协议
  output/harvest_v224/*/{summary.json,train_log.jsonl}   §74/§75 固定 2000 步（含整条曲线）

产出：docs/figs/v2_scaling_20261017/{fig_scaling_fit,fig_extrapolation,fig_epoch_R}.{png,svg}
     docs/appendices/v2_scaling_20261017/{fit_params.json,extrapolation.csv}

拟合对象：trigram-only（v24ms tri，clean 表 R=2^20），E=3，5 档 backbone × 6 池（含 seed 43/44）。
v24ms n25 臂的 bigram 走的是旧 2-hash 表（launcher 漏传 --bigram_clean_table），只作旁证。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
H_E3 = os.path.join(ROOT, "output", "harvest_e3")
H_224 = os.path.join(ROOT, "output", "harvest_v224")
FIG = os.path.join(ROOT, "docs", "figs", "v2_scaling_20261017")
APP = os.path.join(ROOT, "docs", "appendices", "v2_scaling_20261017")
TB = 147456
plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
BG, BORDER, MUTED, ANCHOR = "#f7f5ef", "#c8c1b6", "#686d73", "#353d79"
NCOL = ["#2d6f9f", "#3c8d5a", "#b67524", "#c4493d", "#7a4b9a"]


def n_backbone(c):
    L, d = c["n_layer"], c["n_embd"]
    return 12 * L * d * d + c["vocab_size"] * d + c["sequence_len"] * d


def load_e3():
    rows = []
    for f in glob.glob(os.path.join(H_E3, "*.json")):
        d = json.load(open(f))
        m = re.match(r"(v23d|v24ms)_(?:l\d+h\d+d\d+_)?(n[\d-]+|tri|n25)_p\w+?_s(\d+)_e(\d)_fd", d["run_id"])
        fam, mode, seed, E = m.groups()
        rows.append(dict(fam=fam, mode=mode, seed=int(seed), E=int(E), N=n_backbone(d["config"]),
                         D=d["train_pool_content_tokens"], gap=d["final_gap"], run_id=d["run_id"]))
    return rows


def epoch_ratios():
    """gap(E)/gap(3) from fixed-2000-step curves (constant LR => E-run is a prefix of the curve)."""
    out = {}
    for f in glob.glob(os.path.join(H_224, "v2[34]*_2000_fd_fixed", "summary.json")):
        d = json.load(open(f))
        if "_n25_" in d["run_id"]:
            continue
        D = d.get("train_pool_content_tokens") or d["train_max_tokens"]
        cur = [json.loads(l) for l in open(f.replace("summary.json", "train_log.jsonl"))]
        st = np.array([x["step"] for x in cur]); g = np.array([x["gap"] for x in cur])

        def at(E):
            s = E * D / TB
            m = np.abs(st - s) <= max(10, 0.03 * s)
            return g[m].mean() if s <= 2000 and m.any() else np.nan

        mode = re.search(r"_(n[\d-]+|tri)_", d["run_id"]).group(1)
        g3 = at(3)
        out.setdefault(mode, []).append([at(E) / g3 for E in EGRID])
    return {k: np.nanmean(np.array(v), 0) for k, v in out.items()}


EGRID = [1, 2, 3, 4, 5, 6, 8]


def design(N, D, kind):
    lN, lD = np.log(N), np.log(D)
    cols = [np.ones_like(lN), lN, lD] + ([lN * lD] if kind == "int" else [])
    return np.stack(cols, -1)


def fit(rows, kind, boot=2000, seed=0):
    N = np.array([r["N"] for r in rows], float); D = np.array([r["D"] for r in rows], float)
    y = np.log([r["gap"] for r in rows]); X = design(N, D, kind)
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed); B = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y)); B.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    rmse = float(np.sqrt(np.mean((y - X @ c) ** 2)))
    return c, np.array(B), rmse


def predict(c, B, N, D, kind):
    x = design(np.atleast_1d(np.asarray(N, float)), np.atleast_1d(np.asarray(D, float)), kind)
    p = np.exp(x @ c); pb = np.exp(x @ B.T)
    return p, np.percentile(pb, 5, axis=1), np.percentile(pb, 95, axis=1)


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, which="major", color=BORDER, lw=0.5, alpha=0.6)


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(APP, exist_ok=True)
    R = load_e3()
    tri = [r for r in R if r["fam"] == "v24ms" and r["mode"] == "tri" and r["E"] == 3]
    n25 = [r for r in R if r["fam"] == "v24ms" and r["mode"] == "n25" and r["E"] == 3]
    fits = {k: fit(tri, k) for k in ("sep", "int")}
    fit_n25 = fit(n25, "sep")
    ratios = epoch_ratios()

    # held-out checks (largest pool / largest model)
    hold = {}
    for kind in ("sep", "int"):
        for name, keep in (("D=49.7M", lambda r: r["D"] < 4e7), ("N=161M", lambda r: r["N"] < 1.5e8)):
            tr = [r for r in tri if keep(r)]; te = [r for r in tri if not keep(r)]
            c, B, _ = fit(tr, kind, boot=10)
            p = predict(c, B, [r["N"] for r in te], [r["D"] for r in te], kind)[0]
            hold[f"{kind}:{name}"] = float(np.mean(p - np.array([r["gap"] for r in te])))

    # order-set D slopes at N=64.5M (clean tables, v23d)
    orders = ["n2", "n2-3", "n2-3-4", "n2-3-4-5", "n2-3-4-5-6", "n2-3-4-5-6-7"]
    oslope = {}
    for o in orders:
        rr = [r for r in R if r["fam"] == "v23d" and r["mode"] == o and r["E"] == 3]
        X = np.stack([np.ones(len(rr)), np.log([r["D"] for r in rr])], 1)
        cc = np.linalg.lstsq(X, np.log([r["gap"] for r in rr]), rcond=None)[0]
        oslope[o] = (float(cc[1]), float(np.exp(cc[0] + cc[1] * np.log(49_692_672))))

    # ---------------- Fig 1: 在测范围内的 scaling ----------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
    Ns = sorted({r["N"] for r in tri}); Ds = sorted({r["D"] for r in tri})
    c, B, _ = fits["sep"]
    ax = axs[0]; style(ax)
    for i, N in enumerate(Ns):
        pts = [r for r in tri if r["N"] == N]
        ax.scatter([r["D"] / 1e6 for r in pts], [r["gap"] for r in pts], s=14, color=NCOL[i], alpha=0.8)
        dd = np.geomspace(min(Ds), max(Ds), 50)
        ax.plot(dd / 1e6, predict(c, B, np.full_like(dd, N), dd, "sep")[0], color=NCOL[i], lw=1.4,
                label=f"backbone {N/1e6:.0f}M")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("唯一数据量 D（M token）", color=MUTED); ax.set_ylabel("final gap（E=3）", color=MUTED)
    ax.set_title("(a) trigram：gap 随数据量下降", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]; style(ax)
    for i, D in enumerate(Ds):
        pts = [r for r in tri if r["D"] == D]
        col = plt.cm.viridis(i / (len(Ds) - 1))
        ax.scatter([r["N"] / 1e6 for r in pts], [r["gap"] for r in pts], s=14, color=col, alpha=0.8)
        nn = np.geomspace(min(Ns), max(Ns), 50)
        ax.plot(nn / 1e6, predict(c, B, nn, np.full_like(nn, D), "sep")[0], color=col, lw=1.4,
                label=f"D={D/1e6:.0f}M")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("backbone 参数 N（M）", color=MUTED); ax.set_ylabel("final gap（E=3）", color=MUTED)
    ax.set_title("(b) trigram：gap 随模型变大上升", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axs[2]; style(ax)
    xs = np.arange(len(orders))
    ax.bar(xs, [oslope[o][1] for o in orders], color=ANCHOR, alpha=0.75, width=0.6)
    ax2 = ax.twinx(); ax2.plot(xs, [oslope[o][0] for o in orders], "o-", color="#c4493d")
    ax2.axhline(0, color=BORDER, lw=0.8); ax2.tick_params(colors="#c4493d", labelsize=8)
    ax2.set_ylabel("数据量指数 b（越接近 0 越不随数据缓解）", color="#c4493d", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([o[1:] for o in orders], fontsize=7, rotation=20)
    ax.set_xlabel("n-gram 阶集合（clean 表，backbone 64.5M）", color=MUTED)
    ax.set_ylabel("final gap @49.7M（E=3）", color=MUTED)
    ax.set_title("(c) n 越大，gap 越高且越不随数据下降", fontsize=10)
    fig.suptitle(
        f"在测范围：log gap = {c[0]:.2f} + {c[1]:.3f}·log N {c[2]:+.3f}·log D"
        f"（trigram，E=3，{len(tri)} 臂，log-RMSE {fits['sep'][2]:.3f}）", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"fig_scaling_fit.{ext}"), dpi=160, facecolor=BG)
    plt.close(fig)

    # ---------------- 外推目标 ----------------
    E4 = float(ratios["tri"][EGRID.index(4)])
    targets = [
        # name, N used by fit, D (unique, repeated E times), note
        ("实验室最大点", 161.5e6, 49.7e6, "已测"),
        ("Dense 1B · D=20B", 1e9, 20e9, "第一阶段"),
        ("FLAME-Large 1.3B 总量 · D=26B（N 取总量）", 1.3e9, 26e9, "第一阶段 MoE 上界"),
        ("FLAME-Large 290M 激活 · D=26B（N 取激活）", 0.29e9, 26e9, "第一阶段 MoE 下界"),
        ("Dense 20B · D=400B", 20e9, 400e9, "第一阶段"),
        ("Dense 100B · D=2T", 100e9, 2e12, "生产（全量重复口径）"),
        ("DSv4.1-Flash 750B 总量 · D=15T", 750e9, 15e12, "生产（全量重复口径）"),
        ("Dense 100B · 重复子集 100B", 100e9, 100e9, "生产（只重复 100B 研究语料）"),
        ("750B 总量 · 重复子集 100B", 750e9, 100e9, "生产（只重复 100B 研究语料）"),
        ("10B 激活 · 重复子集 100B", 10e9, 100e9, "生产（只重复 100B 研究语料）"),
    ]
    out = []
    for name, N, D, note in targets:
        row = dict(target=name, N=N, D=D, D_over_N=D / N, note=note)
        for kind in ("sep", "int"):
            p, lo, hi = predict(*fits[kind][:2], N, D, kind)
            row[f"{kind}_E3"], row[f"{kind}_E3_lo"], row[f"{kind}_E3_hi"] = float(p[0]), float(lo[0]), float(hi[0])
            row[f"{kind}_E4"] = float(p[0]) * E4
        out.append(row)
    with open(os.path.join(APP, "extrapolation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    # ---------------- Fig 2: 沿 D=20N 外推 ----------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)
    nn = np.geomspace(1e7, 1e12, 200)
    for ax, (ratio_lbl, Dfun) in zip(axs, (("D = 20·N（Chinchilla 口径，全量重复 3 遍）", lambda n: 20 * n),
                                            ("D = 100B（生产：只重复 100B 研究语料）", lambda n: np.full_like(n, 1e11)))):
        style(ax)
        for kind, col, lbl in (("sep", ANCHOR, "可分幂律（主估计）"), ("int", "#c4493d", "含 N×D 交互项（上界情景）")):
            p, lo, hi = predict(*fits[kind][:2], nn, Dfun(nn), kind)
            ax.plot(nn, p, color=col, lw=1.8, label=lbl)
            ax.fill_between(nn, lo, hi, color=col, alpha=0.15, lw=0)
        ax.axvspan(min(Ns), max(Ns), color="#3c8d5a", alpha=0.12, lw=0)
        ax.text(np.sqrt(min(Ns) * max(Ns)), 0.12, "已测\nN 范围", ha="center", fontsize=7, color="#3c8d5a")
        ax.axhline(0.03, color=MUTED, ls=":", lw=1)
        ax.text(1.2e7, 0.034, "1 pass 下的 gap 量级（E=1 ≈ 0.02–0.06 × E3）", fontsize=7, color=MUTED)
        marks = [(1e9, "1B"), (1.3e9, ""), (20e9, "20B"), (100e9, "100B"), (750e9, "750B")]
        for x, t in marks:
            ax.axvline(x, color=BORDER, lw=0.8, ls="--")
            if t:
                ax.text(x, 60, t, fontsize=8, color=MUTED, ha="center")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.02, 200)
        ax.set_xlabel("模型参数 N（MoE 取总量为上界、激活量为下界）", color=MUTED)
        ax.set_ylabel("预测 trigram gap（E=3）", color=MUTED)
        ax.set_title(ratio_lbl, fontsize=10); ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("外推：两种拟合在已测范围内同样好，到 1B 以上分道扬镳（阴影 = bootstrap 90%）", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"fig_extrapolation.{ext}"), dpi=160, facecolor=BG)
    plt.close(fig)

    # ---------------- Fig 3: epoch 剂量 + 表容量 ----------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=BG)
    ax = axs[0]; style(ax)
    for i, o in enumerate(["tri", "n2", "n2-3", "n2-3-4", "n2-3-4-5", "n2-3-4-5-6-7"]):
        ax.plot(EGRID, ratios[o], "o-", ms=3, lw=1.5, color=(NCOL + ["#555"])[i % 6], label=o)
    ax.axvspan(3, 4, color="#c4493d", alpha=0.08, lw=0)
    ax.text(3.5, 0.15, "生产关注\n第 3–4 遍", ha="center", fontsize=7, color="#c4493d")
    ax.set_xlabel("重复遍数 E", color=MUTED); ax.set_ylabel("gap(E) / gap(E=3)", color=MUTED)
    ax.set_title("(a) epoch 剂量：第 1 遍几乎没有 gap，第 2 遍起跳，之后低阶增长更快", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax = axs[1]; style(ax)
    rmap = {"1": 1, "10": 10, "100": 100, "1k": 1e3, "10k": 1e4, "100k": 1e5, "300k": 3e5, "1m": 1e6, "2m": 2e6, "3m": 3e6}
    distinct = {2: 3.4, 3: 18.1, 4: 33.2, 5: 41.0, 6: 44.2, 7: 45.5}  # §76: distinct/R at R=2^20, D=49.7M
    for i, o in enumerate(["n2", "n2-3", "n2-3-4", "n2-3-4-5", "n2-3-4-5-6-7"]):
        pts = []
        for f in glob.glob(os.path.join(H_224, f"v22r_{o}_r*_s42_2000_fd_fixed", "summary.json")):
            d = json.load(open(f)); rr = re.search(r"_r(\w+?)_s42", d["run_id"]).group(1)
            hi = int(o.split("-")[-1][-1]); load = distinct[hi] * 2 ** 20 / rmap[rr]
            pts.append((load, d["final_gap"]))
        pts.sort()
        ax.plot([p[0] for p in pts], [max(p[1], 1e-2) for p in pts], "o-", ms=3, lw=1.5, color=NCOL[i], label=o)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("每行承载的上下文数（最高阶 distinct / R，越往右表越大）", color=MUTED)
    ax.set_ylabel("final gap（2000 步 ≈ 6 遍）", color=MUTED)
    ax.set_title("(b) 表容量：每行上下文越多 gap 越小，>10⁴ 个/行时基本消失", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"fig_epoch_R.{ext}"), dpi=160, facecolor=BG)
    plt.close(fig)

    params = dict(
        data=dict(n_tri=len(tri), n_n25_legacy=len(n25), N_range=[min(Ns), max(Ns)], D_range=[min(Ds), max(Ds)]),
        tri_sep=dict(coef=fits["sep"][0].tolist(), ci90=np.percentile(fits["sep"][1], [5, 95], 0).tolist(), log_rmse=fits["sep"][2]),
        tri_int=dict(coef=fits["int"][0].tolist(), ci90=np.percentile(fits["int"][1], [5, 95], 0).tolist(), log_rmse=fits["int"][2]),
        n25_legacy_sep=dict(coef=fit_n25[0].tolist(), log_rmse=fit_n25[2]),
        holdout_mean_bias=hold,
        order_D_slope_and_gap49=oslope,
        epoch_ratio_vs_E3={k: dict(zip(map(str, EGRID), np.round(v, 3).tolist())) for k, v in ratios.items()},
    )
    json.dump(params, open(os.path.join(APP, "fit_params.json"), "w"), indent=1, ensure_ascii=False)
    for r in out:
        print(f"{r['target']:42s} D/N={r['D_over_N']:7.2f}  sep E3 {r['sep_E3']:.2f} [{r['sep_E3_lo']:.2f},{r['sep_E3_hi']:.2f}] E4 {r['sep_E4']:.2f} | int E3 {r['int_E3']:.1f}")
    print(json.dumps({k: params[k] for k in ("holdout_mean_bias",)}, ensure_ascii=False))
    print("sep a+b CI:", np.percentile(fits["sep"][1][:, 1] + fits["sep"][1][:, 2], [5, 95]))
    print("int coef CI:", params["tri_int"]["ci90"])


if __name__ == "__main__":
    main()
