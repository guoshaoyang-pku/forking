#!/usr/bin/env python3
"""f x pass full gap accounting (batch 0 of the y/v scaling wave).

Reads the replay-snapshot decomp.jsonl files (input / nogram / freezebb arms,
bigram branch, 6 pass-boundary steps) and produces the complete damage table:

  total damage(f, pass)   = input_full  - nogram          (per-token nats)
  table-direct(f, pass)   = input_full  - input_no_table  (same weights, table off)
  backbone-carried(f,pass)= input_no_table - nogram       (backbone only)
  margin(f, pass)         = logit(train-dominant cont.) - logit(true token)

Outputs:
  docs/figs/theory/tab_v5_f_by_pass_accounting.csv
  docs/figs/theory/fig_v5_f_by_pass_accounting.png

Source data: data/runs_theory/{input,nogram,freezebb}_snap/decomp.jsonl on
ophis-gpu (replay of seed-42 128x standard, pass-boundary snapshots).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STEPS = [337, 674, 1011, 1348, 1685, 2022]
PASS = {s: i + 1 for i, s in enumerate(STEPS)}

# exact low f, then doubling bins
BINS = ([(i, i) for i in range(1, 9)]
        + [(9, 12), (13, 18), (19, 26), (27, 38), (39, 54), (55, 78),
           (79, 110), (111, 156), (157, 220), (221, 312), (313, 440),
           (441, 624), (625, 880), (881, 1244), (1245, 2048),
           (2049, 10**9)])
BIN_LABELS = ([str(i) for i in range(1, 9)]
              + [f"{a}-{b}" for a, b in BINS[8:-1]] + ["2049+"])
F0 = (0, 0)  # f=0 handled separately (novel-only)


def load(path):
    recs = {}
    for line in open(path):
        r = json.loads(line)
        recs[(r["step"], r["f"])] = r
    return recs


def bucket_of(f):
    if f == 0:
        return -1
    for i, (a, b) in enumerate(BINS):
        if a <= f <= b:
            return i
    raise ValueError(f)


def aggregate(recs):
    """returns dict (step, bucket) -> summed 12-vector"""
    out = {}
    for (step, f), r in recs.items():
        bi = bucket_of(f)
        key = (step, bi)
        a = out.setdefault(key, np.zeros(12))
        a += [r["n_seen"], r["n_novel"], r["loss_seen_t"], r["loss_novel_t"],
              r["loss_seen_c"], r["loss_novel_c"], r["marg_seen_sum"],
              r["marg_seen_sq"], r["marg_novel_sum"], r["marg_novel_sq"],
              r["marg_seen_n"], r["marg_novel_n"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp_dir", default="/tmp/replay_decomp")
    ap.add_argument("--out_csv", default=str(ROOT / "docs/figs/theory/tab_v5_f_by_pass_accounting.csv"))
    ap.add_argument("--out_fig", default=str(ROOT / "docs/figs/theory/fig_v5_f_by_pass_accounting.png"))
    args = ap.parse_args()

    d = Path(args.decomp_dir)
    inp = aggregate(load(d / "input_decomp.jsonl"))
    ng = aggregate(load(d / "nogram_decomp.jsonl"))

    nb = len(BINS) + 1  # +1 for f=0 at index -1 -> 0
    bidx = [-1] + list(range(len(BINS)))
    blabels = ["0"] + BIN_LABELS
    xs = np.arange(len(bidx))

    rows = []
    # per-bucket arrays for plotting: [pass][bucket]
    seen_dmg = np.full((6, nb), np.nan)
    novel_dmg = np.full((6, nb), np.nan)
    seen_td = np.full((6, nb), np.nan)   # table-direct seen
    novel_td = np.full((6, nb), np.nan)
    seen_bc = np.full((6, nb), np.nan)   # backbone-carried seen
    novel_bc = np.full((6, nb), np.nan)
    marg_n = np.full((6, nb), np.nan)
    tok_seen = np.zeros(nb)
    tok_novel = np.zeros(nb)

    for pi, step in enumerate(STEPS):
        for bj, bi in enumerate(bidx):
            a = inp.get((step, bi))
            g = ng.get((step, bi))
            if a is None:
                continue
            ns, nn = a[0], a[1]
            if pi == 0:
                tok_seen[bj], tok_novel[bj] = ns, nn
            if ns > 0:
                seen_dmg[pi, bj] = (a[2] - g[2]) / ns
                seen_td[pi, bj] = (a[2] - a[4]) / ns
                seen_bc[pi, bj] = (a[4] - g[2]) / ns
            if nn > 0:
                novel_dmg[pi, bj] = (a[3] - g[3]) / nn
                novel_td[pi, bj] = (a[3] - a[5]) / nn
                novel_bc[pi, bj] = (a[5] - g[3]) / nn
            if a[11] > 0:
                marg_n[pi, bj] = a[8] / a[11]
            if pi in (0, 5):
                rows.append(dict(
                    pass_=pi + 1, f_bin=blabels[bj], n_seen=int(ns), n_novel=int(nn),
                    seen_total=seen_dmg[pi, bj], seen_table_direct=seen_td[pi, bj],
                    seen_backbone=seen_bc[pi, bj], novel_total=novel_dmg[pi, bj],
                    novel_table_direct=novel_td[pi, bj], novel_backbone=novel_bc[pi, bj],
                    margin_novel=marg_n[pi, bj]))

    # per-pass increments (slope over 6 passes) for margin and novel damage
    def slope(y):
        m = ~np.isnan(y)
        if m.sum() < 3:
            return np.nan
        return np.polyfit(np.arange(1, 7)[m], y[m], 1)[0]

    r_marg = [slope(marg_n[:, bj]) for bj in range(nb)]
    r_novel = [slope(novel_dmg[:, bj]) for bj in range(nb)]

    with open(args.out_csv, "w") as fh:
        fh.write("pass,f_bin,n_seen,n_novel,seen_total,seen_table_direct,seen_backbone,"
                 "novel_total,novel_table_direct,novel_backbone,margin_novel\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in
                              ["pass_", "f_bin", "n_seen", "n_novel", "seen_total",
                               "seen_table_direct", "seen_backbone", "novel_total",
                               "novel_table_direct", "novel_backbone", "margin_novel"]) + "\n")
        fh.write("\nslope_per_pass,f_bin,r_margin,r_novel_damage\n")
        for bj in range(nb):
            fh.write(f",{blabels[bj]},{r_marg[bj]},{r_novel[bj]}\n")

    # ---- figure ----
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("f x pass full gap accounting (bigram branch, replay seed 42, 128x)\n"
                 "damage = input - nogram per-token val loss (nats); passes 1-6",
                 fontsize=12)

    ax = axes[0, 0]
    for pi in range(6):
        ax.plot(xs, seen_dmg[pi], "o-", ms=3, lw=1, color=cmap[pi], label=f"pass {pi+1}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("(a) seen damage vs f")
    ax.set_xticks(xs); ax.set_xticklabels(blabels, rotation=60, fontsize=7)
    ax.set_ylabel("nats/token"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for pi in range(6):
        ax.plot(xs, novel_dmg[pi], "o-", ms=3, lw=1, color=cmap[pi], label=f"pass {pi+1}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("(b) novel damage vs f")
    ax.set_xticks(xs); ax.set_xticklabels(blabels, rotation=60, fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for pi in range(6):
        ax.plot(xs, marg_n[pi], "o-", ms=3, lw=1, color=cmap[pi], label=f"pass {pi+1}")
    ax.set_title("(c) margin (train-dominant logit - true logit) on novel tokens")
    ax.set_xticks(xs); ax.set_xticklabels(blabels, rotation=60, fontsize=7)
    ax.set_ylabel("logits"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    w = 0.38
    ax.bar(xs - w / 2, novel_td[5], w, label="novel: table-direct", color="#d62728", alpha=0.8)
    ax.bar(xs - w / 2, novel_bc[5], w, bottom=novel_td[5], label="novel: backbone-carried",
           color="#ff9896", alpha=0.8)
    ax.bar(xs + w / 2, seen_td[5], w, label="seen: table-direct", color="#1f77b4", alpha=0.8)
    ax.bar(xs + w / 2, seen_bc[5], w, bottom=seen_td[5], label="seen: backbone-carried",
           color="#aec7e8", alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("(d) pass-6 decomposition: table-direct vs backbone-carried")
    ax.set_xticks(xs); ax.set_xticklabels(blabels, rotation=60, fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out_fig, dpi=160)
    print(f"[done] csv -> {args.out_csv}")
    print(f"[done] fig -> {args.out_fig}")

    # ---- console summary ----
    print("\n=== per-pass slope r(f): margin / novel damage (nats per pass) ===")
    for bj in range(nb):
        print(f"  f={blabels[bj]:>8s}  r_margin={r_marg[bj]:+.3f}  r_novel={r_novel[bj]:+.3f}"
              f"  (tok: seen={int(tok_seen[bj])}, novel={int(tok_novel[bj])})")
    print("\n=== seen damage by pass (key bins) ===")
    for bj in range(nb):
        col = " ".join(f"{seen_dmg[pi, bj]:+.2f}" for pi in range(6))
        print(f"  f={blabels[bj]:>8s}  {col}")


if __name__ == "__main__":
    main()
