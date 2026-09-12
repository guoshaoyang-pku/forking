#!/usr/bin/env python3
"""§53 logit-sharpening probe recovery: ls20ep input vs nogram (20 epochs, 128x).

Sources (authoritative, data/runs_fixed/):
  - ls20ep_input_v5_128x_fd_fixed   (seed 42, shard-1 replay, 6740 steps, probe every 10)
  - ls20ep_nogram_v5_128x_fd_fixed  (same, no table)
Inputs per run: logit_stats.jsonl (agg + exemplar rows), freq_bin_loss.jsonl (val CE per bin),
               train_log.jsonl (online train / fixed val / gap).

Metric separation (IMPORTANT):
  - CE side:  val mean_loss per freq bin (true-token dependent, = damage axis)
  - sharpening side: mean_entropy H(p), mean_margin p(1)-p(2), mean_ptrue p(y_true)
    (entropy/margin are true-token independent distribution-shape measures)

Outputs:
  - docs/appendices/ls20ep_logit_stats/agg_epoch_boundary.csv
  - docs/appendices/ls20ep_logit_stats/freqbin_val_ce_epoch_boundary.csv
  - docs/appendices/ls20ep_logit_stats/exemplars_selected.csv
  - docs/figs/main/fig_ls20ep_sharpening_vs_ce.png
  - docs/figs/main/fig_ls20ep_exemplar_evolution.png
Points = raw per-epoch-boundary records; thin lines = visual connection only (no smoothing).
"""
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "input": ROOT / "data/runs_fixed/ls20ep_input_v5_128x_fd_fixed",
    "nogram": ROOT / "data/runs_fixed/ls20ep_nogram_v5_128x_fd_fixed",
}
APP = ROOT / "docs/appendices/ls20ep_logit_stats"
FIGS = ROOT / "docs/figs/main"
APP.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

EPOCH_STEPS = 337
METRICS = ["mean_entropy", "mean_margin", "mean_ptrue"]
SHOW_SEEN = ["f1", "f2-3", "f16-63", "f256+"]
COLORS = {
    "novel-f0": "#000000",
    "f1": "#d55e00",
    "f2-3": "#e69f00",
    "f4-7": "#cc79a7",
    "f8-15": "#56b4e9",
    "f16-63": "#0072b2",
    "f64-255": "#009e73",
    "f256+": "#009e73",
}
CE_COLORS = {"novel": "#000000", "1": "#d55e00", "6-8": "#e69f00", "31-50": "#0072b2"}


def last_per_epoch(rows):
    """rows: iterable of dicts with 'epoch','step' -> keep max step per epoch."""
    out = {}
    for d in rows:
        ep = d["epoch"]
        if ep not in out or d["step"] > out[ep]["step"]:
            out[ep] = d
    return out


def load_agg(run_dir):
    """-> {branch: {epoch: {"step":s, (group,bucket): {metric: val, "count": n}}}}"""
    data = {}
    with open(run_dir / "logit_stats.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("type") != "agg":
                continue
            rec = {"step": d["step"]}
            for grp, buckets in d["buckets"].items():
                for bname, v in buckets.items():
                    rec[(grp, bname)] = v
            cur = data.setdefault(d["branch"], {})
            ep = d["epoch"]
            if ep not in cur or d["step"] > cur[ep]["step"]:
                cur[ep] = rec
    return data


def load_exemplars(run_dir):
    """-> {branch: {exemplar_id: {"hit_count":h, epoch: {"step":s,"probs":[...],"ids":[...]}}}}"""
    data = {}
    with open(run_dir / "logit_stats.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("type") != "exemplar":
                continue
            br = data.setdefault(d["branch"], {})
            ex = br.setdefault(d["exemplar_id"], {"hit_count": d["hit_count"], "epochs": {}})
            ep = d["epoch"]
            if ep not in ex["epochs"] or d["step"] > ex["epochs"][ep]["step"]:
                ex["epochs"][ep] = {
                    "step": d["step"],
                    "probs": d["top20_probs"],
                    "ids": d["top20_ids"],
                }
    return data


def load_freqbin(run_dir):
    """-> {branch: {epoch: {"step":s, "val": {bin: {"mean_loss":x,"token_count":n,"frac":f}}}}}"""
    rows = []
    with open(run_dir / "freq_bin_loss.jsonl") as fh:
        for line in fh:
            rows.append(json.loads(line))
    out = {}
    by_branch = {}
    for d in rows:
        for br, bins in d["val"].items():
            by_branch.setdefault(br, []).append({"epoch": d["epoch"], "step": d["step"], "bins": bins})
    for br, rws in by_branch.items():
        out[br] = {ep: rec for ep, rec in last_per_epoch(rws).items()}
    return out


def load_train_log(run_dir):
    rows = [json.loads(l) for l in open(run_dir / "train_log.jsonl")]
    return last_per_epoch(rows)


def main():
    agg = {r: load_agg(d) for r, d in RUNS.items()}
    freq = {r: load_freqbin(d) for r, d in RUNS.items()}
    tlog = {r: load_train_log(d) for r, d in RUNS.items()}
    exemp = {r: load_exemplars(d) for r, d in RUNS.items()}

    # ---------- CSV: agg epoch boundary ----------
    with open(APP / "agg_epoch_boundary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "branch", "epoch", "step", "group", "bucket", "count",
                    "mean_entropy", "mean_margin", "mean_ptrue"])
        for run, branches in agg.items():
            for br, eps in branches.items():
                for ep in sorted(eps):
                    rec = eps[ep]
                    for (grp, bname), v in sorted((kv for kv in rec.items() if isinstance(kv[1], dict))):
                        w.writerow([run, br, ep, rec["step"], grp, bname, v["count"],
                                    v["mean_entropy"], v["mean_margin"], v["mean_ptrue"]])

    # ---------- CSV: freq_bin val CE ----------
    def bin_key(s):
        if s == "novel":
            return -1.0
        m = re.match(r"\d+", s.lower().replace("k", "000"))
        return float(m.group()) if m else 1e9

    all_bins = sorted({b for run in freq for br in freq[run] for b in freq[run][br][1]["bins"]}, key=bin_key)
    with open(APP / "freqbin_val_ce_epoch_boundary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "branch", "epoch", "step", "bin", "token_count", "frac", "mean_loss"])
        for run, branches in freq.items():
            for br, eps in branches.items():
                for ep in sorted(eps):
                    rec = eps[ep]
                    for bname, v in rec["bins"].items():
                        w.writerow([run, br, ep, rec["step"], bname,
                                    v["token_count"], v["frac"], v["mean_loss"]])

    # ---------- console tables ----------
    print("=" * 100)
    print("TRAIN LOG @ epoch boundaries (gap = val - train, online train / fixed val)")
    for run in RUNS:
        e20 = tlog[run][20]
        e1 = tlog[run][1]
        print(f"  {run:8s} e1(step {e1['step']}): train {e1['train_loss']:.3f} val {e1['val_loss']:.3f} gap {e1['gap']:.3f}"
              f" | e20(step {e20['step']}): train {e20['train_loss']:.3f} val {e20['val_loss']:.3f} gap {e20['gap']:.3f}")

    for br in ["bigram", "trigram"]:
        print("=" * 100)
        print(f"BRANCH {br} — sharpening metrics at epoch boundaries (input vs nogram)")
        header = f"{'bucket':>10s} {'metric':>13s} " + " ".join(f"{'e'+str(e):>9s}" for e in [1, 5, 10, 15, 20]) + f" {'Δe20-e1':>9s}"
        for run in ["input", "nogram"]:
            print(f"-- run={run}")
            print(header)
            series = {}
            for grp, bname in [("novel", "f0")] + [("seen", b) for b in SHOW_SEEN]:
                for m in METRICS:
                    vals = []
                    for e in [1, 5, 10, 15, 20]:
                        v = agg[run][br][e].get((grp, bname))
                        vals.append(v[m] if v and v["count"] > 0 else float("nan"))
                    series[(grp, bname, m)] = vals
                    print(f"{grp+'-'+bname:>10s} {m:>13s} " + " ".join(f"{x:9.4f}" for x in vals)
                          + f" {vals[-1]-vals[0]:+9.4f}")
            # counts at e20 for reference
            cnts = {f"{g}-{b}": agg[run][br][20].get((g, b), {}).get("count", 0)
                    for g, b in [("novel", "f0")] + [("seen", b) for b in SHOW_SEEN]}
            print(f"   e20 token counts: {cnts}")

        print(f"-- val CE per freq_bin (native bins), run=input vs nogram")
        ce_bins = ["novel", "1", "6-8", "31-50", all_bins[-1]]
        for run in ["input", "nogram"]:
            print(f"   run={run}")
            for b in ce_bins:
                vals = [freq[run][br][e]["bins"].get(b, {}).get("mean_loss", float("nan"))
                        for e in [1, 5, 10, 15, 20]]
                fr = freq[run][br][20]["bins"].get(b, {}).get("frac", float("nan"))
                print(f"     {b:>8s} (frac {fr:.3f}): " + " ".join(f"{x:8.3f}" for x in vals))

    # novel share consistency check (task-1 caveat): freq_bin novel frac per branch/run
    print("=" * 100)
    print("NOVEL TOKEN FRAC (freq_bin val, e20) — branch composition check")
    for run in RUNS:
        for br in ["bigram", "trigram"]:
            fr = freq[run][br][20]["bins"]["novel"]["frac"]
            print(f"  {run:8s} {br:8s} novel frac = {fr:.4f}")

    # ---------- Figure 1: sharpening vs CE ----------
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.6), sharex=True)
    epochs = list(range(1, 21))
    for ri, br in enumerate(["bigram", "trigram"]):
        # col 0: val CE (native freq bins)
        ax = axes[ri][0]
        for b in ["novel", "1", "6-8", "31-50"]:
            for run, ls, lw in [("input", "-", 1.6), ("nogram", "--", 1.0)]:
                ys = [freq[run][br][e]["bins"].get(b, {}).get("mean_loss", float("nan")) for e in epochs]
                ax.plot(epochs, ys, ls, color=CE_COLORS[b], linewidth=lw, alpha=1.0 if run == "input" else 0.55,
                        marker="o" if run == "input" else None, markersize=2.5)
                if run == "input":
                    ax.annotate(f"{b}", (epochs[-1], ys[-1]), fontsize=8, color=CE_COLORS[b],
                                xytext=(4, 0), textcoords="offset points")
        ax.set_ylabel("val CE (nats/token)")
        ax.set_title(f"{br} · val CE per freq-bin (damage axis)", fontsize=10)
        ax.set_xlim(0.5, 22.5)
        # cols 1-3: entropy / margin / ptrue
        for ci, m in enumerate(METRICS, start=1):
            ax = axes[ri][ci]
            for grp, bname in [("novel", "f0")] + [("seen", b) for b in SHOW_SEEN]:
                col = COLORS["novel-f0"] if bname == "f0" else COLORS[bname]
                lbl = f"novel f0" if bname == "f0" else f"seen {bname}"
                for run, ls, lw, alpha in [("input", "-", 1.6, 1.0), ("nogram", "--", 1.0, 0.55)]:
                    ys = []
                    for e in epochs:
                        v = agg[run][br][e].get((grp, bname))
                        ys.append(v[m] if v and v["count"] > 0 else float("nan"))
                    ax.plot(epochs, ys, ls, color=col, linewidth=lw, alpha=alpha,
                            marker="o" if run == "input" else None, markersize=2.5,
                            label=lbl if (run == "input" and ci == 1 and ri == 0) else None)
                if ci == 3:
                    ax.annotate(lbl, (epochs[-1], ys[-1]), fontsize=8, color=col,
                                xytext=(4, 0), textcoords="offset points")
            if m != "mean_entropy":
                ax.set_yscale("log")
            ax.set_title(f"{br} · {m}" + (" (shape axis, log-y)" if m != "mean_entropy" else " (shape axis)"), fontsize=10)
            ax.set_xlim(0.5, 22.5)
    for ri in range(2):
        axes[ri][0].set_xlabel("epoch (boundary eval, last step of epoch)")
        for ci in range(4):
            axes[ri][ci].set_xlabel("epoch (boundary eval)")
    handles = [
        plt.Line2D([], [], color="#444444", ls="-", lw=1.6, marker="o", ms=3, label="input (n-gram table)"),
        plt.Line2D([], [], color="#444444", ls="--", lw=1.0, label="nogram control"),
        plt.Line2D([], [], color=CE_COLORS["novel"], ls="-", label="novel (f=0)"),
        plt.Line2D([], [], color=COLORS["f1"], ls="-", label="seen f=1"),
        plt.Line2D([], [], color=COLORS["f2-3"], ls="-", label="seen f=2-3"),
        plt.Line2D([], [], color=COLORS["f16-63"], ls="-", label="seen f=16-63"),
        plt.Line2D([], [], color=COLORS["f256+"], ls="-", label="seen f=256+"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle("logit sharpening vs CE damage over 20 replay epochs · 128× · seed 42 · shard-1 replay\n"
                 "runs: ls20ep_input_v5_128x_fd vs ls20ep_nogram_v5_128x_fd · probe every 10 steps, epoch-boundary samples shown\n"
                 "CE = true-token damage axis; entropy/margin/p(y_true) = distribution-shape axis (margin = p(top1)−p(top2), probability scale)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(FIGS / "fig_ls20ep_sharpening_vs_ce.png", dpi=150)
    plt.close(fig)
    print(f"[fig] {FIGS/'fig_ls20ep_sharpening_vs_ce.png'}")

    # ---------- Figure 2: exemplar evolution ----------
    br = "bigram"
    ex_in = exemp["input"].get(br, {})
    ex_nog = exemp["nogram"].get(br, {})
    if not ex_in:
        # exemplars may be tagged under a single branch key
        for k, v in exemp["input"].items():
            if v:
                br, ex_in = k, v
                ex_nog = exemp["nogram"].get(k, {})
                break
    seen_ids = [i for i, v in ex_in.items() if v["hit_count"] > 0]
    novel_ids = [i for i, v in ex_in.items() if v["hit_count"] == 0]
    picks = []
    if seen_ids:
        hi = max(seen_ids, key=lambda i: ex_in[i]["hit_count"])
        lo = min(seen_ids, key=lambda i: ex_in[i]["hit_count"])
        picks.append(("high-F seen", hi))
        picks.append(("low-F seen", lo))
    if novel_ids:
        picks.append(("novel (f=0)", novel_ids[0]))
    cmap = plt.get_cmap("viridis")
    ep_sel = [1, 5, 10, 15, 20]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 4.0), sharey=True)
    if len(picks) == 1:
        axes = [axes]
    ex_rows = []
    for ax, (label, eid) in zip(axes, picks):
        for k, e in enumerate(ep_sel):
            rec = ex_in[eid]["epochs"].get(e)
            if not rec:
                continue
            c = cmap(k / (len(ep_sel) - 1))
            ax.plot(range(1, len(rec["probs"]) + 1), rec["probs"], "-o", color=c, lw=1.4, ms=3,
                    label=f"epoch {e} (step {rec['step']})")
            ex_rows.append({"run": "input", "branch": br, "quadrant": label, "exemplar_id": eid,
                            "hit_count": ex_in[eid]["hit_count"], "epoch": e, "step": rec["step"],
                            "rank": None, "top20_probs": json.dumps(rec["probs"]),
                            "top20_ids": json.dumps(rec["ids"])})
        rec = ex_nog.get(eid, {}).get("epochs", {}).get(20)
        if rec:
            ax.plot(range(1, len(rec["probs"]) + 1), rec["probs"], "--", color="#888888", lw=1.2,
                    label="nogram e20 (control)")
            ex_rows.append({"run": "nogram", "branch": br, "quadrant": label, "exemplar_id": eid,
                            "hit_count": ex_in[eid]["hit_count"], "epoch": 20, "step": rec["step"],
                            "rank": None, "top20_probs": json.dumps(rec["probs"]),
                            "top20_ids": json.dumps(rec["ids"])})
        ax.set_yscale("log")
        ax.set_xlabel("rank in top-20")
        ax.set_title(f"{label} · hit_count={ex_in[eid]['hit_count']} · id={eid}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("softmax prob (log)")
    fig.suptitle(f"exemplar context top-20 softmax evolution ({br} branch, input arm) · flat → peaked over 20 replay epochs\n"
                 "runs: ls20ep_input_v5_128x_fd / ls20ep_nogram_v5_128x_fd · seed 42 · 128× · prob scale (shift-invariant)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGS / "fig_ls20ep_exemplar_evolution.png", dpi=150)
    plt.close(fig)
    print(f"[fig] {FIGS/'fig_ls20ep_exemplar_evolution.png'}")

    with open(APP / "exemplars_selected.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ex_rows[0].keys()))
        w.writeheader()
        w.writerows(ex_rows)
    print(f"[csv] {APP}/agg_epoch_boundary.csv, freqbin_val_ce_epoch_boundary.csv, exemplars_selected.csv")


if __name__ == "__main__":
    main()
