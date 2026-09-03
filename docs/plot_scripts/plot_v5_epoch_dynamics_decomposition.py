"""Decompose net-gap growth into within-pass drift vs boundary jumps, and test
"write-once table" / "decaying per-pass write" against raw v5 128x data.

Sources (all authoritative, seed 42, 128x, warmup_constant(100)):
  data/runs_fixed/nglab1x_input_v5_128x_freq10_fd_fixed/{train_log,table_norm}.jsonl
  docs/appendices/s1_scaling_three_axis/s1_epoch_long_replay_points.csv (20ep trigram-only + nogram)
  docs/figs/theory/theory_backbone_lr_epoch_points.csv (blrabs, abs table LR 0.0768)
Output: docs/figs/theory/fig_v5_epoch_dynamics_decomposition.png
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/runs_fixed/nglab1x_input_v5_128x_freq10_fd_fixed"
P = 337  # device steps per pass at 1x dose

tl = [r for r in (json.loads(l) for l in open(RUN / "train_log.jsonl")) if r["step"] <= 2020]  # drop final-step summary point
steps = np.array([r["step"] for r in tl]); gap = np.array([r["gap"] for r in tl])
tn = [json.loads(l) for l in open(RUN / "table_norm.jsonl")]
tsteps = np.array([r["step"] for r in tn])
trms = np.array([r["trigram.layer_01.table_0.rms"] for r in tn])
brms = np.array([r["bigram.layer_01.table_0.rms"] for r in tn])

def mean_gap(lo, hi):
    m = (steps >= lo) & (steps <= hi); return gap[m].mean()

passes, jumps, drifts = [], [], []
for e in range(2, 7):
    lo, hi = (e - 1) * P + 10, min(e * P, steps.max())
    start, end = mean_gap(lo, lo + 20), mean_gap(hi - 20, hi)
    prev_end = mean_gap((e - 1) * P - 20, (e - 1) * P)
    passes.append(e); jumps.append(start - prev_end); drifts.append(end - start)

rows = list(csv.DictReader(open(ROOT / "docs/appendices/s1_scaling_three_axis/s1_epoch_long_replay_points.csv")))
tri = {int(r["epoch"]): float(r["gap"]) for r in rows if "20ep" in r["run_id"] and r["arm"] == "trigram-only"}
nog = {int(r["epoch"]): float(r["gap"]) for r in rows if "20ep" in r["run_id"] and r["arm"] != "trigram-only"}
ep = np.array(sorted(e for e in tri if e in nog)); net = np.array([tri[e] - nog[e] for e in ep])
fit_m = ep >= 5; a, b = np.polyfit(ep[fit_m], net[fit_m], 1)
r2 = 1 - np.var(net[fit_m] - (a * ep[fit_m] + b)) / np.var(net[fit_m])

bl = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(ROOT / "docs/figs/theory/theory_backbone_lr_epoch_points.csv"))]
def series(lr, key):
    pts = sorted((r["pass"], r[key]) for r in bl if abs(r["lr"] - lr) < 1e-12)
    return np.array([p for p, _ in pts]), np.array([v for _, v in pts])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
# (a) raw gap with pass boundaries + jump/drift bars
a0 = ax[0, 0]
a0.plot(steps, gap, ".", ms=2.5, color="#1f77b4", alpha=0.6, label="online gap (raw, input arm)")
k = np.convolve(gap, np.ones(3) / 3, mode="same"); a0.plot(steps, k, "-", lw=0.8, color="#1f77b4")
for e in range(1, 7): a0.axvline(e * P, color="gray", ls=":", lw=0.8)
a0.set_xlabel("step (1x dose: 337 steps / pass)"); a0.set_ylabel("gap = val − online train")
a0.set_title("(a) gap trajectory, 128x input arm, 2022 steps")
ins = a0.inset_axes([0.08, 0.52, 0.42, 0.42])
w = 0.38; x = np.arange(len(passes))
ins.bar(x - w / 2, jumps, w, color="#d62728", label="jump at boundary")
ins.bar(x + w / 2, drifts, w, color="#2ca02c", label="drift within pass")
ins.set_xticks(x); ins.set_xticklabels([f"p{e}" for e in passes], fontsize=8)
ins.tick_params(labelsize=8); ins.set_title("Δgap per pass", fontsize=9); ins.legend(fontsize=7, frameon=False)
a0.legend(loc="lower right", fontsize=8)

# (b) 20ep net gap: linear in pass count from e5
a1 = ax[0, 1]
a1.plot(ep, net, "o-", ms=5, lw=0.8, color="#9467bd", label="net gap = trigram-only − nogram (20ep, 1xL4)")
xx = np.linspace(4.5, 20.5, 10); a1.plot(xx, a * xx + b, "--", color="k", lw=1,
        label=f"linear fit e5–e20: {a:.3f}/pass, R²={r2:.4f}")
a1.set_xlabel("pass (= complete traversal of the fixed train shard)"); a1.set_ylabel("net gap")
a1.set_title("(b) no saturation: net gap linear in passes to e20"); a1.legend(fontsize=8)
a1b = a1.twinx(); inc = np.diff(net)
a1b.bar(ep[1:], inc, width=0.5, alpha=0.25, color="#9467bd"); a1b.set_ylabel("increment per pass", color="#9467bd")
a1b.set_ylim(0, 3)

# (c) train_benefit saturates, val_penalty linear (blrabs, abs table LR locked)
a2 = ax[1, 0]
for lr, c in ((3e-4, "#ff7f0e"), (6e-4, "#1f77b4"), (1e-3, "#2ca02c")):
    p, tb = series(lr, "train_benefit"); _, vp = series(lr, "val_penalty")
    a2.plot(p, tb, "-", color=c, lw=1.2, label=f"train_benefit  lr={lr:g}")
    a2.plot(p, vp, "--", color=c, lw=1.2, label=f"val_penalty  lr={lr:g}")
a2.set_xlabel("pass"); a2.set_ylabel("contribution to net gap")
a2.set_title("(c) blrabs: net gap = train_benefit (saturates ≈2.5, LR-independent)\n+ val_penalty (keeps growing, carries all LR dependence)", fontsize=10)
a2.legend(fontsize=7, ncol=2); a2.set_xlim(0, 30)

# (d) table RMS keeps growing, yet freezing it at e2 keeps 94% of gap
a3 = ax[1, 1]
a3.plot(tsteps, trms, color="#8c564b", label="trigram table RMS")
a3.plot(tsteps, brms, color="#e377c2", label="bigram table RMS")
for e in range(1, 7): a3.axvline(e * P, color="gray", ls=":", lw=0.8)
for e in range(1, 7):
    i = np.argmin(np.abs(tsteps - e * P)); a3.annotate(f"{trms[i]:.2f}", (tsteps[i], trms[i]), textcoords="offset points",
                                                      xytext=(-4, 6), fontsize=7, ha="right")
a3.set_xlabel("step"); a3.set_ylabel("table RMS (no weight decay)")
a3.set_title("(d) table is NOT written once: RMS ×3.8 over passes 1→6", fontsize=10)
a3.text(0.03, 0.95, "yet the growth is irrelevant to gap:\nfreeze_table@e2 → 5.386 vs none 5.733 (94%) @2022\nfreeze_table@e1 → 3.452 > none 2.724 @1000",
        transform=a3.transAxes, va="top", fontsize=8, bbox=dict(fc="white", ec="gray", alpha=0.9))
a3.legend(fontsize=8, loc="lower right")
fig.suptitle("v5 128x · seed 42 · why gap grows with passes: continuous scale growth, not per-pass writes", fontsize=12)
fig.tight_layout()
out = ROOT / "docs/figs/theory/fig_v5_epoch_dynamics_decomposition.png"
fig.savefig(out, dpi=140); print("saved", out)
print("jumps", np.round(jumps, 3), "drifts", np.round(drifts, 3),
      f"jump share {100*sum(jumps)/(sum(jumps)+sum(drifts)):.0f}%")
print(f"20ep linear fit e5-e20 slope {a:.3f} R2 {r2:.4f}")
