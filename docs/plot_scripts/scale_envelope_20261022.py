#!/usr/bin/env python3
"""Scaling panel 2026-10-22: (A) D/N envelope at fixed E=3, (B) table-size R/N scan with
DeepSeek-V4.1-Flash positions. Numbers = last-60-val window per epoch, same-host
nogram pairing, seed 42 (docs/experiment-log.md §137; harvest /tmp/hall.jsonl).
dVal = nogram - arm (mK, >0 = OE helps); gap = val - train at E3 window."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "figs", "scale_envelope_20261022")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]

# (A) D/N envelope, exact R/N=8, tls1, E=3. run_ids v26dn_{50m,104m}_exact_r{400,832}k_{d1b|d2p1b|dn10|dn5}_tls1_s42_e3_fd
DN = [5, 10, 20]
ENV = {  # model: (E1 dVal, E3 dVal, E3 gap) per D/N in DN order
    "50M": ([86.6, 55.2, 48.0], [16.5, 8.2, 33.7], [0.132, 0.099, 0.036]),
    "104M": ([59.4, 77.9, 64.4], [-12.7, 18.9, 28.1], [0.153, 0.095, 0.036]),
}
# (B) 50M, D/N=20, E=3, tls1: R per order (k) -> (E1, E3 dVal, E3 gap); diverged arm marked
RS = [(6, -40.1, -13.8, -0.030), (12, -26.6, -14.5, -0.029), (50, 7.0, 5.9, -0.020),
      (100, 19.9, 19.5, -0.011), (128, 37.1, 35.0, -0.006), (256, 53.6, 39.4, 0.015),
      (400, 48.0, 33.7, 0.036), (800, 75.6, 33.8, 0.088), (1600, 98.7, 13.9, 0.151)]
DIVERGED = [(25, -1.6)]  # r25k: E3 diverged (val 4.43)
N50 = 50e6
# V4.1-Flash (HF config): rows 384,006,168 + 384,016,682, head_dim 256 -> 196.6B table params;
# backbone ~551B total / ~16B active (estimated from config). Sept release ~150B table.
# Mapped to our per-order R/N by table-params / N ratio (ours: 4 orders x R x 768 / N).
def rn_equiv(table_params, n_params):
    return table_params / n_params / (4 * 768 / 1000)

MARK = {"Oct（N=总参 551B）": rn_equiv(196.6e9, 551e9), "Oct（N=激活 ≈16B）": rn_equiv(196.6e9, 16.4e9),
        "Sept（N=总参）": rn_equiv(150e9, 551e9), "Sept（N=激活）": rn_equiv(150e9, 16.4e9)}

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
c = {"50M": "#1f5f8b", "104M": "#a33b20"}
for m, (e1, e3, gp) in ENV.items():
    ax[0].plot(DN, e1, "o--", color=c[m], alpha=0.55, label=f"{m} E1")
    ax[0].plot(DN, e3, "o-", color=c[m], lw=2.2, label=f"{m} E3（验收）")
    ax[1].plot(DN, gp, "s-", color=c[m], lw=2.2, label=m)
for a in ax[:2]:
    a.set_xscale("log"); a.minorticks_off(); a.set_xticks(DN); a.set_xticklabels(DN); a.set_xlabel("D/N（unique token / 参数，E=3 固定）")
ax[0].axhline(0, color="#888", lw=1)
ax[0].set_ylabel("ΔVal = nogram - OE（mK，越大越好）"); ax[0].set_title("A  D/N 包络：收益")
ax[0].legend(fontsize=9)
ax[1].set_ylabel("E3 gap = val - train（越小越好）"); ax[1].set_title("A′  D/N 包络：forking gap")
x = np.log(DN); 
for m, (_, _, gp) in ENV.items():
    k, b = np.polyfit(x, np.log(gp), 1)
    xx = np.linspace(np.log(5), np.log(200), 50)
    ax[1].plot(np.exp(xx), np.exp(b + k * xx), ":", color=c[m], alpha=0.7)
    ax[1].text(30, 0.03 + (0.012 if m == '50M' else 0.0), f"{m}: gap∝(D/N)^{k:.2f}", color=c[m], fontsize=9)
ax[1].set_xlim(4, 220); ax[1].set_xticks([5, 10, 20, 50, 100, 200]); ax[1].set_xticklabels([5, 10, 20, 50, 100, 200])
ax[1].legend(fontsize=9)

rn = np.array([r for r, *_ in RS]) * 1000 / (N50 / 1000)
ax[2].plot(rn, [e for _, e, _, _ in RS], "o--", color="#2e6b3a", alpha=0.55, label="E1 ΔVal")
ax[2].plot(rn, [e for _, _, e, _ in RS], "o-", color="#2e6b3a", lw=2.2, label="E3 ΔVal（验收）")
ax2 = ax[2].twinx()
ax2.plot(rn, [g * 1000 for *_, g in RS], "s:", color="#a33b20", label="E3 gap（右轴, mK）")
ax2.set_ylabel("E3 gap（mK，越小越好）", color="#a33b20")
for r, e in DIVERGED:
    ax[2].plot(r * 1000 / (N50 / 1000), 0, "x", color="red", ms=10, mew=2.5)
    ax[2].annotate("E3 发散", (r * 1000 / (N50 / 1000), 0), (r * 1000 / (N50 / 1000), -32), color="red", fontsize=9, ha="center")
for i, (lab, v) in enumerate(MARK.items()):
    col = "#6a4c93" if "激活" in lab else "#888"
    ax[2].axvline(v, color=col, ls="--", lw=1.3)
    ax[2].text(v * 1.06, 95 - 15 * i, f"V4.1 {lab}\nR/N≈{v:.2f}", color=col, fontsize=8)
ax[2].axhline(0, color="#888", lw=1)
ax[2].set_xscale("log"); ax[2].set_xlabel("R/N（每阶表行数 ÷ 千参数；50M，D/N=20，E=3）")
ax[2].set_ylabel("ΔVal（mK，越大越好）"); ax[2].set_title("B  表大小 R：收益与 gap，含 V4.1-Flash 位置")
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right")
fig.suptitle("Scaling（seed 42，同机 nogram 配对；V4.1 位置按表参数/N 折算，N 取总参还是激活参决定落在哪一侧）", fontsize=12)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"fig_scale_envelope.{ext}"), dpi=130)
print(OUT, {k: round(v, 3) for k, v in MARK.items()})
