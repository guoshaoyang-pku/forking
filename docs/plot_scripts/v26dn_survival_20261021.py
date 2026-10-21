"""v26dn arm-survival figure (2026-10-21).

§133 audit concluded that the missing first-class metric was ARM SURVIVAL,
not a read-convention dispute.  This script plots, for the rho=8 / tls=1
iso-ratio rungs (D/N=20, exact tables, E=3):

  A · survival rate (arms that stay in-window / arms launched)
  B · E3 gain dVal (mK) of the surviving arms vs N, with the ~N^-0.3 shape

All numbers are same-machine paired (arm vs nogram, same seed, E3 last-60-row
window).  See docs/experiment-log.md §133 and docs/experiment-lines.md M11.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figs", "v26dn_iso_ratio_20261021")

# rho = 8, tls = 1, D/N = 20, E = 3, exact tables.  Surviving-arm E3 gain in
# mK (nogram final val - arm final val; positive = OE gain).
SURVIVAL = [
    # N(M), survived, launched, mean gain (mK) over surviving seeds, label
    (17, 0, 2, None, "17M  r272k tls0.5*"),
    (35, 2, 3, 51.6, "35M  r280k"),
    (50, 2, 2, 38.1, "50M  r400k"),
    (104, 2, 2, 32.9, "104M r832k"),
]

# Diverged siblings: (N, run_id, final val) -- printed under the bars as a
# reminder that the denominator is real arms, not hypothetical ones.
DIVERGED = [
    (17, "17m_exact_r272k_tls0.5_s42_oph_fd", 7.1528),
    (17, "17m_exact_r272k_tls0.25_s42", 3.42),
    (35, "35m_exact_r280k_tls1_s43", 5.80),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    ns = [n for n, *_ in SURVIVAL]
    rates = [s / t for _n, s, t, _g, _l in SURVIVAL]
    labels = [f"{n}M\n{s}/{t}" for n, s, t, _g, _l in SURVIVAL]
    colors = ["#c0392b" if r < 0.5 else ("#e59b2b" if r < 1.0 else "#2e8b57") for r in rates]

    xpos = range(len(ns))
    ax_a.bar(xpos, rates, color=colors, width=0.62, edgecolor="white", lw=1.2)
    for i, r in enumerate(rates):
        ax_a.text(i, r + 0.035, f"{r:.0%}", ha="center", fontsize=10.5, fontweight="bold")
    ax_a.set_xticks(list(xpos))
    ax_a.set_xticklabels(labels, fontsize=9.5)
    ax_a.set_ylim(0, 1.18)
    ax_a.set_ylabel("arms that stay in-window / arms launched")
    ax_a.set_title("A · the first real gate is arm survival\n"
                   "(rho=8, tls=1, identical recipe — only N moves)", fontsize=10.5)
    ax_a.axhline(1.0, color="#888", lw=0.8, ls=":")
    ax_a.annotate("iso-ratio lower edge:\nbelow ~35M the identical recipe\ndiverges (0/2 at 17M)",
                  xy=(0, 0.06), xytext=(0.18, 0.42), fontsize=8.6, color="#c0392b",
                  arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
    ax_a.grid(alpha=0.25, axis="y")

    surv = [(n, g) for n, _s, _t, g, _l in SURVIVAL if g is not None]
    xs = [n for n, _g in surv]
    ys = [g for _n, g in surv]
    ax_b.plot(xs, ys, marker="o", ms=8, lw=2.2, color="#1f4e79",
              label="surviving arms  (mean of seeds)")
    for n, g in surv:
        ax_b.annotate(f"{g:+.1f}", (n, g), textcoords="offset points", xytext=(0, 9),
                      ha="center", fontsize=9.5, color="#1f4e79")
    # ~N^-0.3 reference through the 35M point
    ref = [(n, ys[0] * (n / xs[0]) ** -0.3) for n in xs]
    ax_b.plot([p[0] for p in ref], [p[1] for p in ref], ls="--", lw=1.4, color="#888",
              label=r"reference  $\propto N^{-0.3}$")
    ax_b.axhline(0.0, color="#888", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([f"{n}M" for n in xs])
    ax_b.minorticks_off()
    ax_b.set_xlim(30, 125)
    ax_b.set_ylim(0, 62)
    ax_b.set_xlabel("model size N   (D = 20N, R = 8N  — all proportional)")
    ax_b.set_ylabel("E3 gain  Δval (mK, nogram − arm)")
    ax_b.set_title("B · when it survives, the same table ratio buys less\n"
                   "as N grows  (+51.6 → +38.1 → +32.9 mK)", fontsize=10.5)
    ax_b.legend(fontsize=8.8, frameon=False, loc="lower left")
    ax_b.grid(alpha=0.25)

    for ax in (ax_a, ax_b):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    note = ("diverged siblings (same recipe, same machine): "
            + "; ".join(f"{r} → final val {v:.2f}" for _n, r, v in DIVERGED)
            + "   |   17M has no surviving clean arm (the one +64 mK cell failed "
              "cross-machine replication, §132(l)7)")
    fig.text(0.5, -0.06, note, ha="center", fontsize=8.4, color="#444", wrap=True)

    fig.suptitle("v26dn iso-ratio ladder: survival is the gate, the gain then thins as N grows",
                 fontsize=12.2, y=1.03)
    fig.tight_layout()
    for ext in ("svg", "png"):
        path = os.path.join(OUT_DIR, f"fig_iso_survival.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print("wrote", path)


if __name__ == "__main__":
    main()
