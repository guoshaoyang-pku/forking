#!/usr/bin/env python3
"""Compact Appendix D plots from locally mirrored fixed-run evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/appendices/appendix_d_minimal"
OUT = ROOT / "docs/figs/appendices/appendix_d"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
    "axes.titlesize": 9, "legend.fontsize": 7.0, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5,
    "lines.linewidth": 1.35, "savefig.bbox": "tight"})

def curve(stem, final_step):
    rows = [json.loads(s) for s in (DATA / "curves" / f"{stem}.jsonl").read_text().splitlines() if s]
    assert rows and rows[-1]["step"] == final_step
    assert len({r["step"] for r in rows}) == len(rows)
    assert all(a["step"] < b["step"] for a, b in zip(rows, rows[1:]))
    assert all(abs(r["gap"] - (r["val_loss"] - r["train_loss"])) < 1e-5 for r in rows)
    return {int(r["step"]): r for r in rows}

def summary(stem, final_step):
    s = json.loads((DATA / f"{stem}.summary.json").read_text())
    assert s["run_id"] == f"{stem}_fixed" and s["seed"] == 42 and s["steps"] == final_step
    c = s["config"]
    assert c["data_mode"] == "fixed" and c["data_seed"] == 42
    assert c["train_shards"] == [1] and c["val_shards"] == [2, 3, 4, 5, 6, 7, 8, 9, 10, 6542]
    assert c["nanogpt_ngram_injection_position"] == "input"
    assert c["nanogpt_adam_lr"] == 0.0006 and c["table_lr_scale"] == 128.0
    assert c["table_optimizer"] == "rmsprop" and c["table_betas"] == [0.0, 0.99]
    assert c["lr_schedule"] == "warmup_constant" and c["warmup_steps"] == 100
    assert c["device_batch_size"] == 72 and c["total_batch_size"] == 147456
    assert c["val_interval_steps"] == 10 and c["val_batches"] == 4
    assert c["out_dir"].endswith(f"/{stem}_fixed")
    return s

def check_mask(stem, mode, threshold):
    s = summary(stem, 1000)
    iv = s["intervention"]
    assert iv["kind"] == f"mask_{mode}_freq" and iv["epochs_due"] == [1]
    assert len(iv["events"]) == 1 and iv["events"][0]["step"] == 338
    fm = iv["events"][0]["frequency_mask"]
    assert fm["mode"] == mode and fm["threshold"] == threshold
    assert fm["low_inclusive"] == (mode == "low")
    assert fm["novel_contexts_masked"] == (mode == "low")
    return s

def export(fig, name):
    for suffix in ("pdf", "png"):
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        print(path.relative_to(ROOT))
    plt.close(fig)

def interventions():
    arms = [
        ("causalv5c_none_128x", "Control", "#505b68", "-"),
        ("causalv5m_mask_low_le8_e1", r"Mask $F\leq8$", "#0072B2", "-"),
        ("causalv5m2_mask_high_t200_e1", r"Mask $F\geq200$", "#D55E00", "--"),
        ("causalv5c_hash_reseed_e1_128x", "Reseed once", "#009E73", "-"),
        ("causalv5c_hash_reseed_e1e2", "Reseed twice", "#009E73", ":"),
        ("causalv5c_freeze_table_e1_128x", "Freeze table", "#CC79A7", "-"),
        ("causalv5c_freeze_backbone_e1_128x", "Freeze backbone", "#E69F00", "-"),
    ]
    records = {stem: curve(stem, 1000) for stem, *_ in arms}
    for stem, *_ in arms:
        assert abs(summary(stem, 1000)["final_gap"] - records[stem][1000]["gap"]) < 1e-5
        assert sorted(records[stem]) == list(range(10, 1001, 10))
    check_mask("causalv5m_mask_low_le8_e1", "low", 8)
    check_mask("causalv5m2_mask_high_t200_e1", "high", 200)
    nog = curve("nglab1x_nogram_v5_128x_freq10", 2000)
    nog_summary = summary("nglab1x_nogram_v5_128x_freq10", 2000)
    assert not nog_summary["config"]["enable_bigram_ve"] and not nog_summary["config"]["enable_trigram_ve"]
    assert sorted(set(nog) & set(records[arms[0][0]])) == list(range(10, 1001, 10))
    fig, axs = plt.subplots(2, 2, figsize=(7.15, 5.55), sharex=True)
    metrics = [("train_loss", "Online train", "loss (nats)"),
               ("val_loss", "Fixed validation", "loss (nats)"),
               ("gap", "Raw gap", "val minus train (nats)"),
               ("net", "Net gap", "minus no-gram gap (nats)")]
    for ax, (metric, title, ylabel), letter in zip(axs.flat, metrics, "abcd"):
        for stem, label, color, style in arms:
            xs = sorted(set(records[stem]) & set(nog))
            ys = [records[stem][t]["gap"] - nog[t]["gap"] if metric == "net"
                  else records[stem][t][metric] for t in xs]
            ax.plot(xs, ys, color=color, ls=style, label=label)
        for t in (337, 674):
            ax.axvline(t, color="#a4aab2", lw=0.65, ls=":", zorder=0)
        if metric in ("gap", "net"):
            ax.axhline(0, color="#555", lw=0.6)
        ax.set(title=f"{letter}  {title}", xlabel="optimizer step", ylabel=ylabel, xlim=(0, 1000))
        ax.title.set_fontweight("bold")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=0.9)
    fig.subplots_adjust(top=0.85, bottom=0.10, left=0.105, right=0.98, hspace=0.38, wspace=0.34)
    export(fig, "fig_app_d_interventions")

def thresholds():
    points = []
    for path in DATA.glob("causalv5m2_mask_high_t*_e1.summary.json"):
        stem = path.name.removesuffix(".summary.json")
        t = int(stem.split("_t")[-1].split("_")[0])
        s = check_mask(stem, "high", t)
        points.append((t, s["final_gap"], s["final_val_loss"]))
    points.sort()
    assert [t for t, *_ in points] == [1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    low = []
    for t in (1, 2, 4, 8):
        s = check_mask(f"causalv5m_mask_low_le{t}_e1", "low", t)
        low.append((t, s["final_gap"], s["final_val_loss"]))
    high, low = np.asarray(points), np.asarray(low)
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.65))
    for ax, j, title, ylabel, letter in zip(axs, (1, 2), ("Raw gap", "Absolute validation loss"),
                                              ("gap (nats)", "validation CE (nats)"), "ab"):
        ax.plot(high[:, 0], high[:, j], "o-", color="#D55E00", ms=3.6, label=r"Mask high $F\geq t$")
        ax.plot(low[:, 0], low[:, j], "s-", color="#0072B2", ms=3.6, label=r"Mask low $F\leq t$")
        ax.set_xscale("log")
        ax.set(title=f"{letter}  {title}", xlabel="frequency threshold $t$", ylabel=ylabel, xlim=(0.8, 16000))
    axs[0].legend(frameon=False, loc="center right")
    fig.tight_layout(w_pad=1.6)
    export(fig, "fig_app_d_thresholds")

def replay():
    arms = [
        ("s1v5_128_ep_tri_1xL4_20ep", "Trigram only", "#0072B2", "-"),
        ("s1v5_128_ep1xL4_20ep_both", "Both tables", "#CC79A7", "-"),
        ("s1v5_128_ep1xL4_20ep_nogram", "No-gram", "#505b68", "--"),
    ]
    runs = {stem: curve(stem, 6740) for stem, *_ in arms}
    for stem, *_ in arms:
        assert abs(summary(stem, 6740)["final_gap"] - runs[stem][6740]["gap"]) < 1e-5
        assert sorted(runs[stem]) == [337 * epoch for epoch in range(1, 21)]
        assert "/data/runs_scaling/" in summary(stem, 6740)["config"]["out_dir"]
    assert summary(arms[0][0], 6740)["config"]["enable_trigram_ve"]
    assert not summary(arms[0][0], 6740)["config"]["enable_bigram_ve"]
    assert all(summary(arms[1][0], 6740)["config"][f"enable_{n}_ve"] for n in ("bigram", "trigram"))
    assert not any(summary(arms[2][0], 6740)["config"][f"enable_{n}_ve"] for n in ("bigram", "trigram"))
    with (DATA / "epoch20_points.csv").open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["run_id"] in runs]
    assert len(rows) == 60
    for row in rows:
        epoch, step = int(row["epoch"]), int(row["step"])
        assert step == 337 * epoch and row["seed"] == "42"
        assert row["source"].endswith(f"/{row['run_id']}_fixed")
        source = runs[row["run_id"]][step]
        assert all(abs(float(row[key]) - source[key]) < 1e-5 for key in ("train_loss", "val_loss", "gap"))
    ref = runs[arms[-1][0]]
    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.65), sharex=True)
    metrics = [("gap", "Raw gap", "gap (nats)"),
               ("net", "Net gap", "minus no-gram (nats)"),
               ("val_loss", "Absolute validation", "validation CE (nats)")]
    for ax, (metric, title, ylabel), letter in zip(axs, metrics, "abc"):
        for stem, label, color, style in arms:
            if metric == "net" and label == "No-gram":
                continue
            xs = sorted(runs[stem])
            ys = [runs[stem][t]["gap"] - ref[t]["gap"] if metric == "net"
                  else runs[stem][t][metric] for t in xs]
            ax.plot(range(1, 21), ys, marker="o", ms=2.7, color=color, ls=style, label=label)
        ax.set(title=f"{letter}  {title}", xlabel="completed replay epochs", ylabel=ylabel, xlim=(1, 20))
        ax.set_xticks([1, 5, 10, 15, 20])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(top=0.80, bottom=0.23, left=0.09, right=0.98, wspace=0.46)
    export(fig, "fig_app_d_replay20")

if __name__ == "__main__":
    interventions()
    thresholds()
    replay()
