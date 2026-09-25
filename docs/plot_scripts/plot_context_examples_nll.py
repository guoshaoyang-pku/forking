"""Three context examples: full-softmax NLL at epochs 1 and 5.

Replot the existing fp32 CPU checkpoint exports without evaluating a model.
Every bar starts at zero; context-specific train/held-out classes are counted
from the archived valid-position audit, not inferred from token identity.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/appendices/context_examples_20260925"
OUT = ROOT / "docs/figs/context_examples_20260925"
CONTEXTS = ["petting your", "introducing your", "there is a"]
ARMS = ["nogram", "both"]
COLORS = ["#D55E00", "#0072B2", "#438F7B", "#8873A5", "#B49751", "#85929D"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs():
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for name, entry in manifest["inputs"].items():
        assert digest(SOURCE / name) == entry["sha256"], name
    source = json.loads((SOURCE / "logits.json").read_text())
    counts = json.loads((SOURCE / "context_counts.json").read_text())
    summaries = {r["arm"]: r["summary"] for r in map(
        json.loads, (SOURCE / "source_summaries.jsonl").read_text().splitlines())}
    for arm, summary in summaries.items():
        cfg = summary["config"]
        assert summary["seed"] == 42 and summary["steps"] == 1685
        assert summary["epoch_batches"] == 337 and cfg["train_shards"] == [1]
        assert cfg["lr_schedule"] == "warmup_constant"
        assert cfg["val_interval_steps"] == 10 and cfg["val_batches"] == 4
        assert cfg["nanogpt_adam_lr"] == .0006 and cfg["table_lr_scale"] == 128.
        if arm == "both":
            assert cfg["bigram_clean_table"] == cfg["trigram_clean_table"] == 2**20
            assert cfg["table_betas"] == [0., .99]
    ids = source["named_token_ids"]
    names = [t.strip() for t in source["named_tokens"]]
    assert names == ["dog", "cat", "fish", "horse", "bird"]
    classes, class_records = {}, []
    for context in CONTEXTS:
        i = next(i for i, r in enumerate(counts["contexts"]) if r["ctx"].strip() == context)
        classes[context] = []
        for name, token_id in zip(names, ids):
            train = counts["shards"]["1"][i]["counts"].get(str(token_id), 0)
            val = sum(counts["shards"][str(s)][i]["counts"].get(str(token_id), 0) for s in [2, 3, 4])
            category = "both" if train and val else "train-only" if train else "val-only" if val else "neither"
            classes[context].append(category)
            class_records.append(dict(context=context, token=name, token_id=token_id,
                                      train_count=train, heldout_count=val, category=category))
    values, records = {}, []
    max_error = 0.
    for row in source["rows"]:
        logits = np.asarray(row["logits"], dtype=float)
        assert logits.shape == (8192,) and np.isfinite(logits).all()
        centered = logits - logits.max()
        logp = centered - np.log(np.exp(centered).sum())
        p = np.exp(logp)
        eligible = np.ones(len(p), dtype=bool)
        eligible[ids] = False
        selected = np.flatnonzero(eligible)[np.argsort(p[eligible])[-20:]]
        assert set(selected.tolist()) == set(row["others_top20_ids"])
        nll = np.r_[-logp[ids], -np.log(p[selected].mean())]
        assert np.all(nll >= 0) and np.isclose(p.sum(), 1.)
        assert np.isclose(np.exp(-nll[-1]), row["others_mean_probability"], rtol=3e-6)
        for j, name in enumerate(names):
            max_error = max(max_error, abs(-nll[j] - row["named_logp"][name]))
        epoch = row["step"] // 337
        assert epoch in [1, 5] and row["pass"] == epoch
        assert row["run_id"] == summaries[row["arm"]]["run_id"]
        values[row["context"], row["arm"], epoch] = nll
        for j, (name, value) in enumerate(zip(names + ["others"], nll)):
            records.append(dict(context=row["context"], arm=row["arm"], run_id=row["run_id"],
                                seed=42, epoch=epoch, step=row["step"], token_or_statistic=name,
                                category=classes[row["context"]][j] if j < 5 else "rank statistic",
                                negative_log_probability=float(value), probability=float(np.exp(-value))))
    assert len(values) == 12 and max_error < 5e-5
    return values, classes, records, class_records, max_error


def write_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    values, classes, records, class_records, max_error = read_inputs()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": .8, "xtick.major.size": 3., "ytick.major.size": 3.})
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 8.6))
    fig.subplots_adjust(left=.09, right=.99, top=.885, bottom=.065, wspace=.18, hspace=.62)
    ytop = float(2 * np.ceil(max(v.max() for v in values.values()) / 2))
    legend = [Patch(facecolor="white", edgecolor="#555555", label="Epoch 1"),
              Patch(facecolor="#555555", edgecolor="#555555", label="Epoch 5")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.54, .963), ncol=2,
               frameon=False, fontsize=10, handlelength=1.5, columnspacing=2.)
    for col, label in enumerate(["No-gram", "N-gram (bigram + trigram)"]):
        pos = axes[0, col].get_position()
        fig.text(pos.x0 + pos.width / 2, .982, label, ha="center", va="top", fontsize=11.5)
    for row, context in enumerate(CONTEXTS):
        pos = axes[row, 0].get_position()
        fig.text(.54, pos.y1 + .015, f'Prediction after "{context} ..."',
                 ha="center", va="bottom", fontsize=11)
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            for epoch, offset in [(1, -.185), (5, .185)]:
                for j, value in enumerate(values[context, arm, epoch]):
                    color = COLORS[j]
                    fill = tuple(.78 + .22 * v for v in matplotlib.colors.to_rgb(color)) if epoch == 1 else color
                    ax.bar(j + offset, value, width=.34, bottom=0, color=fill,
                           edgecolor=color, linewidth=.65, zorder=3)
            labels = [f"{word}\n{category}" for word, category in zip(
                ["dog", "cat", "fish", "horse", "bird"], classes[context])]
            labels.append("others\n(top-20)")
            ax.set_xticks(np.arange(6), labels, fontsize=8)
            ax.set_xlim(-.65, 5.65)
            ax.set_ylim(0, ytop)
            ax.set_yticks(np.arange(0, ytop + 1, 4))
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", color="#D8DDE3", linewidth=.6, alpha=.75)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(r"$-\log p$ (nats)", fontsize=10)
            ax.text(.02, .97, f"({chr(97+2*row+col)})", transform=ax.transAxes,
                    va="top", fontsize=9, color="#333333")
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"fig_context_examples_nll.{ext}", dpi=240, bbox_inches="tight", pad_inches=.04)
    svg = OUT / "fig_context_examples_nll.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    write_csv(OUT / "plotted_values.csv", records)
    write_csv(OUT / "token_classes.csv", class_records)
    manifest = dict(source="docs/appendices/context_examples_20260925/logits.json",
                    source_sha256=digest(SOURCE / "logits.json"),
                    counts_sha256=digest(SOURCE / "context_counts.json"),
                    script=str(Path(__file__).relative_to(ROOT)), metric="negative natural log probability",
                    max_abs_logp_difference_from_export=max_error,
                    epochs=[1, 5], steps=[337, 1685], seed=42,
                    shape=[3, 2], shared_ylim=[0, ytop],
                    token_classes="valid chunk positions; training shard 1; held-out shards 2,3,4",
                    others="negative log of mean top20 probability excluding named tokens; reselected per distribution")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
