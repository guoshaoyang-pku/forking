#!/usr/bin/env python3
"""Harvest epoch-boundary readings for every v26dn run in a runs_fixed tree.

Why this exists
---------------
`train_log.jsonl` carries an `epoch` field that advances one step early on the
final epoch (e.g. 159m E3 ends with a single row tagged `epoch=4`). Reading
"last row per epoch" naively therefore misses the true endpoint. This script:

  * groups rows by the logged `epoch` field,
  * drops trailing stub groups (< STUB_ROWS rows) that only contain the
    wrap-around rows,
  * reports the last logged row of each surviving epoch, plus `summary.json`
   final values when the run has finished.

Readings are `own gap = val_loss - train_loss`.

Single-row caveat (measured 2026-10-21)
---------------------------------------
The online train loss has a step-to-step std of ~0.059 (50m r800k tls1, epoch 2:
mean 2.718, std 0.059, range 2.549-2.875 over 678 logged rows), while the fixed
val reading moves only ~+-0.002 between adjacent logged steps. A single-row
`val - train` therefore carries +-0.06 of sampling noise. Every epoch boundary is
reported both ways:

  * `*_single`: the last logged row (what the figures used before this fix),
  * `*_win`:    mean over the last ``--window`` logged rows of the epoch
                (default 60 rows = 600 steps), which is the estimator for the
                local loss level.

Same-seed replicas cannot expose this noise: identical data order reproduces the
same batch composition, so the artifact repeats exactly.

Divergence screen (added 2026-10-21)
-----------------------------------
`status=done` does not mean the arm trained stably: an OE arm can blow up late and
still write a summary. Measured 2026-10-21 across 57 v26dn arms (360-2 + ophis),
9 arms re-inflate their train loss in the final epoch (final train up to 7.45 vs
a 3.35 minimum inside the same epoch, i.e. back to the ~7.9 init loss), which
invalidates their end-of-run gap/gain readings. The screen is the ratio

    tail_drift_ratio = final_train_loss / min(train_loss over the final epoch)

Flagged when `tail_drift_ratio >= 1.15` (`tail_flagged`). Healthy arms measure
1.00-1.12 (e.g. 104m r400k s42 = 1.12, 50m r128k = 1.06); every observed blow-up
measures 1.21-2.75, and the 159m val-plunge arms also land above the threshold.

That ratio has a blind spot: an arm which enters its final epoch already blown up
has ratio ~1.0. A second, absolute screen therefore compares the arm's final train
loss with the median final train loss of the nogram arms of the *same model size*
in the harvested tree (D/N=20 fixes the data budget, so nogram is the reference
level):

    level_ratio = final_train / median(final_train of nogram arms at the same N)

Flagged when `level_ratio >= 1.10` (`level_flagged`). Measured 2026-10-21, healthy
exact arms sit at 0.88-1.02 of their nogram level and every observed blow-up at
1.11-2.59, so the two screens agree on all finished arms.

Usage
-----
    python3 docs/plot_scripts/v26dn_harvest.py [runs_fixed_dir] [--tsv]
                                              [--window 60] [--epochs 5]

Prints a human table by default, or TSV with --tsv for merging across machines.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

STUB_ROWS = 10
# final_train / min(train over final epoch) at or above this flags a late blow-up
TAIL_DRIFT_FLAG = 1.15
# final_train / median(nogram final_train at the same N) at or above this flags it
LEVEL_FLAG = 1.10
# `v26dn_<size>m_<arm>_r<rows>k_...` -> model size in M
RUN_ID = re.compile(r"^v26dn_(\d+)m_(\w+?)_r")


def epoch_boundaries(rows: list[dict]) -> list[tuple[int, dict]]:
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(int(row.get("epoch", 0)), []).append(row)
    keep = [e for e in sorted(groups) if len(groups[e]) >= STUB_ROWS]
    return [(e, groups[e][-1]) for e in keep]


def window_stats(rows: list[dict], size: int) -> dict:
    window = rows[-size:] if size > 0 else rows[-1:]
    vals = [r["val_loss"] for r in window if r.get("val_loss") is not None]
    trains = [r["train_loss"] for r in window if r.get("train_loss") is not None]
    if not vals or not trains:
        return {}
    mean_val = sum(vals) / len(vals)
    mean_train = sum(trains) / len(trains)
    return {"val": mean_val, "train": mean_train, "gap": mean_val - mean_train, "n": len(window)}


def harvest(root: Path, window: int) -> list[dict]:
    out = []
    for run_dir in sorted(root.glob("v26dn*_fixed")):
        log = run_dir / "train_log.jsonl"
        if not log.exists():
            continue
        rows = []
        with log.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            continue
        summary = None
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                summary = None
        record: dict = {
            "run_id": run_dir.name[: -len("_fixed")],
            "status": "done" if summary else "running",
            "last_step": rows[-1].get("step"),
            "last_epoch": rows[-1].get("epoch"),
            "epochs": {},
        }
        groups: dict[int, list[dict]] = {}
        for row in rows:
            groups.setdefault(int(row.get("epoch", 0)), []).append(row)
        for epoch, row in epoch_boundaries(rows):
            record["epochs"][epoch] = {
                "step": row.get("step"),
                "val": row.get("val_loss"),
                "train": row.get("train_loss"),
                "gap": (
                    None
                    if row.get("val_loss") is None or row.get("train_loss") is None
                    else row["val_loss"] - row["train_loss"]
                ),
                "elapsed_s": row.get("elapsed_s"),
            }
            record["epochs"][epoch].update(
                {f"win_{k}": v for k, v in window_stats(groups[epoch], window).items()}
            )
        if summary:
            record["final_val"] = summary.get("final_val_loss")
            record["final_train"] = summary.get("final_train_loss")
        last_epochs = epoch_boundaries(rows)
        if last_epochs and record.get("final_train") is not None:
            tail = groups[last_epochs[-1][0]]
            trains = [r["train_loss"] for r in tail if r.get("train_loss") is not None]
            if trains:
                tail_min = min(trains)
                record["tail_min_train"] = tail_min
                if tail_min > 0:
                    ratio = record["final_train"] / tail_min
                    record["tail_drift_ratio"] = ratio
                    record["tail_flagged"] = ratio >= TAIL_DRIFT_FLAG
        out.append(record)

    # absolute level screen: the nogram arms of the same model size set the
    # reference level (D/N=20 pins the data budget per model size).
    nogram_level: dict[str, list[float]] = {}
    for rec in out:
        match = RUN_ID.match(rec["run_id"])
        if match and match.group(2) == "nogram" and rec.get("final_train") is not None:
            nogram_level.setdefault(match.group(1), []).append(rec["final_train"])
    for rec in out:
        match = RUN_ID.match(rec["run_id"])
        if not match or rec.get("final_train") is None:
            continue
        refs = nogram_level.get(match.group(1))
        if not refs:
            continue
        record_level = statistics.median(refs)
        rec["nogram_level"] = record_level
        ratio = rec["final_train"] / record_level
        rec["level_ratio"] = ratio
        rec["level_flagged"] = ratio >= LEVEL_FLAG
    return out


def fmt(value: float | None, width: int = 8) -> str:
    return " " * (width - 6) + "  ---" if value is None else f"{value:>{width}.4f}"


def print_table(records: list[dict]) -> None:
    for rec in records:
        head = f"{rec['run_id']:<52} {rec['status']:<7} st={rec['last_step']}"
        flags = []
        if rec.get("tail_flagged"):
            flags.append(f"tail-drift final_train={rec['final_train']:.3f} / min={rec['tail_min_train']:.3f}"
                         f" = {rec['tail_drift_ratio']:.2f}x")
        if rec.get("level_flagged"):
            flags.append(f"level final_train={rec['final_train']:.3f} / nogram={rec['nogram_level']:.3f}"
                         f" = {rec['level_ratio']:.2f}x")
        if flags:
            head += "  UNSTABLE: " + "; ".join(flags)
        elif rec.get("final_train") is not None:
            head += (f"  stable (tail {rec.get('tail_drift_ratio', 0):.2f}x"
                     f", level {rec.get('level_ratio', 0):.2f}x)")
        print(head)
        for epoch in sorted(rec["epochs"]):
            row = rec["epochs"][epoch]
            print(
                f"    E{epoch} st={row['step']:>6} "
                f"single val={fmt(row['val'])} train={fmt(row['train'])} gap={fmt(row['gap'])}"
                f" | win val={fmt(row.get('win_val'))} train={fmt(row.get('win_train'))}"
                f" gap={fmt(row.get('win_gap'))}"
            )


def print_tsv(records: list[dict]) -> None:
    print("run_id\tstatus\tlast_step\t" + "\t".join(
        f"e{e}_{k}" for e in range(1, 7) for k in ("val", "train", "gap")
    ) + "\t" + "\t".join(
        f"e{e}_win_{k}" for e in range(1, 7) for k in ("val", "train", "gap")
    ) + "\tfinal_val\tfinal_train\ttail_min_train\ttail_drift_ratio\ttail_flagged"
        "\tnogram_level\tlevel_ratio\tlevel_flagged")
    for rec in records:
        cells = [rec["run_id"], rec["status"], str(rec["last_step"])]
        for epoch in range(1, 7):
            row = rec["epochs"].get(epoch)
            for key in ("val", "train", "gap"):
                value = None if row is None else row[key]
                cells.append("" if value is None else f"{value:.4f}")
        for epoch in range(1, 7):
            row = rec["epochs"].get(epoch)
            for key in ("win_val", "win_train", "win_gap"):
                value = None if row is None else row.get(key)
                cells.append("" if value is None else f"{value:.4f}")
        cells.append("" if rec.get("final_val") is None else f"{rec['final_val']:.4f}")
        cells.append("" if rec.get("final_train") is None else f"{rec['final_train']:.4f}")
        cells.append("" if rec.get("tail_min_train") is None else f"{rec['tail_min_train']:.4f}")
        cells.append("" if rec.get("tail_drift_ratio") is None else f"{rec['tail_drift_ratio']:.3f}")
        cells.append("1" if rec.get("tail_flagged") else "0")
        cells.append("" if rec.get("nogram_level") is None else f"{rec['nogram_level']:.4f}")
        cells.append("" if rec.get("level_ratio") is None else f"{rec['level_ratio']:.3f}")
        cells.append("1" if rec.get("level_flagged") else "0")
        print("\t".join(cells))


def main(argv: list[str]) -> int:
    argv = list(argv)
    window = 60
    if "--window" in argv:
        window = int(argv[argv.index("--window") + 1])
        del argv[argv.index("--window") : argv.index("--window") + 2]
    args = [a for a in argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path("data/runs_fixed")
    records = harvest(root, window)
    if "--tsv" in argv:
        print_tsv(records)
    else:
        print_table(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
