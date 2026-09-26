#!/usr/bin/env python3
"""Check endpoint probabilities, token membership, and checkpoint coordinates.

This checks the packaged export, not checkpoint inference or corpus recounting.
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "evidence/context-examples"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    metadata = json.loads((PACKET / "context_examples_manifest.json").read_text())
    with (PACKET / "context_examples_plotted_values.csv").open() as f:
        values = list(csv.DictReader(f))
    with (PACKET / "context_examples_token_classes.csv").open() as f:
        classes = list(csv.DictReader(f))
    membership = {}
    for row in classes:
        key = row["context"], row["token"]
        require(key not in membership, "Duplicate token membership row")
        train, val = int(row["train_count"]), int(row["heldout_count"])
        require(train >= 0 and val >= 0, "Negative corpus count")
        category = "both" if train and val else "train-only" if train else "val-only" if val else "neither"
        require(row["category"] == category, f"Membership mismatch: {key}")
        membership[key] = category
    checkpoints = {(c["run_id"], c["step"]) for c in metadata["checkpoints"]}
    observed = set()
    pairs = Counter()
    for row in values:
        context, arm, token = row["context"], row["arm"], row["token_or_statistic"]
        epoch, step = int(row["epoch"]), int(row["step"])
        key = context, arm, token, epoch
        require(key not in observed, f"Duplicate endpoint: {key}")
        observed.add(key)
        require(arm in ("both", "nogram") and (epoch, step) in ((1, 337), (5, 1685)), "Unexpected endpoint")
        require(row["run_id"] == f"ep5ckpt_v5_{arm}_s42_fixed" and int(row["seed"]) == 42, "Run identity mismatch")
        require((row["run_id"], step) in checkpoints, "Missing checkpoint identity")
        probability, logp = float(row["probability"]), float(row["log_probability"])
        require(math.isfinite(logp) and 0 < probability <= 1, "Invalid probability")
        require(math.isclose(math.log(probability), logp, rel_tol=1e-11, abs_tol=1e-10), "Log probability mismatch")
        category = "rank statistic" if token == "others" else membership.get((context, token))
        require(category is not None and row["category"] == category, "Token class mismatch")
        pairs[context, arm, token] += 1
    expected = {(context, arm, token, epoch)
                for context, token in membership
                for arm in ("nogram", "both") for epoch in (1, 5)}
    expected |= {(context, arm, "others", epoch)
                 for context in {k[0] for k in membership}
                 for arm in ("nogram", "both") for epoch in (1, 5)}
    require(observed == expected and all(n == 2 for n in pairs.values()), "Incomplete endpoints")
    print(json.dumps({"status": "PASS", "endpoint_rows": len(values), "membership_rows": len(classes),
                      "contexts": len({k[0] for k in membership}), "seed": 42,
                      "scope": "Export consistency; no checkpoint inference or raw corpus recount."}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}))
        raise SystemExit(1)
