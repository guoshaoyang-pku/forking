#!/usr/bin/env python3
"""Recompute the packaged three-seed coverage result using only Python's stdlib.

This is an independent numerical check of derived observations. It does not
retrain a model, recount a corpus, or certify the provenance of the observations.
The snapshot manifest pins the exact inputs; mismatches fail before calculation.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "results/coverage"
EDGES = (1, 2, 3, 4, 5, 6, 9, 13, 21, 31, 51, 76, 101, 151, 201, 301, 501, 751, 1001)
BINS = tuple((lo, hi - 1) for lo, hi in zip(EDGES, EDGES[1:]))
STEPS = (670, 1000, 1340, 2000)
SEEDS = {42, 43, 44}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(packet, name):
    with (packet / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def affine_fit(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    variance = sum((u - mx) ** 2 for u in x)
    require(variance > 0, "Cannot fit constant novelty fractions")
    slope = sum((u - mx) * (v - my) for u, v in zip(x, y)) / variance
    return slope, my - slope * mx


def rmse(residuals):
    return math.sqrt(statistics.mean(value ** 2 for value in residuals))


def verify(packet):
    manifest = json.loads((packet / "snapshot.json").read_text(encoding="utf-8"))
    hashes = {}
    for name in ("online_points.csv", "online_fits.csv", "corpus_bins.csv"):
        hashes[name] = hashlib.sha256((packet / name).read_bytes()).hexdigest()
        require(hashes[name] == manifest["data_hashes"][name], f"SHA-256 mismatch: {name}")

    points = read_csv(packet, "online_points.csv")
    recorded = read_csv(packet, "online_fits.csv")
    corpus = read_csv(packet, "corpus_bins.csv")
    fits = []
    for branch in ("bigram", "trigram"):
        corpus_rows = [row for row in corpus if row["branch"] == branch]
        require(len(corpus_rows) == len(BINS), f"{branch}: expected 18 corpus bins")
        corpus_by_bin = {(int(row["F_lo"]), int(row["F_hi"])): row for row in corpus_rows}
        require(set(corpus_by_bin) == set(BINS), f"{branch}: incorrect corpus-bin boundaries")
        for step in STEPS:
            label = f"{branch}/step={step}"
            groups = defaultdict(list)
            for row in points:
                if row["branch"] == branch and row["arm"] == "both" and int(row["step"]) == step:
                    groups[int(row["F_lo"]), int(row["F_hi"])].append(row)
            require(set(groups) == set(BINS), f"{label}: incorrect frequency-bin boundaries")
            x, y, deviations = [], [], []
            for bounds in BINS:
                sample = groups[bounds]
                require(len(sample) == 3 and {int(row["seed"]) for row in sample} == SEEDS,
                        f"{label}/{bounds}: expected one observation from each of seeds 42, 43, 44")
                novelty, gaps = [], []
                for row in sample:
                    require(row["run_id"] == f"fig1v5_rp_both_s{row['seed']}_fixed",
                            f"{label}/{bounds}: unexpected run family")
                    require(min(int(row["train_n"]), int(row["val_n"])) >= 300,
                            f"{label}/{bounds}: fewer than 300 train or validation tokens")
                    novelty.append(float(row["f"]))
                    gaps.append(float(row["gap"]))
                require(all(math.isfinite(value) for value in novelty + gaps),
                        f"{label}/{bounds}: non-finite observation")
                require(all(0 <= value <= 1 and abs(value - novelty[0]) <= 1e-12 for value in novelty),
                        f"{label}/{bounds}: novelty must agree across training seeds")
                require(abs(novelty[0] - float(corpus_by_bin[bounds]["probe"])) <= 1e-12,
                        f"{label}/{bounds}: novelty disagrees with corpus probe summary")
                x.append(novelty[0])
                y.append(statistics.mean(gaps))
                deviations.append(statistics.stdev(gaps))

            a, b = affine_fit(x, y)
            residuals = [v - a * u - b for u, v in zip(x, y)]
            error = rmse(residuals)
            origin_slope = sum(u * v for u, v in zip(x, y)) / sum(u * u for u in x)
            holdout_a, holdout_b = affine_fit(x[::2], y[::2])
            mean_y = statistics.mean(y)
            result = dict(
                branch=branch, step=step, n_bins=len(BINS), a=a, b=b, rmse=error,
                relative_rmse=error / statistics.mean(abs(v) for v in y),
                r2=1 - sum(value ** 2 for value in residuals) / sum((v - mean_y) ** 2 for v in y),
                proportional_rmse=rmse([v - origin_slope * u for u, v in zip(x, y)]),
                alternating_holdout_rmse=rmse([v - holdout_a * u - holdout_b for u, v in zip(x[1::2], y[1::2])]),
                max_seed_sd=max(deviations),
            )
            expected = [row for row in recorded if row["branch"] == branch and int(row["step"]) == step
                        and row["seed"] == "mean_3" and row["weighting"] == "equal_bin"]
            require(len(expected) == 1, f"{label}: expected exactly one recorded mean-three fit")
            for key, value in result.items():
                if key in ("branch", "step"):
                    continue
                require(math.isclose(value, float(expected[0][key]), rel_tol=1e-9, abs_tol=1e-10),
                        f"{label}: {key} differs from the recorded result")
            fits.append(result)
    return dict(
        status="PASS", hashed_inputs=hashes, observations_read=len(points),
        observations_in_fits=8 * len(BINS) * len(SEEDS), recomputed_affine_fits=len(fits),
        seeds=sorted(SEEDS), fits=fits,
        scope="Derived-observation numerical reproduction; no training, raw-corpus recount, or causal validation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET,
                        help="Directory containing snapshot.json and the three coverage CSV files")
    args = parser.parse_args()
    try:
        result = verify(args.packet)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, csv.Error) as error:
        print(json.dumps(dict(status="FAIL", error=str(error)), indent=2))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
