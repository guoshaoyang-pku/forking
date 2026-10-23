#!/usr/bin/env python3
"""Recompute Appendix E table candidates from frozen compact inputs."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/appendices/coverage_takeover_20260925"


def class_totals(raw, arm, branch, kind):
    values = raw[f"{arm}|e3|val"][branch][kind]
    assert len(values) % 8 == 0
    novel = sum(values[i] for i in range(len(values)) if i % 8 < 4)
    seen = sum(values) - novel
    return novel, seen


def main():
    raw = json.loads((BASE / "inputs/perpos_decomp.json").read_text())
    both = raw["both|e3|val"]
    nogram = raw["nogram|e3|val"]
    assert both["n"] == nogram["n"] == 589248
    overall = both["nll_mean"] - nogram["nll_mean"]
    print(f"N={both['n']}, overall excess CE={overall:.10f} nats/position")
    for branch in ("bigram", "trigram"):
        bn, bs = class_totals(raw, "both", branch, "n")
        nn, ns = class_totals(raw, "nogram", branch, "n")
        assert (bn, bs) == (nn, ns) and bn + bs == both["n"]
        bln, bls = class_totals(raw, "both", branch, "nll")
        nln, nls = class_totals(raw, "nogram", branch, "nll")
        novel = (bln - nln) / both["n"]
        seen = (bls - nls) / both["n"]
        assert abs(novel + seen - overall) < 1e-10
        print(f"{branch}: novel positions={bn} ({bn/both['n']:.4%}), "
              f"novel excess={novel:.10f} ({novel/overall:.4%}); "
              f"seen excess={seen:.10f} ({seen/overall:.4%})")

    with (BASE / "online_fits.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    print("branch,step,a,b,rms,relative_rms,alternating_holdout_rms")
    for branch in ("bigram", "trigram"):
        for step in (670, 1000, 1340, 2000):
            matches = [r for r in rows if r["branch"] == branch
                       and r["step"] == str(step) and r["seed"] == "mean_3"
                       and r["weighting"] == "equal_bin"]
            assert len(matches) == 1 and matches[0]["n_bins"] == "18"
            r = matches[0]
            print(",".join([branch, str(step)] +
                           [r[k] for k in ("a", "b", "rmse", "relative_rmse",
                                          "alternating_holdout_rmse")]))


if __name__ == "__main__":
    main()
