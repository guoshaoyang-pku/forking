#!/usr/bin/env python3
"""Build the Appendix C source crosswalk from frozen small evidence files.

This does not read remote hosts or infer provenance for missing original logs.
"""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AX = ROOT / "docs/appendices/s1_scaling_three_axis"
EPOCH = ROOT / "docs/appendices/epoch_protocol_audit_20260925"
COVERAGE = ROOT / "docs/appendices/coverage_takeover_20260925"
OUT = ROOT / "paper/supplement/controlled_setting_manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel(name, lab):
    paper = ROOT / "paper/figures" / name
    lab_file = ROOT / lab
    assert paper.is_file() and lab_file.is_file()
    assert digest(paper) == digest(lab_file), name
    return {"paper_figure": f"figures/{name}", "figure_sha256": digest(paper),
            "lab_figure": lab}


def main():
    cap_file = AX / "s1_table_size_points.csv"
    cap = rows(cap_file)
    assert len(cap) == 62
    assert all(r["branch"] in ("bigram", "trigram") and
               r["source"].endswith("_fixed") and
               r["source"].startswith("data/runs_scaling/") and
               r["step"] == "1000" and r["seed"] == "42" for r in cap)
    assert sorted([sum(r["branch"] == b for r in cap) for b in ("bigram", "trigram")]) == [31, 31]
    for r in cap:
        assert int(r["R"]) >= 1
        assert abs(float(r["gap_1000"]) -
                   (float(r["final_val_loss"]) - float(r["final_train_loss"]))) < 1e-5
    assert len({r["run_id"] for r in cap}) == len(cap)
    dose_file = AX / "s1_dose_epoch_endpoints.csv"
    dose = rows(dose_file)
    inventory_file = EPOCH / "run_inventory.csv"
    inventory = [r for r in rows(inventory_file)
                 if r["run_id"].startswith("s1v5_128_ep20_tri_")
                 and r["status"] == "summary_present"]
    by_batches = {int(r["declared_epoch_batches"]): r for r in inventory}
    assert len(dose) == len(inventory) == len(by_batches) == 20
    assert {int(r["epoch_batches"]) for r in dose} == set(by_batches)
    corpus = []
    for r in dose:
        batches = int(r["epoch_batches"])
        source = by_batches[batches]
        assert int(source["total_steps"]) == 3 * batches
        assert int(source["pool_capacity_batches"]) >= batches
        assert source["pool_smaller_than_declared_epoch"] == "False"
        corpus.append({"run_id": source["run_id"], "seed": 42,
                       "epoch_batches": batches,
                       "prediction_positions_per_pass": batches * 72 * 2048,
                       "steps": {"end_epoch_2": 2 * batches,
                                 "end_epoch_3": 3 * batches},
                       "online_gap": {"end_epoch_2": float(r["gap_epoch_2"]),
                                      "end_epoch_3": float(r["gap_epoch_3"])}})
    data_hashes = {}
    for line in (ROOT / "docs/notes/data/a100-onboarding-20260925/data.sha256").read_text().splitlines():
        h, name = line.split(maxsplit=1)
        data_hashes[name] = h
    assert len(data_hashes) == 12
    coverage_manifest = COVERAGE / "source_manifest.json"
    coverage_sources = json.loads(coverage_manifest.read_text())["sources"]
    online_run_ids = [f"fig1v5_rp_{arm}_s{seed}_fixed"
                      for arm in ("both", "nogram") for seed in (42, 43, 44)]
    for run_id in online_run_ids:
        assert any(f"/{run_id}/" in source["path"] for source in coverage_sources)
    manifest = {
        "description": "Appendix C crosswalk for existing main scaling panels and controlled measurements; generated from archived small inputs.",
        "generator": "docs/plot_scripts/build_appendix_c_manifest.py",
        "setting_authority": "agents.md section 1; deviations belong to individual runs",
        "measurement": {
            "online_gap": "fixed held-out validation CE after update minus current-batch online training CE before update, at the same logged step",
            "fixed_probe_gap": "fixed validation CE minus fixed training-probe CE at one checkpoint; separate from online gap",
            "net_gap": "ngram gap minus matched no-gram gap of the same definition and step",
            "context_frequency": "count in one unrepeated training corpus; F=0 has no train loss or defined gap",
            "reference_update": "72 sequences x 2048 prediction positions = 147456 positions",
            "reference_probe": "first four fixed batches of validation shard 2; 589824 prediction positions",
            "reference_cadence": "validation and frequency every 10 steps; diagnostic exceptions described per experiment"
        },
        "standard_packed_inputs_sha256": data_hashes,
        "standard_packed_inputs_source": "docs/notes/data/a100-onboarding-20260925/data.sha256",
        "current_identified_main_tokenizer_sha256": "f6e7dad994ed5f8bc2da99d3eba8de3e466d9a0a61b99e209c3b7fd24ef0012e",
        "tokenizer_caveat": "Current identified tokenizer file hash does not establish byte-for-byte historical linkage to the packed August shards; original preparation manifest was not found.",
        "panels": {
            "frequency": {**panel("fig03a_frequency_coverage.pdf", "docs/figs/coverage_20260925/fig03a_frequency_coverage.pdf"),
                "run_ids": online_run_ids, "seeds": [42, 43, 44], "step": 1000,
                "displayed_arms": ["both"],
                "control_role": "matched no-gram runs are archived for comparison but are not plotted as markers in the main frequency panel",
                "metric": "online gap by native corpus-frequency bin; three-seed means and sample standard deviations",
                "plotted_values": "docs/appendices/coverage_takeover_20260925/online_points.csv",
                "plotted_values_sha256": digest(COVERAGE / "online_points.csv"),
                "source_manifest": "docs/appendices/coverage_takeover_20260925/source_manifest.json",
                "source_manifest_sha256": digest(coverage_manifest),
                "fit": "affine online gap versus directly measured novel-continuation fraction; 18 native F=1..1000 bins, at least 300 tokens on train and validation side, equal-bin OLS on three-seed means; see online_fits.csv for steps 670, 1000, 1340, 2000",
                "fit_values": "docs/appendices/coverage_takeover_20260925/online_fits.csv",
                "fit_values_sha256": digest(COVERAGE / "online_fits.csv")},
            "capacity": {**panel("fig03b_capacity.pdf", "docs/figs/paper/fig03b_capacity.pdf"),
                "measurement": "online gap at step 1000; bigram-only or trigram-only; seed 42; each point records its physical table rows R",
                "display_rule": "the log-log panel omits non-positive online gaps; all 62 recorded runs remain in points",
                "points": [{"run_id": r["run_id"], "branch": r["branch"],
                            "R": int(r["R"]), "seed": 42, "step": 1000,
                            "online_gap": float(r["gap_1000"]),
                            "displayed": float(r["gap_1000"]) > 0,
                            "source": r["source"]} for r in cap],
                "plotted_values": "docs/appendices/s1_scaling_three_axis/s1_table_size_points.csv",
                "plotted_values_sha256": digest(cap_file),
                "fit": "no fitted line in the paper capacity panel",
                "caveat": "Historical _fixed runs are in data/runs_scaling, outside the agents.md data/runs_fixed namespace; recheck authority before treating these as canonical."},
            "corpus": {**panel("fig03c_corpus.pdf", "docs/figs/paper/fig03c_corpus.pdf"),
                "measurement": "online endpoint gap; trigram-only, 2^20 rows, seed 42; each selected pool is consumed for exactly three passes",
                "train_shards": "epoch20 packed shards 0-20", "validation_shards": "epoch20 packed shards 21-23",
                "points": corpus,
                "plotted_values": "docs/appendices/s1_scaling_three_axis/s1_dose_epoch_endpoints.csv",
                "plotted_values_sha256": digest(dose_file),
                "run_inventory": "docs/appendices/epoch_protocol_audit_20260925/run_inventory.csv",
                "run_inventory_sha256": digest(inventory_file),
                "fit": "no fitted line in the paper corpus-size panel",
                "caveat": "Historical pool-length sweep uses data/runs_scaling; larger pools also entail more optimizer updates and this is one seed, without a matched no-gram run at each length."}
        }
    }
    # Keep the current frequency-fit protocol when regenerating other panels.
    powerlaw = ROOT / "paper/supplement/frequency_powerlaw_manifest.json"
    if powerlaw.exists():
        evidence = json.loads(powerlaw.read_text())
        assert evidence["figure_sha256"] == manifest["panels"]["frequency"]["figure_sha256"]
        manifest["panels"]["frequency"].update(
            metric="online gap by native corpus-frequency bin; three-seed arithmetic means; no error bars",
            fit="equal-bin log-log OLS of gap versus F; 18 native F=1..1000 bins represented by geometric midpoints, at least 300 tokens per train/validation bin; step 1000 three-seed means",
            fit_values="docs/appendices/section4_close_20260926/powerlaw_fits.json",
            fit_values_sha256=digest(ROOT / "docs/appendices/section4_close_20260926/powerlaw_fits.json"),
            paper_fit_manifest="supplement/frequency_powerlaw_manifest.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
