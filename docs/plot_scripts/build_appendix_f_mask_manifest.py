#!/usr/bin/env python3
"""Record plotted values and local inputs for Appendix F masking panels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from plot_paper_fig7_ctxmask_real import epoch_means, load_ctx

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/appendices/context_subspace_mask_20260921"
RUNS = BASE / "runs_scaling"
SPLIT = BASE / "context_kl/kl_restricted_split_e1_l4.json"
OUTPUT = BASE / "appendix_f_mask_plotted_values.json"


def source(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    prefix = "s1v5_128_marm_both_ctxmask_"
    ids = {kind: f"{prefix}{kind}_l4_fixed" for kind in ("ctrl", "e1", "e2")}
    rows = {kind: load_ctx(RUNS, run_id.removesuffix("_fixed"))
            for kind, run_id in ids.items()}
    sources = []
    for run_id in ids.values():
        for name in ("context_gap.jsonl", "summary.json"):
            sources.append(source(RUNS / run_id / name))
    sources.append(source(SPLIT))

    def means(kind: str, cohort: str) -> dict:
        values = epoch_means(rows[kind], f"tr_online_{cohort}",
                             f"val_mean_{cohort}", f"val_share_{cohort}")
        return {str(epoch): value for epoch, value in sorted(values.items())}

    pairs = json.loads(SPLIT.read_text())["pairs"]
    seen_mass = {}
    for arm in ("both", "nogram"):
        for cohort, name in (("A", "masked"), ("B", "unmasked")):
            values = [pairs[f"{arm}_e2"]["sets"][cohort]["z_seen_ref_mean"]]
            values.extend(pairs[f"{arm}_e{epoch}"]["sets"][cohort]
                          ["z_seen_new_mean"] for epoch in (2, 3, 4))
            seen_mass[f"{arm}_{name}"] = dict(zip(("1", "2", "3", "4"), values))

    figures = [source(ROOT / "docs/figs/paper" / name) for name in
               ("fig07_context_mask_epochmean.pdf", "fig07b_seen_mass.pdf")]
    manifest = {
        "scope": "Appendix F four-epoch type-mode masking panels, seed 42",
        "gap_panel": {
            "figure": figures[0],
            "script": "docs/plot_scripts/plot_paper_fig7_ctxmask_real.py:fig7_epochmean",
            "source_runs": ids,
            "metric": "mean over eligible logged windows within each epoch of val_share_S * (val_mean_S - tr_online_S)",
            "window_filter": "tr_online_n_A >= 20000, including for cohort B",
            "epoch_steps": 337,
            "values": {
                "normal_A": means("ctrl", "A"),
                "unmasked_B": means("ctrl", "B"),
                "masked_from_epoch_2_A": means("e1", "A"),
                "masked_from_epoch_3_A": means("e2", "A"),
            },
        },
        "seen_mass_panel": {
            "figure": figures[1],
            "script": "docs/plot_scripts/plot_paper_fig7_ctxmask_real.py:fig7b_seen_mass",
            "source_json": source(SPLIT),
            "source_runs": [
                "s1v5_128_marm_both_ctxmask_e1_l4_fixed",
                "s1v5_128_marm_nogram_ctxmask_e1_l4_fixed",
            ],
            "epoch_steps": [337, 674, 1011, 1348],
            "cohort_validation_positions": {"A": pairs["both_e2"]["sets"]["A"]["n"],
                                            "B": pairs["both_e2"]["sets"]["B"]["n"]},
            "metric": "mean full-vocabulary probability mass on each context's train-seen continuations at fixed validation positions",
            "values": seen_mass,
        },
        "archived_local_inputs": sources,
        "limitations": [
            "The local archive contains the both-table gap JSONL and summaries, but only the derived split-evaluation JSON for the no-gram mass curve.",
            "These four-epoch runs are distinct from the older collaborator PDF-extracted masking curves and the one-epoch ctxloss audit.",
        ],
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
