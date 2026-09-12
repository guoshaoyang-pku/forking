#!/usr/bin/env python3
"""Generate Appendix C.4 coverage table from the tracked S1 CSV files."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
OUT = SRC / "c4_cross_axis_coverage.md"

def read(name):
    with (SRC / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def val(row, key="final_gap"):
    return row.get(key) or "—"

def main():
    table, dose, epoch, fits = (read(x) for x in (
        "s1_table_size_points.csv", "s1_dose_points_128x.csv",
        "s1_epoch_length_points.csv", "s1_scaling_fits.csv"))
    branches = sorted({r.get("branch", "") for r in table if r.get("branch")})
    lines = [
        "# Appendix C.4 · Three-axis cross-table", "",
        "> Generated from tracked CSVs. This is a coverage and endpoint table; it does not invent a three-way fit.", "",
        "## Measured axis families", "",
        "| axis | rows | branch/family | step | source |", "|---|---:|---|---:|---|",
        f"| table-size | {len(table)} | {', '.join(branches) or '—'} | {sorted({r.get('step','') for r in table})} | s1_table_size_points.csv |",
        f"| dose (128×) | {len(dose)} | input | {sorted({r.get('steps','') for r in dose})} | s1_dose_points_128x.csv |",
        f"| epoch length | {len(epoch)} | multiple | {sorted({r.get('target_steps','') for r in epoch})} | s1_epoch_length_points.csv |", "",
        "## Table-size endpoint range", "", "| branch | R range | n | gap min | gap max |", "|---|---:|---:|---:|---:|",
    ]
    for branch in branches:
        subset = [r for r in table if r.get("branch") == branch]
        rs = [int(float(r["R"])) for r in subset if r.get("R")]
        gs = [float(r["final_gap"]) for r in subset if r.get("final_gap")]
        lines.append(f"| {branch} | {min(rs):,}–{max(rs):,} | {len(subset)} | {min(gs):.4f} | {max(gs):.4f} |" if gs else f"| {branch} | — | {len(subset)} | — | — |")
    lines += ["", "## Dose endpoints (128× input)", "", "| run_id | dose | final gap | source |", "|---|---:|---:|---|"]
    lines += [f"| {r.get('run_id','')} | {r.get('dose','—')} | {val(r)} | {r.get('source','')} |" for r in dose]
    lines += ["", "## Epoch-length endpoints", "", "| run_id | multiplier | target steps | final gap | source |", "|---|---:|---:|---:|---|"]
    lines += [f"| {r.get('run_id','')} | {r.get('epoch_multiplier_L4','—')} | {r.get('target_steps','—')} | {val(r)} | {r.get('source','')} |" for r in epoch]
    lines += ["", "## Identification boundary", "", "- These files provide separate table-size, dose, and epoch-length families; they do not provide a fully crossed table-size × dose × epoch design.", "- Fits below are one-axis fits. Empty cells mean unmeasured, not zero; a crossed run needs a new registration.", "", "| family | branch | model | n | slope | R² |", "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r.get('family','—')} | {r.get('branch','—')} | {r.get('model','—')} | {r.get('n','—')} | {r.get('slope','—')} | {r.get('r2','—')} |" for r in fits if r.get('slope') or r.get('model')]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(table)} table, {len(dose)} dose, {len(epoch)} epoch rows")
if __name__ == "__main__": main()
