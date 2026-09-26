#!/usr/bin/env python3
"""Trace every figure in paper/figures/ back to its evidence source.

For each mirrored paper figure the script looks for a same-named file under
docs/figs/ or docs/appendices/ (the canonical figure area) and, when found,
for the plot script in docs/plot_scripts/ that writes it. Output is a markdown
table; write it to scripts/FIGURE_PROVENANCE.md.

Usage:
    python3 scripts/figure_provenance.py            # print table
    python3 scripts/figure_provenance.py --write    # refresh the .md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGS = REPO_ROOT / "paper" / "figures"
SEARCH_DIRS = [REPO_ROOT / "docs" / "figs", REPO_ROOT / "docs" / "appendices"]
SCRIPT_DIR = REPO_ROOT / "docs" / "plot_scripts"
FIGURE_INDEX = REPO_ROOT / "docs" / "figure-index.json"
FIVE_EPOCH_CE_FIGURE = "fig09_crowd_ce_seen_novel.pdf"
VERIFIED_ALIASES = {
    "fig_sft_combined.pdf": (
        "docs/figs/posttraining_fixtok_20260925/fig_sft_rl_combined.pdf",
        "docs/plot_scripts/plot_posttraining_fixtok.py",
    ),
}
VERIFIED_GENERATORS = {
    "fig07_context_mask_epochmean.pdf": "docs/plot_scripts/plot_paper_fig7_ctxmask_real.py",
}


def index_by_basename() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for root in SEARCH_DIRS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                found.setdefault(path.name, []).append(path)
    return found


def index_scripts() -> dict[str, list[str]]:
    """basename -> plot scripts that mention it."""
    hits: dict[str, list[str]] = {}
    if not SCRIPT_DIR.is_dir():
        return hits
    for path in SCRIPT_DIR.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in re.findall(r"[\w.\-()]+\.(?:pdf|png|svg|jpg|jpeg)", text):
            hits.setdefault(name, []).append(str(path.relative_to(REPO_ROOT)))
    # A few compact generators compose the basename with a suffix at export.
    appendix_d = SCRIPT_DIR / "plot_appendix_d_minimal.py"
    if appendix_d.is_file():
        for stem in ("fig_app_d_interventions", "fig_app_d_thresholds", "fig_app_d_replay20"):
            for suffix in ("pdf", "png"):
                hits.setdefault(f"{stem}.{suffix}", []).append(str(appendix_d.relative_to(REPO_ROOT)))
    return hits


def index_manuscript_meta() -> dict[str, dict]:
    """basename -> {number, manuscript_section, clusters} from figure-index.json."""
    meta: dict[str, dict] = {}
    if not FIGURE_INDEX.is_file():
        return meta
    try:
        data = json.loads(FIGURE_INDEX.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return meta
    for entry in data.get("figures", []):
        path = entry.get("path")
        if path:
            meta.setdefault(Path(path).name, entry)
    return meta


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write scripts/FIGURE_PROVENANCE.md")
    args = parser.parse_args()

    if not PAPER_FIGS.is_dir():
        print(f"error: {PAPER_FIGS} not found", file=sys.stderr)
        return 1

    by_name = index_by_basename()
    scripts = index_scripts()
    meta = index_manuscript_meta()

    rows, traced = [], 0
    for fig in sorted(PAPER_FIGS.iterdir()):
        if not fig.is_file():
            continue
        sources = by_name.get(fig.name, [])
        alias = VERIFIED_ALIASES.get(fig.name)
        if alias:
            candidate = REPO_ROOT / alias[0]
            if candidate.is_file() and sha256(candidate) == sha256(fig):
                sources = [candidate]
        if sources:
            traced += 1
        source_cell = "<br>".join(
            f"`{p.relative_to(REPO_ROOT)}`" for p in sorted(sources)
        ) or "**not found**"
        script_cell = "<br>".join(
            f"`{s}`" for s in sorted(set(scripts.get(fig.name, [])))
        ) or "-"
        generator = VERIFIED_GENERATORS.get(fig.name)
        if generator:
            script_cell = f"`{generator}`"
        if alias and sources and sources == [REPO_ROOT / alias[0]]:
            script_cell = f"`{alias[1]}` (verified alias; SHA-256 match)"
        if fig.name == FIVE_EPOCH_CE_FIGURE and sources:
            # The same-named lab PDF is an older three-checkpoint figure. A
            # filename match cannot establish provenance of the five-pass PDF.
            if all(sha256(source) != sha256(fig) for source in sources):
                source_cell += "<br>**hash mismatch: older three-point asset**"
                script_cell = ("**five-point generator unresolved**; older "
                               "`docs/plot_scripts/plot_paper_fig9_crowding_row.py` "
                               "covers steps 337/674/1000 only")
        rows.append(
            f"| `{fig.name}` | {fig.stat().st_size // 1024} KB | {source_cell} | {script_cell} |"
        )

    total = len(rows)
    lines = [
        "# Paper figure provenance",
        "",
        "Generated by `scripts/figure_provenance.py`; do not hand-edit.",
        "",
        "`paper/figures/` is the figure directory of the in-repo Overleaf clone;",
        "evidence lives under `docs/figs/` (see `agents.md` P7).",
        "This table lists same-named source candidates and scripts that mention each",
        "figure. A name match alone does not verify identical content or run provenance.",
        "A `**not found**` row means the figure has no",
        "same-named source in `docs/figs/` or `docs/appendices/` -- either it was",
        "renamed, or it predates the current evidence layout and should be traced",
        "by hand before the paper is cited.",
        "",
        f"Traced: {traced}/{total} figures have a same-named source.",
        "",
        "| Paper figure | Size | Source in repo | Plot script |",
        "|---|---|---|---|",
        *rows,
        "",
    ]

    if meta:
        lines += [
            "## Curated paper-figure index",
            "",
            "Entries from `docs/figure-index.json` that also appear in `paper/figures/`.",
            "",
            "| Paper figure | Number | Manuscript section | Cluster |",
            "|---|---|---|---|",
        ]
        for fig in sorted(PAPER_FIGS.iterdir()):
            entry = meta.get(fig.name)
            if not entry:
                continue
            clusters = ", ".join(entry.get("clusters", [])) or "-"
            lines.append(
                f"| `{fig.name}` | {entry.get('number', '-')} | "
                f"{entry.get('manuscript_section', '-')} | {clusters} |"
            )
        lines.append("")

    out = "\n".join(lines)
    if args.write:
        (REPO_ROOT / "scripts" / "FIGURE_PROVENANCE.md").write_text(out, encoding="utf-8")
        print(f"wrote scripts/FIGURE_PROVENANCE.md ({traced}/{total} traced)")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
