# Article-phase handoff task cards · 2026-09-13

These cards are ready to assign to `shanbin` and `haoran`. They are documentation/review tasks and do not authorize GPU use, large-file transfer, deletion, branch changes, or push.

## Card A · shanbin · evidence and appendix closure

**Question**: Can the current fixed evidence support the article tables and Appendix C–H without claiming unmeasured interactions?

**Inputs**

- `agents.md` (setting SSOT and claim ceiling)
- `docs/experiment-lines.md`
- `docs/experiment-log.md` §§35–54
- `docs/claims-ledger.md`
- `docs/appendices/s1_scaling_three_axis/c4_cross_axis_coverage.md`
- `docs/appendices/run_id_index.md`

**Deliverables**

1. A line-by-line list of article tables/figures and their source run/artifact.
2. A list of numbers that lack `_fixed` or lack step/seed metadata.
3. Proposed wording for C.4, F.2 and G.2.1 that preserves the observational/proxy ceiling.
4. Review comments in a standalone file; do not edit `docs/report/index.html` directly.

**Acceptance**

- Every retained number points to a source path and run_id/step/seed.
- No full three-axis interaction is implied by separate one-axis fits.
- Novel context is described with hit-count language, not continuation-level language.

## Card B · haoran · prose and reproducibility closure

**Question**: Is the article readable and reproducible from a clean checkout plus declared artifact paths?

**Inputs**

- `README.md`, `docs/resource-index.md`, `docs/coordination/README.md`
- `docs/report/index.html` (read-only local copy)
- `docs/plot_scripts/README.md` and figure index
- `docs/coordination/untracked-resource-audit-20260913.md`

**Deliverables**

1. A terminology/symbol consistency list (gap, train/val timing, f/hit-count, branch names, R).
2. Broken-link and stale-path list for README/resource/report/appendix links.
3. A prose patch proposal; changes to the blog source remain a separate reviewable commit.

**Acceptance**

- Links resolve within the checkout or are explicitly labelled remote-only.
- Historical/current settings are visibly separated.
- No claim is strengthened beyond `claims-ledger.md`.

## Shared return format

```text
owner:
status: planned | running | done | stalled
input commit:
files reviewed:
findings:
proposed patch paths:
validation commands:
claim ceiling / unresolved decision:
```
