# Appendix B.3: optimization parameter selection

Finalized 2026-09-25 for the controlled vanilla nanoGPT setting. This evidence packet supports the choice of backbone LR `0.0006`, table LR multiplier `128` (absolute maximum `0.0768`), RMSProp `(0, 0.99)`, and the 100-step warmup from 0.25 to 1. It does not replace the discovery-stage configuration in B.1-B.2.

## What the evidence supports

- The original screens inspected the replay gap. These are diagnostic settings intended to fit the training corpus and expose forking within a short budget; they were not selected to minimize validation loss.
- The initial backbone screen fixes the table multiplier at 2, so the absolute table LR changes too. Its three rates are not a backbone-only intervention.
- The later `blrabs` family holds the table LR trajectory fixed at maximum `0.0768` and includes matched no-gram controls. All comparisons here read the same step 2000, despite different final budgets. The diagnostic family evaluates fixed validation every 50 steps, an explicit exception to the main every-10-step cadence.
- At step 1000, the archived table-rate selection curve peaks at approximately 2.732 gap at 128x, inside a broad high-gap range. At step 2000, the 8x-1024x rates all have large gaps; 128x is not the largest. Separate 2000-step runs reverse the very small 64x/128x ordering at step 1000. This supports a broad regime, not a universal optimum.
- Low-rate RMSProp and AdamW comparisons at 2x use seeds 42, 43, and 44. The AdamW table implementation has beta1=0 and no decay; it is essentially the same normalized-gradient update as bias-corrected RMSProp (different epsilon), so this is an implementation check rather than robustness to conventional momentum AdamW.
- Rate screens otherwise use seed 42. They are not a factorial or multi-seed hyperparameter optimization study. Raw same-step gap is fixed-validation CE minus the current batch's online training CE, in nats/token.

## Files and reproduction

- `sources/`: exact `summary.json` and `train_log.jsonl` snapshots for 54 complete runs (6920 loss records); no checkpoints or training corpora. Historical summary files may contain JSON `NaN` for unused fixed-probe fields; the generator validates only the finite online/fixed-val metrics reported here.
- `source_origins.json`: host and original path for the archived snapshots, plus SHA256. This internal file is not copied to Overleaf.
- `build_evidence.py`: validates configurations, complete budgets, final-summary agreement, finite loss values and the gap identity, and regenerates table fragments and numeric claims.
- `evidence.json`: complete parsed configurations, loss rows, and per-source hashes.
- `claims.json`: exact run/seed/step/value mapping for tables, prose readouts and three-seed mean/sample SD.
- `parameter_selection_manifest.json`: anonymous source supplement with claim mapping, configurations and hashes; personal paths removed.
- `parameter_selection_{initial,locked,scale}.tex`: complete generated tabular environments retained for audit. The four `*_rows.tex` files also retain bare row fragments, including the warmup check summarized in prose.
- `docs/plot_scripts/plot_appendix_c_parameter_selection.py`: regenerates the four-panel parameter-selection figure from this packet's claims and manifest; PDF, PNG and SVG outputs are in `docs/figs/appendix_c_parameter_selection/`.
- `overleaf_patch/`: frozen final manuscript section, B/C integration files, tables and anonymous manifest.
- `integration.json`: upstream and final Overleaf commit IDs, compiled artifact hashes and checks.

From the lab repository root:

```sh
python3 docs/appendices/parameter_selection_20260925/build_evidence.py
```

This command reads only the archived packet, not the live remote hosts. `--archive` was used during acquisition and is not needed for reproduction; it refuses to replace a frozen source with different contents. The exact selection-curve mapping is defined in `INITIAL`, `SCALE`, `EXTENDED`, `LOCKED`, `WARM` and `SUPPORT` in the generator. The 32x/64x/128x step-1000 selection points and step-2000 extension points come from separate run IDs, explicitly disclosed in the caption.

## Provenance and reconciliation

The experiment history was traced through `docs/experiment-log.md` sections 21a, 24b, 27, 30-32, 35 and 42; `docs/report/index.html`; and `docs/plot_scripts/plot_v5_beta099_gap_step1000.py`. The older `forking` checkout at `3262ddd` corroborates the historical selection mapping but is not used as an independent data source. The old `OPHIS_gap` checkout was not present. No numerical evidence in B.3 comes from the current-shell/multi-hash discovery setting.

Most snapshots came from local `data/runs_fixed`. Fourteen missing `blrabs` pairs/arms were downloaded as small JSON/JSONL files from ophis-gpu (11) and 360-2 (3); their remote SHA256 values match the archived files. One initial no-gram control was downloaded from 360-1. Source ownership is recorded per run in `source_origins.json`; do not infer hosts from the shared `/data/home/...` path. Four key cached selection snapshots were additionally compared to live remote file hashes during the audit.

The older prose table in experiment-log section 32 lists stale 64x/128x values (including 2.606 at 128x). B.3 uses the actual frozen loss logs and matching summaries, not that stale table. The standalone 128x record is 2.732055 at step 1000; the separate 2000-step run has 2.731463 at step 1000 and 5.724030 at step 2000. Exact values are in the manifests. No claim is made that those tiny differences are statistically meaningful.

Pre-fix optimizer experiments in which `table_betas` could be ignored are excluded as beta-sweep evidence. The supported decay sweep uses the corrected `optv5c` family. Optimizer semantics and warmup indexing were checked against `code/train.py` (`MixedOptimizer`, `get_lr_multiplier`).

The packet preserves available configuration and loss evidence. It does not reconstruct missing historical training-code or tokenizer-content fingerprints, and source-file hashes should not be mistaken for those fingerprints.

## Integration and verification

B.3 is appended to the existing Appendix B; the B title now mentions parameter selection. Appendix C links to B.3. No other appendix is renumbered. The final content is integrated through a separate Overleaf checkout so concurrent local edits to Sections 3/4 and Appendix E are preserved. Only the seven files in `overleaf_patch/` are included in the manuscript commit.

Validation includes evidence regeneration, byte comparisons of generated and manuscript tables, a complete `main.tex` build with resolved references/citations, and visual review of the new appendix pages. The final PDF excerpt is stored at `output/pdf/appendix_b_parameter_selection.pdf`. Whole-paper scientific readiness is outside this appendix task.
