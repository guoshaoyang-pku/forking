# Candidate run protocol audit · 2026-09-13

This report uses only small remote summary.json and train_log.jsonl files recovered from the named host/path. Duplicate run IDs remain host scoped; no endpoint is promoted to the experiment log.

| host | run_id | steps | seed | final gap | fixed probe batches | probe hash | probe mode | interval | table scale | train/val shards | log rows |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---|---:|
| ophis-gpu | nglab1x_input_rho_v5_fixed | 2000 | 42 | 5.583336 | 4 | 38d1254a827759d6 | first | 10 | 2.0 | [1] / [2..10,6542] | 2000 |
| 360-2 | nglab1x_input_rho_v5_fixed | 2000 (summary; incomplete log) | 42 | not established | 4 | 38d1254a827759d6 | first | 10 | 2.0 | [1] / [2..10,6542] | 12 |
| ophis-gpu | nglab2x_input_rho_v5_fixed | 2000 | 42 | 1.248748 | 4 | 38d1254a827759d6 | first | 10 | 2.0 | [1,2] / [3..10,6542] | 2000 |
| 360-2 | nglab2x_input_rho_v5_fixed | 2000 | 42 | 1.237259 | 4 | 38d1254a827759d6 | first | 10 | 2.0 | [1,2] / [3..10,6542] | 2000 |
| 360-1 | nglab1x_nogram_long_v5_fixed | 8000 | 42 | 1.101865 | 0 | — | — | 10 | 2.0 (inert) | [1] / [2..10,6542] | 8000 |

## Qualification decisions

- T8 candidate (nglab1x_nogram_long_v5_fixed, 360-1): 8,000 steps, seed 42, complete train_log.jsonl, disjoint train/val shards, and final gap 1.101865. Usable as a host-scoped long no-gram endpoint. It has no fixed-train probe, so it cannot support a probe-based rho claim. Historical table scale 2.0 is inert because both n-gram branches are disabled.
- T9 1x candidates: both summaries explicitly record four fixed-train probe batches, the same probe hash, train_probe_mode=first, and interval 10. This proves fixed-probe evaluation was configured. The 360-2 train log is only 12 rows despite summary.steps=2000, so it is incomplete; the ophis artifact has full 2,000-step coverage. Keep both host scoped until code and data checks are complete.
- T9 2x candidates: both summaries record four fixed-train probe batches and interval 10, but final gaps differ (ophis 1.248748; 360-2 1.237259) and both configs retain historical table_lr_scale=2.0. They cannot be used as current 128x mainline evidence.

## Next validation

1. Compare code revision or md5sum for train.py, ngram_freq.py, and the launcher on each host.
2. Compare log row counts and final-step continuity with summary.steps; a short log is incomplete even when a summary exists.
3. Match probe hash, data split, frequency index, and measurement interval before any experiment-log overwrite.
4. Register host-scoped evidence first; collapse to a canonical run only after an explicit provenance decision.
5. Remote code identity is a separate gate; see `remote-code-md5-audit-20260913.md`.
