# Remote summary recovery - 2026-09-13

Only small summary.json files were copied from 360-2 using rsync include filters. No model checkpoint, data shard, or table was copied; local data symlink was left unchanged.

- Source: 360-2:/data/home/guoshaoyang/ngram-gap-lab/data/runs_fixed/
- Local audit mirror: /tmp/ngram_gap_remote_summaries/
- Files recovered: 223
- Total bytes: 721734
- Manifest: remote-summary-inventory-360-2-20260913.csv

The CSV records run_id, steps, seed, final losses/gap, source mirror path and SHA-256. These are final-summary coordinates only, not a substitute for config or curve validation.

## Confirmed terminal runs

The five M10 summaries are present: three ffqv5 freeze arms at 3370 steps, and two mixv5 arms at 1011 steps, all seed 42. Their values agree with experiment-log §45. The two causalv5m3_hash_reseed_e2/e2e3 summaries are present at 2022 steps, seed 42, and agree with §41.

## Remaining work

- Recover the other hosts' small summary/config files before numerical backfill.
- Match each run to the correct historical/current setting and training measurement contract.
- Do not infer missing runs or completed trajectories from a summary count alone.
