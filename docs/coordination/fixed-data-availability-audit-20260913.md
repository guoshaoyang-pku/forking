# Fixed-data availability audit · 2026-09-13

## Result

The local `data` path is a dangling symlink:

```text
data -> /Volumes/Extreme SSD/Mac备份/workdir_offload/ngram-gap-lab/data
```

The target volume/path is not mounted or does not exist in the current session. Consequently, the local checkout contains **zero discoverable `data/runs_fixed/*_fixed/summary.json` files**. This prevents a truthful T1/T2 numerical backfill and prevents local regeneration of data-backed figures.

## Safe consequence

- No historical numbers were overwritten or guessed.
- No run directory was deleted, moved, or copied.
- T1/T2 remain `stalled: missing authoritative data mount`, rather than `done`.
- Once the data volume is mounted, rerun the checks against the real `_fixed` directories before editing `experiment-log.md`.

## Required recovery evidence

1. `readlink data` resolves to an existing directory.
2. `find data/runs_fixed -maxdepth 2 -name summary.json` returns the expected run set.
3. Every changed log value maps to a matching `_fixed/summary.json`, with step and seed recorded.
4. `git diff --check` and a post-edit stale-number scan pass.
