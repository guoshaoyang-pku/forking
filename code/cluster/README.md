# Cluster launchers

## Start here for a new experiment

`run_baseline.sh` is the only generic new-experiment entry point in this
directory:

```bash
bash code/cluster/run_baseline.sh <gpu> <run_id> [steps]
```

It implements the repository contract for a non-table-size experiment:
`input` injection; clean bigram and trigram tables with
`R_bigram = R_trigram = 2^20`; backbone LR `0.0006`; RMSProp table optimizer
with `--table_betas 0.0,0.99`; table LR scale `128.0` (actual table LR `0.0768`); a 100-step linear warmup
then fixed LR (`--lr_schedule warmup_constant --warmup_steps 100`); 1000 steps
by default; online train loss and fixed validation/frequency evaluation every
10 steps.

For a table-size experiment, do not pass extra positional arguments to this
launcher. Register the run first, then create a dedicated launcher/command
which states the two alternate clean-table capacities. R must be its only
scientific variable.

Before either route, use the versioned workflows in `.agents/skills/`:

1. `ngram-gap-settings` checks the one-variable contract.
2. `ngram-gap-experiment-registration` registers the `run_id` and handoff.
3. `ngram-gap-plotting` governs the resulting figure.

## Historical launchers

Every other experiment launcher in this directory is historical or
wave-specific. It may select the deprecated multi-layer/two-hash table, a
nonstandard optimizer, epoch-anchored scheduling, or a run-ID convention that
does not satisfy the new contract. They are retained for provenance and
reproduction only; do not start a new main-line experiment from one without a
fresh settings review and a new run registration.

The `run_injpos.sh`, `run_table_opt.sh`, `run_scaling_epoch.sh`, and
`run_scaling_table.sh` scripts explicitly preserve their historical warmdown
behavior. The repository-wide default is warmup then fixed LR, so any other
historical replay must state its intended `--lr_schedule` before it is
executed.

`setup_env.sh` and the `launch_*`, `assign_gpus.py`, and `rerun_*` helpers are
environment/scheduling utilities rather than canonical experiment definitions.
