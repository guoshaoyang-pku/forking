# Cluster launcher setting audit - 2026-09-13

Canonical new experiments use `code/cluster/run_baseline.sh` and the contract in `agents.md` §1. Other launchers are retained for provenance or separately registered variables. This table is an audit, not a command to run.

| launcher | classification |
|---|---|
| `backfill_v5_table_occupancy.sh` | utility-or-review |
| `launch_360_1.sh` | utility-or-review |
| `launch_360_1_extra.sh` | utility-or-review |
| `launch_360_2.sh` | utility-or-review |
| `launch_360_2_gpu0123.sh` | utility-or-review |
| `launch_360_2_gpu456.sh` | utility-or-review |
| `queue_v5_backbone_lr_abslock.sh` | utility-or-review |
| `rerun_one.sh` | utility-or-review |
| `run_baseline.sh` | canonical |
| `run_causal_minimal.sh` | utility-or-review |
| `run_epoch_scale_v10.sh` | utility-or-review |
| `run_epoch_short_b2.sh` | utility-or-review |
| `run_injpos.sh` | utility-or-review |
| `run_injpos_parallel.sh` | utility-or-review |
| `run_probe_v3.sh` | utility-or-review |
| `run_rerun_v2.sh` | utility-or-review |
| `run_rerun_v4.sh` | utility-or-review |
| `run_scaling_epoch.sh` | historical/variant |
| `run_scaling_epoch_full.sh` | historical/variant |
| `run_scaling_table.sh` | utility-or-review |
| `run_scaling_table_full.sh` | utility-or-review |
| `run_schedule_compare.sh` | utility-or-review |
| `run_schedule_search.sh` | utility-or-review |
| `run_shard_sweep.sh` | utility-or-review |
| `run_shard_sweep_360.sh` | utility-or-review |
| `run_shard_sweep_v2.sh` | utility-or-review |
| `run_table_opt.sh` | historical/variant |
| `run_table_opt_2x.sh` | utility-or-review |
| `run_table_sweep_minimal.sh` | utility-or-review |
| `run_train2x.sh` | utility-or-review |
| `run_v5_128x_rerun.sh` | historical/variant |
| `run_v5_backbone_lr_abslock.sh` | utility-or-review |
| `run_v5_clean.sh` | utility-or-review |
| `run_v5_main_manifest.sh` | utility-or-review |
| `run_v5_optimizer_sweep.sh` | historical/variant |
| `run_v5_s1_three_axis.sh` | utility-or-review |
| `run_v5_s1_three_axis_queue.sh` | utility-or-review |
| `run_v5_table_grid.sh` | utility-or-review |
| `run_v5_yv_scaling.sh` | utility-or-review |
| `setup_env.sh` | utility-or-review |

No launcher was edited or executed by this audit. Historical launchers require a fresh settings review and run registration; current SSOT table_lr_scale is 128.0.
