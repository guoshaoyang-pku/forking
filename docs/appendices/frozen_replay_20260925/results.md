# Frozen-table post-training: completed replay-intensity grid

The table is frozen before downstream update 1; only the backbone learns. All eight runs use seed 42 and stop at the registered update 1680. All four architecture comparisons are retained, with no choice of a favorable endpoint.

| Unique target tokens/pass | LR | Passes | Frozen online gap | No-gram online gap | Online excess | Fixed excess | Initial-adjusted fixed-growth excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6.19M | .0006 | 40 | 5.731846 | 5.928494 | -0.196648 | -0.274677 | -0.269245 |
| 6.19M | .00015 | 40 | 2.504793 | 3.993828 | -1.489034 | -1.359715 | -1.354283 |
| 12.39M | .0006 | 20 | 2.075438 | 2.665319 | -0.589881 | -0.529711 | -0.529526 |
| 12.39M | .00015 | 20 | 1.074913 | 1.692961 | -0.618048 | -0.605758 | -0.605574 |

0/4 tested settings have both positive online excess and positive initial-adjusted fixed-gap growth excess. This count is descriptive and is not a significance test. A large absolute frozen gap alone does not meet the requested excess-over-no-gram criterion.

## Raw endpoint losses

| Pool updates/pass | LR | Arm | Online train loss | Fixed validation loss | Fixed train loss | Fixed gap at initialization | Fixed gap at 1680 |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | base | frozen0 | 0.864777 | 6.596623 | 1.038573 | -0.009197 | 5.558050 |
| 42 | base | nogram | 0.387244 | 6.315738 | 0.483011 | -0.003765 | 5.832727 |
| 42 | q4 | frozen0 | 1.861360 | 4.366154 | 1.839764 | -0.009197 | 2.526389 |
| 42 | q4 | nogram | 0.680627 | 4.674454 | 0.788350 | -0.003765 | 3.886105 |
| 84 | base | frozen0 | 2.178613 | 4.254051 | 2.153182 | 0.006437 | 2.100869 |
| 84 | base | nogram | 1.441649 | 4.106967 | 1.476388 | 0.006621 | 2.630579 |
| 84 | q4 | frozen0 | 2.565260 | 3.640172 | 2.522186 | 0.006437 | 1.117987 |
| 84 | q4 | nogram | 1.734235 | 3.427195 | 1.703450 | 0.006621 | 1.723745 |

## What this tests

The 6.19M/12.39M-token pools are nested prefixes of the same-domain shard 21, which was held out from base-model optimization. Both use the same disjoint 288-row validation prefix of shard 22. The 1680-update budget is 247,726,080 target-token exposures for every run. Each architecture pair has identical training bytes, stream order, effective batch, LR, warmup and evaluation probes. The bases are the original input-trigram and separately pretrained no-gram, both trained on a 20× data pool for three passes. No artificial pretraining/no-replay claim is made.

Online gap is post-update fixed validation minus pre-update current training batch loss. Fixed gap uses the same parameter state on uniform 288-row training and validation probes. Initial-adjusted excess is [G_frozen(1680)−G_frozen(0)]−[G_nogram(1680)−G_nogram(0)]. The table is never trained on these downstream pools in the frozen arms.

The experiment is all-token continued LM, not response-only SFT or on-policy RL. Small-pool replay is a stress test. Results do not establish any universal absence or presence of frozen-table excess, and they do not isolate the mechanism from differing pretrained backbone quality. Changing only downstream seed cannot provide independent replication with fixed weights/data/dropout0.

## Prior freeze evidence and external practice

The older pretraining runs causalv5c_freeze_table_e1_128x{,_s43,_s44}_fixed freeze at step338 after joint training on the same replay pool. Their final online gaps at1000 are3.452124/3.425178/3.427931. All three summaries have fixed_train_probe_batches=0 and missing/NaN final_fixed_gap. Their small raw logs and hashes are archived under prior_pretraining/. They cannot supply a fixed-gap growth trajectory or be silently relabeled as post-training freeze-from-start.

Official Engram README and paper checked on2026-09-25: [repository](https://github.com/deepseek-ai/Engram), [paper](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf). The paper explicitly updates embedding parameters with Adam at5× LR in pretraining. Its static-memory terminology describes lookup architecture; it does not itself mean frozen parameters. This check did not establish a DeepSeek production post-training freeze policy. Freezing a feature encoder, freezing a memory table, and freezing the backbone for parameter-efficient fine-tuning are different protocols.

## Artifacts and acceptance

Code snapshot b744c0b; source/data hashes in code_hashes.json and data_manifest.json. Both full-model two-update smoke arms passed before formal launches. All final checkpoints and probe files remain remote. Each frozen final table must be bitwise identical to its initial table with a changed backbone; acceptance checks also cover config, cadence, epochs, finite arithmetic, probe identity and saved artifacts. No table norm proxy replaces the bitwise check.

[Full curves](../../figs/frozen_replay_20260925/fig_frozen_replay.png), [endpoint data](endpoints.csv), [paired contrasts](comparisons.csv), [raw curve data](curves.csv). PNG/PDF/SVG generated by docs/plot_scripts/plot_frozen_replay.py; small logs collected by collect_frozen_replay.py. Sources are SHA256-verified remote data/runs_fixed artifacts. The local external-SSD data symlink may be unavailable; all plot inputs are archived inside this packet.
