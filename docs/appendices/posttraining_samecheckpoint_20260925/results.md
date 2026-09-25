# Post-training excess over no-gram: final readout

All numbers below use the predeclared step 1665 (five 333-update passes), seed 42. This is all-token continued language-model training from pretrained weights. It is not response-only instruction SFT or on-policy RLVR.

The primary architecture baseline is the separately pretrained no-gram model. The table-removed control starts from the table model checkpoint with its table zeroed and frozen; it answers a supplementary memory-removal question and has a severely degraded starting function.

## Primary comparison

| Pool | Trainable online gap | No-gram online gap | Ratio | Online excess | Fixed excess | Excess fixed-gap growth |
|---|---:|---:|---:|---:|---:|---:|
| unseen | 0.786708 | 0.212958 | 3.69× | +0.573750 | +0.653798 | +0.658583 |
| seen | 1.256596 | 0.186789 | 6.73× | +1.069807 | +1.191382 | +1.003355 |

Online gap is fixed validation after the update minus the current training batch before the update. Fixed gap uses a uniform 288-row training probe and 288 fixed validation rows at identical parameters. Excess fixed-gap growth is [G_table(1665) − G_table(0)] − [G_control(1665) − G_control(0)]. Thus the large trainable effect is not explained by the initial fixed-gap difference alone.

The unseen pool is same-domain shard 21, not used for pretraining optimization (it was a pretraining validation shard). The seen pool is an exact prefix of pretraining shard 1. Each contains 23,976 rows and 49,102,848 target tokens/pass. Both use the same disjoint validation prefix of shard 22. Pool identity also changes the text, so the difference between pools does not isolate familiarity as a causal variable.

## All arms and absolute losses

| Pool | Arm | Initial val | Final online train | Final val | Final online gap | Initial fixed gap | Final fixed gap |
|---|---|---:|---:|---:|---:|---:|---:|
| unseen | trainable | 3.345949 | 2.589238 | 3.375946 | 0.786708 | -0.031200 | 0.886040 |
| unseen | frozen0 | 3.345949 | 3.078106 | 3.262732 | 0.184627 | -0.031200 | 0.176200 |
| unseen | frozen1 | 3.345949 | 2.870223 | 3.224824 | 0.354602 | -0.031200 | 0.299963 |
| unseen | nogram | 2.736904 | 2.507964 | 2.720922 | 0.212958 | -0.026416 | 0.232242 |
| unseen | disabled | 10.540319 | 3.416884 | 3.485911 | 0.069027 | 0.016906 | 0.045766 |
| seen | trainable | 3.345949 | 2.362371 | 3.618967 | 1.256596 | 0.165633 | 1.431314 |
| seen | frozen0 | 3.345949 | 2.969055 | 3.338559 | 0.369504 | 0.165633 | 0.434013 |
| seen | frozen1 | 3.345949 | 2.747741 | 3.339799 | 0.592058 | 0.165633 | 0.600510 |
| seen | nogram | 2.736904 | 2.536572 | 2.723361 | 0.186789 | -0.022395 | 0.239932 |
| seen | disabled | 10.540319 | 3.495566 | 3.486328 | -0.009238 | 0.004779 | 0.039048 |

The independently pretrained no-gram baseline has lower absolute validation loss. A larger forking effect is an overfitting result, not an overall performance benefit of the table model. The table-removed arm starts with validation loss 10.54, so it must not replace the pretrained no-gram baseline.

## Freeze interpretation

| Pool | Arm | Fixed gap at step 333 | Growth after step 333 | Excess growth after step 333 over no-gram |
|---|---|---:|---:|---:|
| unseen | trainable | 0.120674 | +0.765366 | +0.570973 |
| unseen | frozen0 | 0.015664 | +0.160536 | -0.033857 |
| unseen | frozen1 | 0.120552 | +0.179411 | -0.014982 |
| unseen | nogram | 0.037848 | +0.194394 | +0.000000 |
| seen | trainable | 0.346454 | +1.084860 | +0.886943 |
| seen | frozen0 | 0.235934 | +0.198080 | +0.000162 |
| seen | frozen1 | 0.346272 | +0.254238 | +0.056320 |
| seen | nogram | 0.042015 | +0.197917 | +0.000000 |

The large trainable growth occurs mainly after the first downstream pass. Freezing after pass 1 does not stop all gap growth, but on unseen data its later growth is slightly below no-gram; on seen data the excess later growth is only about 0.056. Freezing from the start on seen data retains an endpoint excess that is almost entirely inherited from initialization (excess total fixed-gap growth is only about 0.006). These results do not support a large backbone-only excess under the frozen-table protocol.

## Evidence scope and reproduction

There is one pretrained seed/data realization per architecture, two specified downstream pools, and no independent replication of either pool. The effects above are large measured contrasts, not a multi-seed significance claim. Changing only the downstream seed with loaded weights, fixed stream and dropout 0 does not supply independent trials. No p-value or confidence interval is inferred from correlated logged steps.

Backbone LR 0.00015, table LR ratio 128 (absolute 0.0192), warmup 100 then constant; input trigram clean R=2^20; A100 microbatch 18 × accumulation 4 preserves 147,456 target tokens/update. Both bases trained a 20× data pool for THREE passes. The result does not repair that base protocol into a no-replay experiment.

Run IDs and all contrasts: [comparisons.csv](comparisons.csv); raw endpoint losses: [endpoints.csv](endpoints.csv); full curves: [curves.csv](curves.csv). Small source logs are archived in references/ and evidence/, with SHA256 manifests; source logs are remote data/runs_fixed/<run_id>/ artifacts. The local data/ link targets an external SSD that became unavailable during collection; packet hashes were checked directly against remote originals. The collector refreshes the local canonical cache when the drive is mounted. Final checkpoints stay on a100_perm.

Figures: [primary excess](../../figs/posttraining_samecheckpoint_20260925/fig_primary_excess.png), [freeze contrasts](../../figs/posttraining_samecheckpoint_20260925/fig_freeze_excess.png), [memory removal](../../figs/posttraining_samecheckpoint_20260925/fig_memory_removal_control.png). Each is also available as PDF and SVG. Raw curves; no smoothing.

From the repository root: python3 docs/plot_scripts/collect_posttraining_samecheckpoint.py; .venv/bin/python docs/plot_scripts/plot_posttraining_excess.py. Plotting validates source hashes, available canonical-byte agreement (otherwise verified remote hashes), registered config/cadence, finite loss arithmetic, probe identities, zero removed-table norms, QC status and remote final-artifact presence.
