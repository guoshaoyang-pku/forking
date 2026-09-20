# Continued pretraining dense-cadence evidence (2026-09-20)

This appendix is a local evidence packet copied from the six completed remote
runs. It contains `summary.json`, `train_log.jsonl`, and `table_norm.jsonl` for
each run; the large `final_model.pt` files remain on `ophis-gpu` under the
recorded run directories.

All six runs start from
`sftbase_ep20_tri_ckpt_fixed/final_model.pt`, restart optimizer state, and use
a new 100-step warmup. The reported gap is `fixed validation loss - online
training-batch loss`. The logs contain steps `10,20,...,1660,1665`; step 1665
is the fifth epoch boundary (`epoch_batches=333`).

| Dataset | Table | Run ID | Final train | Final val | Final gap |
|---|---|---|---:|---:|---:|
| ID | trainable | `sft_id_tri_e5_freq10_trainable_s42` | 1.913206 | 4.376126 | +2.462920 |
| ID | frozen | `sft_id_tri_e5_freq10_frozen_s42` | 3.030482 | 3.371526 | +0.341044 |
| Magicoder OOD | trainable | `sft_ood_magicoder_e5_freq10_trainable_s42` | 0.739568 | 1.751940 | +1.012372 |
| Magicoder OOD | frozen | `sft_ood_magicoder_e5_freq10_frozen_s42` | 1.342152 | 1.554543 | +0.212391 |
| OpenR1-10shard OOD | trainable | `sft_ood_openr1_10shards_e5_freq10_trainable_s42` | 0.753883 | 1.480376 | +0.726493 |
| OpenR1-10shard OOD | frozen | `sft_ood_openr1_10shards_e5_freq10_frozen_s42` | 1.225898 | 1.324105 | +0.098207 |

The remote run directories also contain `final_model.pt` (about 3.3 GB each).
The evidence packet intentionally omits those duplicate large files.

## Figures

The canonical dense-cadence figures are split by table update policy so the
smaller frozen OOD gaps remain readable:

- `docs/figs/continue_pretrain_20260920_freq10/fig_continue_pretrain_trainable_freq10.svg`
- `docs/figs/continue_pretrain_20260920_freq10/fig_continue_pretrain_frozen_freq10.svg`

Each figure places ID, Magicoder OOD, and OpenR1-10shard OOD in three columns.
The upper row of panels shows online train loss (solid) and fixed validation
loss (dashed); the lower row shows `gap = fixed validation − online train`
with an independent y-scale for each dataset. Epoch boundaries are marked at
steps 333, 666, 999, 1332, and 1665. The generator is
`docs/plot_scripts/plot_continue_pretrain.py`; `val_effects.csv` records the
endpoint and epoch-1 deltas used in the accompanying discussion.

## Relation to earlier 20× runs

There are two earlier 20× protocols and they answer different questions:

| Protocol | Run(s) | Endpoint gap | Interpretation |
|---|---|---:|---|
| Full-pool epoch-length array, 20× corpus, three replay passes | `s1v5_128_ep20_tri_20p0xL4_3ep_v2_fixed` | `+0.220282` at step 20220 | Low-gap base checkpoint used for continued pretraining. The original scientific run retained no final weights; `sftbase_ep20_tri_ckpt_fixed` is a checkpoint-materialization rerun with final gap `+0.2140`. |
| Single-shard, 20 epochs over the same small shard | `s1v5_128_ep_tri_1xL4_20ep` | `+15.552` | Strong replay stress test; the small data pool produces much larger fork. |
| Single-shard, 20 epochs, both tables | `s1v5_128_ep1xL4_20ep_both` | `+12.674` | Same small-pool protocol with bigram + trigram. |
| Single-shard, 20 epochs, no n-gram | `s1v5_128_ep1xL4_20ep_nogram` | `+1.012` | Backbone-only replay control; slow overfitting remains but is much smaller. |

The full-pool 20× result and the single-shard 20-epoch results must therefore
not be placed on one common “20× gap” curve. The former has a large and diverse
training pool with three passes; the latter repeats one small pool twenty times.
The source boundary table is `docs/experiment-log.md` §41 and §55, with the
full-pool snapshot in `docs/appendices/epoch20_20260914/README.md`.

## Continued-pretraining effect

All six runs above start from the low-gap checkpoint, reset optimizer state,
and warm up for 100 steps. After the first complete epoch (step 330), the
trainable table reopens a large gap: final gap increases are `+2.464` (ID),
`+0.962` (Magicoder), and `+0.749` (OpenR1). Freezing the table reduces these
increases to `+0.342`, `+0.174`, and `+0.120`, respectively. On ID, train loss
falls by `1.389` from step 330 to 1665 while validation loss rises by `1.075`
for the trainable-table run; the validation improvement is therefore fully
erased and reverses. The frozen ID run has only a `+0.072` validation change
over the same interval. OOD validation still decreases with a frozen table
(`−0.733` Magicoder and `−0.427` OpenR1), whereas trainable-table validation
decreases only `−0.233` and `−0.105` despite larger train-loss reductions.

These are seed-42, five-epoch continued-LM measurements, not response-only
instruction-SFT results; they support the table-write mechanism for this
protocol but do not establish a cross-domain or multi-seed universal effect.
