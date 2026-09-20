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
