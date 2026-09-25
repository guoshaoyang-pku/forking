# Epoch/replay protocol audit (2026-09-25)

This is a read-only audit of the length-scaling and post-training base runs. It
separates the size of the training pool from the number of passes through that
pool. It does not start training, move checkpoints, or alter remote artifacts.

The audit script is [`audit_epoch_protocol_20260925.py`](../../plot_scripts/audit_epoch_protocol_20260925.py). It
collected summaries/logs and hashes from `ophis-gpu`, `360-1`, and `360-2`,
then checked the historical epoch20 snapshot against the live artifacts. The
machine-readable outputs are `ophis-gpu.json`, `360-1.json`, `360-2.json`,
`epoch20_checks.json`, and `run_inventory.csv`.

Source conversation extracts, with source path, line, UTC timestamp and record
SHA256, are in [`source_quotes.json`](source_quotes.json). They contain the
original user requests and the specific assistant clarification the user
responded to, rather than a later handoff's paraphrase.

## What the protocol actually says

The 2026-09-14 request asked for at least 20 length points up to a 20×
training-length data pool, explicitly saying to ensure enough source data, and
then asked to inspect epoch 1, epoch 2, and epoch 3. The resulting §55 launcher
used train shards 0–20, validation shards 21–23, and an `epoch_batches` value
for each point. The training iterator takes the first `epoch_batches` batches
of the ordered pool on each pass and restarts at shard 0; `steps=3*epoch_batches`
therefore means **three passes through the selected pool**, not a single pass
through a pool three times larger.

That is a valid protocol for the question “how does the boundary gap change
with pool length at a fixed three-pass budget?” It is not a no-replay protocol.
The user’s later 2026-09-19 SFT design independently asked for a one-pool
continued-pretraining stress test repeated for five epochs, and the 20× base
was explicitly reused as a three-pass checkpoint. No source message found in
the bounded session audit imposes “20× must be one pass” on either the §55
length sweep or the §57 SFT base.

The relevant original messages are:

- 2026-09-14 15:34 CST: “应该还是老的重复消费数据导致的随着epoch增长，gap变大的图片。看一下本地是否有修复的。” This refers to the already discovered old pool-wrap bug.
- 2026-09-14 20:40 CST: “最大的倍数做到20倍的训练长度（确保源数据够用）” and “先看 1epoch和2epoch的位置的gap”.
- 2026-09-15 01:42 CST: “epoch3呢？你的图片只包含了epoch2”.
- 2026-09-19 16:55 CST: the SFT proposal allowed an existing model trained on a 20× corpus with a small gap, followed by five epochs on a smaller target corpus.
- 2026-09-19 22:06 CST: after the assistant explicitly reported 20,220 steps and three epochs for the old run and the materialized checkpoint, the user said “是的，你直接复用然后继续做SFT（其实是continue pretrain），然后测试效果。”

These records support three passes for these particular experiments; they do
not prove that no single-pass requirement existed in any other conversation.

## Live checks

All 20 `s1v5_128_ep20_tri_<label>xL4_3ep_v2_fixed` runs pass the following
checks: `steps=3*epoch_batches`; pool capacity is at least the declared epoch
length; the 10-step plus exact epoch-boundary logging grid is complete; the
train/validation shard split is 0–20 versus 21–23; seed is 42; and
`gap=val_loss-train_loss` holds to floating-point tolerance. The historical
summary and train-log SHA256 hashes match the frozen snapshot.

The fresh packed-row audit finds 6,981 available training batches and 6,740
batches required for the 20× point. It finds no duplicate between the used
training prefix and validation, and one exact duplicate row within that prefix (shard 4 row 13,439
and shard 7 row 16,516). This is a natural source duplicate, not an iterator
wraparound. The duplicate is recorded rather than silently calling the pool
fully deduplicated. The clean single-pass Figure 1 replacement uses shards
1,11–16 and has 2,326 available batches for its 2,000-batch budget; it has no
exact train/train or train/validation duplicate in the used prefix.

## Findings by family

| Family | Actual exposure | Compliance judgement |
|---|---|---|
| Old `s1v5_128_ep_tri_*` fractional/`>1×L4` batch | Shard 1 contained 337 batches, while some declared epochs were 421–674; the iterator wrapped inside the shard. | **Invalid as pool-length scaling.** The `>1×` points are replay/pass-count points and the old U-shaped interpretation was withdrawn in §40. The ≤1× nested-prefix points remain valid for their narrower scope. |
| §40 `s1v5_128_epfx_tri_{2,3,4}p0xL4_3ep` | Expanded pools, 3 passes, 670/1000/1330 batches per pass. | **Pool/exposure protocol passes**, with the documented validation-set change and single-seed/dose confound. Actual loss logs contain only three boundaries despite `val_interval_steps=10`; do not call these dense curves. |
| §55 20-point `s1v5_128_ep20_tri_*_3ep_v2` | 20 pool lengths, each three passes; 20× uses 6,740 batches/pass and 20,220 updates. | **Protocol-compliant for the registered three-pass sweep.** It must be described as “20× data pool, 3 passes,” never “20× with no replay.” |
| §57/§58 continued pretraining | The same 20× pool is consumed for 3 base passes, then each target pool for 5 passes with a fresh optimizer. | **Compliant with the 2026-09-19 SFT/continued-pretraining design.** It intentionally tests replay after a low-gap base; it is not evidence for a no-replay base. |
| §64 no-gram base | Same 20× pool and 3 passes, saved as a checkpoint for the matched post-training control. | **Compliant with its registered matched-base role.** It is a no-gram control, not a no-replay control. |
| §64 Figure 1 old `sp` arms | Intended as single-pass, but shards 60–66 were copies of train/validation shards. | **Quarantined/invalid.** The clean `spc` replacement uses independent epoch20 shards and is the valid single-pass comparison. |
| §41 single-pool 10/20-epoch runs | 337 batches/pass, 3,370 or 6,740 steps. | **Intentional replay-count experiments**, not expanded-pool experiments. Boundary-only logging supports epoch endpoints. |
| §59/60 multi-layer v/y/Engram proxy | The completed 360-2 runs use 6,740 batches/pass and 20,220 steps; the earlier ophis attempts failed from quota. | **Pool/exposure protocol passes for the completed runs; logging does not meet the requested dense cadence.** Explicit `val_steps=6740,13480,20220` overrides `val_interval=10`; only three loss records exist. Failed/partial runs are not completed evidence. |

The §57/58 OOD results above refer only to initialization/exposure semantics.
Their original tokenizer-mismatched losses remain invalid under §66, and the
corrected-tokenizer runs must be used for any downstream performance claim.
The SFT base itself has sparse loss logging too (18 points, nominal interval
337); it cannot supply a dense no-replay training curve.

## What can be inferred from the scaling sweep

The 20-point v2 batch measures raw online gap under fixed learning rate and
three passes. Longer pools also mean more optimizer steps and a different
warmup fraction; the batch has only seed 42 and no matched no-gram run at
each length. It supports a descriptive pool-length curve under that protocol,
not a pure isolated coverage effect, quantitative Good–Turing validation, or
a universal large-model extrapolation.

## The actual post-training bases

[`base_boundaries.json`](base_boundaries.json) records the live source hashes
and the two checkpoint-producing base runs (seed 42). Both consume
993,853,440 prediction tokens per pass and 2,981,560,320 over three passes.
These are token positions; the one duplicated packed row above prevents
claiming that every source row is unique.

| Completed pass | Step | Trigram base val | Trigram base gap | No-gram base val | No-gram base gap |
|---|---:|---:|---:|---:|---:|
| 1 | 6,740 | 4.165069 | 0.028087 | 2.864167 | 0.003779 |
| 2 | 13,480 | 3.548695 | 0.125962 | 2.751566 | 0.008591 |
| 3 | 20,220 | 3.336919 | 0.213983 | 2.717110 | 0.011442 |

The trigram checkpoint is `sftbase_ep20_tri_ckpt_fixed`; the no-gram checkpoint
is `s1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed`. The earlier scientific sweep's
20× endpoint is 0.220282, not the materialization rerun's 0.213983. Only final
weights are present in these base directories; epoch-1 metrics cannot be
used to reconstruct an epoch-1 checkpoint.

Trigram validation is already much worse at the first-pass boundary. Replay
is therefore not a sufficient explanation of its absolute-loss disadvantage.
Both models' validation losses improve across subsequent passes while their
online gaps grow. Smaller gap alone does not establish a better pretrained
model, and a rise in gap does not necessarily mean an increase in validation
loss. The precise cause of the first-pass optimization difference is not
isolated by this audit.

## Why the earlier statement was wrong

The earlier explanation collapsed two coordinates into one label. “20×” was read
as “a 20×-larger pool and one traversal,” while the launcher and summaries
encoded “a 20× pool, `epoch_batches=6740`, and `3*6740` optimizer steps.” The
word `replay_mix_passes=0` in the summary means the optional shuffled replay
mode was off; it does **not** mean zero ordinary epoch replay. The fixed-order
loader explicitly restarts from shard 0 after each epoch.

The corrected claim is: the base was trained on a 20× data pool for three
passes, then used for five-pass continued-pretraining stress tests. Claims
about a no-replay 20× base must be removed. Claims about the clean Figure 1
single-pass comparison can use the `spc` family only.

## Local registry repair during this audit

The previous LR-sweep backfill used an unescaped `|` in a regular expression.
It matched every line and replaced almost the entire experiment registry.
The last clean registry at commit `54994d8` was recovered; everything before
§68 was verified byte-for-byte identical, the 24 §68 rows were rebuilt from
the saved completed-run assignments, and the later final readout was
preserved. The script now uses exact row prefixes and rejects missing or
duplicate rows/header/section; repeat updates are idempotent. This repair
does not mutate training artifacts. The pre-existing staged mirror migration
was verified byte-for-byte unchanged after the audit commit was isolated.
