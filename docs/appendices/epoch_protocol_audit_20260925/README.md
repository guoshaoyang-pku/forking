# Epoch/replay protocol audit (2026-09-25)

This is a read-only audit of the length-scaling and post-training base runs. It
separates the size of the training pool from the number of passes through that
pool. It does not start training, move checkpoints, or alter remote artifacts.

The audit script is [`audit_epoch_protocol_20260925.py`](../../plot_scripts/audit_epoch_protocol_20260925.py). It
collected summaries/logs and hashes from `ophis-gpu`, `360-1`, and `360-2`,
then checked the historical epoch20 snapshot against the live artifacts. The
machine-readable outputs are `ophis-gpu.json`, `360-1.json`, `360-2.json`,
`epoch20_checks.json`, and `run_inventory.csv`.

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

## Live checks

All 20 `s1v5_128_ep20_tri_<label>xL4_3ep_v2_fixed` runs pass the following
checks: `steps=3*epoch_batches`; pool capacity is at least the declared epoch
length; the 10-step plus exact epoch-boundary logging grid is complete; the
train/validation shard split is 0–20 versus 21–23; seed is 42; and
`gap=val_loss-train_loss` holds to floating-point tolerance. The historical
summary and train-log SHA256 hashes match the frozen snapshot.

The fresh packed-row audit finds 6,981 available training batches and 6,740
batches required for the 20× point. It finds no train/validation duplicate
row, and one exact duplicate row within the training pool (shard 4 row 13,439
and shard 7 row 16,516). This is a natural source duplicate, not an iterator
wraparound. The duplicate is recorded rather than silently calling the pool
fully deduplicated. The clean single-pass Figure 1 replacement uses shards
1,11–16 and has 2,326 available batches for its 2,000-batch budget; it has no
exact train/train or train/validation duplicate in the used prefix.

## Findings by family

| Family | Actual exposure | Compliance judgement |
|---|---|---|
| Old `s1v5_128_ep_tri_*` fractional/`>1×L4` batch | Shard 1 contained 337 batches, while some declared epochs were 421–674; the iterator wrapped inside the shard. | **Invalid as pool-length scaling.** The `>1×` points are replay/pass-count points and the old U-shaped interpretation was withdrawn in §40. The ≤1× nested-prefix points remain valid for their narrower scope. |
| §40 `s1v5_128_epfx_tri_{2,3,4}p0xL4_3ep` | Expanded pools, 3 passes, 670/1000/1330 batches per pass. | **Valid for fixed-three-pass pool-length comparison**, with the documented validation-set change and single-seed/dose confound. |
| §55 20-point `s1v5_128_ep20_tri_*_3ep_v2` | 20 pool lengths, each three passes; 20× uses 6,740 batches/pass and 20,220 updates. | **Protocol-compliant for the registered three-pass sweep.** It must be described as “20× data pool, 3 passes,” never “20× with no replay.” |
| §57/§58 continued pretraining | The same 20× pool is consumed for 3 base passes, then each target pool for 5 passes with a fresh optimizer. | **Compliant with the 2026-09-19 SFT/continued-pretraining design.** It intentionally tests replay after a low-gap base; it is not evidence for a no-replay base. |
| §64 no-gram base | Same 20× pool and 3 passes, saved as a checkpoint for the matched post-training control. | **Compliant with its registered matched-base role.** It is a no-gram control, not a no-replay control. |
| §64 Figure 1 old `sp` arms | Intended as single-pass, but shards 60–66 were copies of train/validation shards. | **Quarantined/invalid.** The clean `spc` replacement uses independent epoch20 shards and is the valid single-pass comparison. |

## Why the earlier statement was wrong

The error came from collapsing two coordinates into one label. “20×” was read
as “a 20×-larger pool and one traversal,” while the launcher and summaries
encoded “a 20× pool, `epoch_batches=6740`, and `3*6740` optimizer steps.” The
word `replay_mix_passes=0` in the summary means the optional shuffled replay
mode was off; it does **not** mean zero ordinary epoch replay. The fixed-order
loader explicitly restarts from shard 0 after each epoch.

The corrected claim is: the base was trained on a 20× data pool for three
passes, then used for five-pass continued-pretraining stress tests. Claims
about a no-replay 20× base must be removed. Claims about the clean Figure 1
single-pass comparison can use the `spc` family only.

