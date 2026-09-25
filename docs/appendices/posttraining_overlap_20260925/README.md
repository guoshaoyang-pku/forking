# Post-training overlap and freeze-timing reproduction

Registered 2026-09-25 by Codex under the user's request to reproduce post-training forking, including trainable tables. Status: done; all eight formal runs reached step 1665 with artifact QC pass. Engineering smoke was separate. Final excess-over-no-gram analysis and the supplementary memory-removal control are in [the follow-up packet](../posttraining_samecheckpoint_20260925/README.md).

This is an all-token continued-LM replay experiment from pretrained weights, not response-only SFT or online RL. Existing corrected-tokenizer trainable-table post-training already has excess gap over no-gram. This batch asks whether the effect replicates under a lower LR and whether exact pretraining-data overlap and freezing after one downstream pass change it. Negative and positive arms are both retained; no endpoint or seed is selected retrospectively.

## Protocol

- Models: original vanilla 8L/6H/768D, vocab 8192, learned absolute positions, LayerNorm, tied embeddings, dropout 0, BF16, no compile. Input trigram-only clean R=1,048,576; no-gram disables that branch. This is the registered §68 downstream architecture, a deviation from the dual-table random-init main pretraining protocol.
- Initialization: table arms use `sftbase_ep20_tri_ckpt_fixed/final_model.pt` (SHA256 `2163a06a5e4db238b2528dcad67d45533022f7a98af7d475d924a950857bf366`); no-gram uses `s1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed/final_model.pt` (`769b6716be6e9bf7b909603e39846c3c0871b67a239d5f51b00cd4cd6853fe50`). Both base models trained a 20× data pool for THREE passes. Architecture comparisons also differ in pretrained weights and initial loss.
- Optimizers: fresh AdamW backbone `(0.8,0.95)`, wd 0.1, LR 0.00015 (1/4 standard); fresh table RMSProp `(0,0.99)`, scale 128, absolute table LR 0.0192. Warmup 100 updates, 0.25× to 1×, then constant. Seed/data seed 42.
- `seen`: first 23,976 packed rows of epoch20 shard 1, all in pretraining optimization. `unseen`: all 23,976 rows of epoch20 shard 21, same domain but not used for pretraining optimization (it was a pretraining validation shard). Both validation sets are the same first 288 rows of shard 22, disjoint from both train pools and pretraining optimization. Explicit overlap is train-to-pretrain, never train-to-validation. Pool identity also changes text; this is not an isolated causal intervention on familiarity alone.
- Each pool has 49,102,848 target tokens/pass; 333 optimizer updates/pass; five passes = 1665 updates and 245,514,240 target-token exposures. Data preparation verifies source SHA256 and exact train/val row disjointness, and writes exact-length pools so uniform probes cannot sample beyond the replay prefix.
- A100 execution: device microbatch 18, accumulation 4, effective batch 72 sequences/147,456 tokens. The same 72 examples are accumulated before one optimizer update; `epoch_batches=1332` MICRO-batches = 333 updates. This is an engineering deviation from batch72 H200; numerical identity is not assumed. All new arms use the same host/software/accumulation.
- Validation: 16 microbatches = the original 288 rows. Uniform fixed-train diagnostic: 288 deterministic rows distributed across the exact replay pool, rebatched into 16 microbatches. Online loss is the mean of four microbatches BEFORE their shared update; validation is after it. Fixed gap evaluates the uniform train and fixed val with the same weights. Primary remains online gap. No frequency-index claim; this is a scalar curve experiment.
- Log every 10 optimizer updates and steps 333/334, 666/667, 999/1000, 1332/1333, 1665. Thus the exact pre-freeze and first-post-freeze measurements are present. Initial fixed probe is evaluated before optimization. Final weights and probe tokens saved.
- Four arms per pool: trainable throughout; frozen0 (before update 1); frozen1 (first 333 updates trainable, freeze before update 334); no-gram. Frozen1 preserves table, hash and optimizer history at intervention. Runs are paired protocol reruns, not bit-identical shared-checkpoint branches; pre-intervention numerical agreement is audited.

## Registered runs and acceptance

All run IDs follow `ptov_<seen|unseen>_<trainable|frozen0|frozen1|nogram>_q4_mb18_s42_fixed`. Output on a100_perm: `/data/guoshaoyang/ngram-gap-lab/provisioning/posttraining_overlap_20260925/data/runs_fixed/<run_id>/`. Planned GPU mapping: unseen trainable/frozen0/frozen1/nogram on 0/1/2/3; seen arms on 4/5/6/7. Launcher checks GPU memory before any allocation.

Acceptance: normal exit; correct 1665 updates and 5 passes; complete logged cadence; finite losses and gap arithmetic; matching probe and validation hashes across arms; frozen1 event exactly at update 334; constant table RMS after freezing; final weights and probe tokens. A gap is not inferred from process completion. Scientific criterion: positive trainable-minus-no-gram online and fixed gap at predeclared endpoint, together with raw train/val curves and changes from initialization. Validation deterioration from the best logged step is reported separately.

Engineering smoke: two optimizer updates at the same full model, sequence length, effective batch, probe and validation settings for trainable and no-gram. IDs have `_smoke2_fixed`, outputs in a separate smoke directory; not scientific results. Replication audit: changing only downstream seed 43/44 with fully loaded weights, fixed data and dropout 0 supplies no intended independent randomness. Such repeats cannot support statistical significance; genuine independent data or pretraining replications require a specified design.

Exact executable entry point (run from the prepared isolated snapshot):

```bash
/data/guoshaoyang/ngram-gap-lab/.venv/bin/python code/cluster/run_posttraining_overlap.py \
  --root /data/guoshaoyang/ngram-gap-lab/provisioning/posttraining_overlap_20260925 \
  --data-root /data/guoshaoyang/ngram-gap-lab/data \
  --out-root /data/guoshaoyang/ngram-gap-lab/provisioning/posttraining_overlap_20260925/data/runs_fixed \
  --pool unseen --arm trainable --seed 42 --gpu 0
```

`--dry-run` emits every train.py flag as structured JSON. Preparation: `code/tools/prepare_posttraining_overlap.py --data-root <data>`. Code is committed before synchronization and hashes checked. Large table checkpoint already resides on a100_perm; only the 258 MB no-gram checkpoint needs staging, within the existing task authorization. Existing training, manuscript mirror and other staged changes are preserved.
