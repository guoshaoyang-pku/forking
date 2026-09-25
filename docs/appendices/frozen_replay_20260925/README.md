# Frozen-table post-training: replay-intensity grid

Registered 2026-09-25 by Codex after the user clarified that table freeze must substantially exceed no-gram. Status done: all eight runs reached1680 updates with QC pass by20:32 Beijing. All four frozen tables are bitwise identical to initialization and their backbones changed. None of the four pairs meets the frozen-excess criterion. See [final results](results.md). Previous trainable-table success does not meet this requirement.

## Scientific question and scope

Can a table held bitwise fixed from the first downstream update enable stronger replay-driven backbone overfitting than independently pretrained no-gram, when the downstream pool is smaller? The table model and no-gram have different pretrained weights; this is a matched-budget architecture comparison. No table-disabled model substitutes for the independently pretrained no-gram. No adaptation with a trainable table is allowed before the freeze in this batch.

This is a bounded exploratory grid: two nested pool sizes × two backbone learning rates × frozen/no-gram = eight runs. Within every comparison only the architecture/frozen-memory path differs; across pairs the data-budget axis is explicit and the LR axis is explicit. All endpoints and all curves are reported, including negative results. No data selection by loss, gap or table coverage. No claim of statistical significance from one pretrained seed. Positive conditions require genuinely independent downstream data or pretraining replication before generalization; changing the seed alone on fixed weights/data/dropout0 is not replication.

## Contract

Relative to §69, the scientific change is unique-data budget/replay intensity at approximately the same total update budget. The LR grid checks the standard 6e-4 and previously tested 1/4 rate 1.5e-4 separately. Each pair shares the exact training bytes, fixed validation, probes, optimizer, LR schedule and endpoint.

- Vanilla nanoGPT 8L/6H/768D, vocab8192, seq2048, learned absolute PE, LayerNorm, tied embeddings, dropout0, bf16/no compile; input trigram-only clean R=1048576 versus no-gram. Trigram-only pretrained base is a declared deviation from the dual-table main pretraining line, inherited from §§68–70.
- Bases: sftbase_ep20_tri_ckpt_fixed/final_model.pt SHA256 2163a06a5e4db238b2528dcad67d45533022f7a98af7d475d924a950857bf366; no-gram s1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed/final_model.pt SHA256 769b6716be6e9bf7b909603e39846c3c0871b67a239d5f51b00cd4cd6853fe50. Both trained a 20× pool for three passes, not no-replay bases.
- Frozen table unchanged from pretraining, frozen before downstream update1; backbone updates. Fresh AdamW(.8,.95), wd.1, LR6e-4 or1.5e-4. Table RMSProp(0,.99), scale128 recorded but table receives no updates. Warmup100 from.25× then constant. seed/data_seed42.
- Data: first3024 or6048 packed rows of pretraining-validation shard21, never optimized by the base; exact nested prefixes of §69 unseen. 42 or84 optimizer updates/pass, 6,193,152 or12,386,304 target tokens/pass. 1680 updates each gives40 or20 passes, both247,726,080 token exposures. No train/validation exact-row overlap; shared fixed288 validation rows from shard22. Full input hashes in data_manifest.json after preparation.
- A100 microbatch18 × accumulation4 =72sequences/update =147456targets/update; epoch_batches168/336 microbatches. Uniform fixed-training probe288rows across each exact pool and fixed validation288rows, initial probe before optimization. Dense10 plus each exact pass boundary and first following update. Primary online gap=post-update fixed val−pre-update current batch; fixed gap uses identical parameter state for train and val. No frequency-index claim.
- Endpoint fixed at1680, report raw train/val losses, online/fixed gap, absolute excess and initial-adjusted fixed-gap growth versus no-gram. Compare learning-rate cases without interpreting slower adaptation as universally better generalization. Full freeze verification includes final table bitwise identity and changed backbone, not only constant norms.

## Execution and acceptance

Target a100_perm GPUs0–7, subject to live idle check; exact mapping in registered_runs.json. Remote isolated root /data/guoshaoyang/ngram-gap-lab/provisioning/frozen_replay_20260925. Shared existing data root /data/guoshaoyang/ngram-gap-lab/data. Outputs <root>/data/runs_fixed/<run_id>; no old outputs overwritten. Code commit and local/remote MD5+SHA256 verification before preparation/training. No cross-cluster weight transfers. Separate two-update full-model frozen/no-gram smoke before formal batch; smoke outputs not scientific evidence.

Preparation: <python> <root>/code/tools/prepare_frozen_replay_pools.py --data-root <data-root>.

Run: /data/guoshaoyang/ngram-gap-lab/.venv/bin/python <root>/code/cluster/run_frozen_replay.py --root <root> --data-root <data-root> --out-root <root>/data/runs_fixed --pool-updates 42 --rate base --arm frozen0 --seed 42 --gpu 0. --dry-run emits exact command; --smoke uses2updates and a separate output root.

Acceptance: process success, correct config/steps/cadence/epoch counts, finite metrics and gap arithmetic, fixed train/val/probe identity within pair, constant table norms, final model and saved probe; frozen initial/final table torch.equal and backbone changed. Complete all registered settings before interpreting the grid. A null/negative result is retained and does not meet the user's requested phenomenon.

## External practice check

Official Engram README and linked paper (Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models) inspected 2026-09-25: https://github.com/deepseek-ai/Engram and https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf . The README reports pretraining and long-context training; the searchable33-page paper did not state a post-training table-freeze protocol in the searched freeze/frozen/post-training/fine-tuning terms. This does not verify any DeepSeek-V4.1 production freeze choice. Frozen feature extractors and PEFT are common in other settings, but freezing table is distinct from freezing the entire backbone.

## Interpretation audit during execution

The pretraining freeze reference was re-read from remote raw logs, not inferred from old plots: causalv5c_freeze_table_e1_128x{,_s43,_s44}_fixed freeze at step338 after joint training of table and backbone on the replay pool, and all three have fixed_train_probe_batches=0 with no final_fixed_gap measurement. Their final online gaps at1000 are3.452124/3.425178/3.427931. The raw small files and SHA256 manifest are in prior_pretraining/ and prior_pretraining_manifest.json. This is distinct from frozen-before-first-downstream-update training.

The current input route has no gates. For gated v/y/engram routes, train.py --freeze_table also freezes gate weights; a future table-only freeze experiment must specify separately whether readout/fusion parameters remain trainable. Existing multi-layer bases are on360-2 at blocks1,4 (zero-based), not blocks0,4; do not silently reinterpret their layer IDs. Two live SSH attempts returned connection closed during this audit, so those checkpoints were not used or transferred for this batch.

LoRA is a separate conventional freeze strategy: the Hugging Face PEFT guide says original pretrained weights are frozen while adapter matrices are trained (https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). This does not establish that n-gram tables are frozen in all full-parameter SFT/RL or a particular DeepSeek release. The official Engram source check and exact PDF hash are in official_engram_source_check.json.
