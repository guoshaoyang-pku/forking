# Dataset-size reconciliation, 2026-09-25

Read-only follow-up to the corrected-tokenizer and freeze-gap audits. User asked why frozen pretraining and continued-LM results differ and how large the datasets are. No training was launched.

## Token budgets

Use predicted target tokens for training dose: batch 72 × sequence 2048 = 147456 targets/update. Stored packed rows contain 2049 IDs, so stored-token counts are slightly larger than loss-target counts. A training pool's token count is not a count of deduplicated text.

| Experiment | Batches per pool pass | Target tokens per pool | Training dose |
|---|---:|---:|---:|
| Pretraining freeze-after-pass-1 | 337 | 49,692,672 | 1000 updates, 147,456,000 targets, about 2.967 passes |
| Pretraining freeze-after-pass-2 long control | 337 | 49,692,672 | 3370 updates, 496,926,720 targets, 10 passes |
| Base checkpoints used downstream | 6740 | 993,853,440 | 20220 updates, 2,981,560,320 targets, 3 passes |
| Each downstream source: ID/Magicoder/OpenR1/RLVR | 333 | 49,102,848 | 1665 updates, 245,514,240 targets, 5 pool passes |
| Actual fixed validation per evaluation | 4 | 589,824 | Reused at every evaluation |

The downstream base naming '20x' refers to a pool about twenty times the small 337-batch corpus, not twenty epochs. Remote summaries were freshly read into training_budgets.json. Both trigram and no-gram bases use the same 6740-batch/20220-step budget, with train shards 0..20 and validation shards 21..23. Downstream ID trains on shard 21 and evaluates on 22/23; these were outside base training. This alone does not establish text-level deduplication between shards.

## Actual OOD source records

The source-count script streams the same raw files in the same order, uses the correct pretrained tokenizer, and rebuilds the SHA256 of both prepared train and validation shards. All four output hashes match the actual corrected data byte for byte. It does not save new tokenized datasets or alter existing files.

| Source | Records available in downloaded files | Records contributing train text | Records contributing val text | Records spanning split |
|---|---:|---:|---:|---:|
| Magicoder | 75,197 | 66,549 | 802 | 1 |
| OpenR1 selected default ten files | 93,733 | 83,978 | 990 | 1 |
| RLVR-MATH | 7,500 | 7,125 | 375 | 0 by source-index split |

Counts are source records, not deduplicated unique tasks. Magicoder/OpenR1 counts include the partly used records at boundaries. Magicoder source record 66548 (zero-indexed) and OpenR1 record 83977 each contribute text to both train and validation because the preparation code splits an already concatenated token stream. Thus exact packed-row disjointness does not establish source-record disjointness. No new strict-split dataset was created in this audit.

The '75K'/'220k' dataset names are not the actual number of examples trained on. In particular, the selected downloaded OpenR1 configuration contains 93,733 records; the prepared training prefix uses 83,978 of them. The OOD preparation code consumes a prefix once and stops when the train/val pools are filled; it does not cycle Magicoder/OpenR1 records at preparation time. Repeated content within distinct source records was not deduplicated by this audit.

RLVR differs: its preparation script explicitly cycles the training split. A raw training-source token cycle is 5,749,963 stored tokens, versus 49,126,824 stored tokens in the prepared pool, so one pool already contains about 8.544 source-token cycles and five pool passes correspond to about 42.719 source-token cycles. These are exposure ratios, not equal repeat counts for every record. Loss includes prompt exemplars, problems, and appended answers. RLVR token accounting is archived in the sibling posttraining_fixtok_20260925 packet.

## Interpretation limits

The roughly 49M-token pool size is similar between the small pretraining freeze experiment and downstream runs. Their different gaps are therefore not explained by a simple larger-versus-smaller packed pool comparison. Differences include random initialization versus a roughly 3B-target pretrained model, table adaptation to the same replay corpus before freezing versus freezing before a new downstream pool, bigram+trigram versus trigram-only, corpus/continuation distributions, and optimizer-state reset. They are not a one-variable experiment.

A plausible explanation is that freezing after fitting the exact replay corpus retains more corpus-specific features than importing a fixed table learned elsewhere. But shared-vocabulary or n-gram overlap was not measured here, and a frozen table can still support later backbone overfitting. The large final full-pool pretraining gap (about 2.52, seeds 43/44) is verified; its full-pool value at the pass-1 freeze boundary is unavailable. The pass-2 long experiment does show growth in the same fixed-head metric after freezing.

No-gram starts downstream from a mature pretrained checkpoint rather than random initialization, then optimizes repeatedly on about 49M targets with constant backbone LR 6e-4 after warmup. All-token continued-LM training on this pool can overfit without a table. RLVR has especially high source-level repetition. These are plausible contributing conditions, not a factorial causal attribution. Fixed checkpoint diagnostics already confirm no-gram gaps: seed 42 sampled fixed gaps ID 0.367393, Magicoder 0.215804, OpenR1 0.171382, RLVR 0.303118. Magicoder/OpenR1 validation still improves overall while train improves more; gap does not imply validation deterioration in every domain.

Reproduction script: docs/plot_scripts/audit_posttraining_source_counts.py. Run on a host with the raw files, pyarrow, numpy, and tokenizer, supplying --data-root and --tokenizer. Outputs: ood_source_counts.json and training_budgets.json. Existing numerical/curve evidence remains in sibling posttraining_fixtok_20260925 and freeze_fixed_gap_20260925 packets.
