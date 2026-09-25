# Correct-tokenizer post-training LR sweep

Status: 24/24 new runs passed artifact and metric QC. Seed 42 only. Original-LR comparison uses the matching seed-42 runs.

## Completed-sweep findings

All twelve source/arm combinations have smaller online and exact-probe fixed gaps at each lower LR. Lower rates reduce the gap at this fixed five-pass budget; this does not establish its converged limit. Trainable-table models retain positive excess gap over their matched no-gram controls at both lower rates in every source.

ID: all three arms improve final validation loss as LR decreases. The trainable-table online gap falls from2.462920 to0.785483 to0.331233. Frozen/no-gram remain close at each LR. The large original ID gap is therefore strongly sensitive to the downstream optimization rate.

Magicoder/OpenR1: trainable-table validation is best among these three tested rates at1/4 LR, while frozen-table/no-gram validation worsens at lower rates despite smaller gaps. Lower gap alone cannot be described as improved generalization; slower adaptation at the fixed budget contributes.

RLVR-MATH: reductions are weaker than for the other sources. At1/10 LR the frozen/no-gram online gaps are0.237961/0.235625 (difference0.002336); fixed gaps0.233927/0.231528 (difference0.002399). The trainable-table gap remains0.375805. Thus the original frozen-table excess over no-gram is not stable to this LR change. The large gaps within both frozen/no-gram arms remain consistent with strong replay of the small static pool, but this sweep does not causally isolate repetition or a particular mechanism.

Domain audit: ID unigram JS divergence is0.000081 versus0.218077/0.309210/0.413521 for the three OOD pools. This supports a real marginal distribution shift, not the claim that the pretraining corpus is defective or that shift causes the gap ranking. Almost all vocabulary is previously seen, and all table addresses are visited during pretraining; exact n-gram identity, address coverage, and conditional continuation distributions are different quantities.

Final QC: all24 runs reach1665 steps with finite dense curves and saved final/probe artifacts. The saved probe tensors reproduce the logged source hashes. All8 frozen final tables are bitwise identical to their initial table, while their backbone changes. The12 historical validation endpoints reproduce exactly under the uniform-probe evaluator. See final_artifact_audit.json and baseline_uniform4.json.

## Protocol

Four sources (ID, Magicoder, OpenR1, static RLVR-MATH) × three arms (trainable trigram table, frozen table, no-gram) × LR 1/4 or 1/10. Backbone LR 0.00015 / 0.00006; table/backbone ratio 128, giving trainable-table LR 0.0192 / 0.00768. Same pretrained checkpoints, correct-tokenizer packed data, fresh optimizer, 100-step warmup then constant, batch 72 × 2048, 1665 optimizer steps = five passes over 333 batches. This is continued all-token LM training, not response-only SFT or on-policy RL.

Each downstream pass has 49,102,848 target tokens; total 245,514,240. Validation has 589,824 target tokens. RLVR packing repeats the small source internally, so five packed passes correspond to approximately 42.7 source cycles. The known one-record train/val boundary crossing in Magicoder/OpenR1 is retained for matched historical comparison; no strict example-disjoint claim is made.

Primary online gap = fixed validation loss after update minus the current 72-example optimization batch loss before update. Diagnostic fixed gap = same-checkpoint validation loss minus 288 uniformly spaced training chunks. The probe is evaluated every 10 steps and logged boundary events. No interpolation or smoothing. An old-LR fixed endpoint is used only after evaluating the exact same uniform probe and checking its hash; the earlier 33-batch audit is not substituted.

## Endpoint results (step 1665, nats/token)

| Source | Arm | LR fraction | Train | Validation | Online gap | Fixed gap |
|---|---|---:|---:|---:|---:|---:|
| ID | trainable | 1 | 1.9132 | 4.3761 | 2.4629 | 2.7786 |
| ID | trainable | 1/4 | 2.5896 | 3.3751 | 0.7855 | 0.8815 |
| ID | trainable | 1/10 | 2.8803 | 3.2115 | 0.3312 | 0.3544 |
| ID | frozen | 1 | 3.0305 | 3.3715 | 0.3410 | 0.3645 |
| ID | frozen | 1/4 | 3.0783 | 3.2635 | 0.1853 | 0.1760 |
| ID | frozen | 1/10 | 3.1469 | 3.2462 | 0.0993 | 0.0707 |
| ID | nogram | 1 | 2.5476 | 2.8636 | 0.3159 | 0.3557 |
| ID | nogram | 1/4 | 2.5063 | 2.7213 | 0.2150 | 0.2344 |
| ID | nogram | 1/10 | 2.5727 | 2.6809 | 0.1082 | 0.1065 |
| Magicoder | trainable | 1 | 0.6187 | 1.5463 | 0.9276 | 1.0293 |
| Magicoder | trainable | 1/4 | 0.9861 | 1.3379 | 0.3518 | 0.4412 |
| Magicoder | trainable | 1/10 | 1.2669 | 1.4186 | 0.1516 | 0.2277 |
| Magicoder | frozen | 1 | 1.1100 | 1.3018 | 0.1919 | 0.2754 |
| Magicoder | frozen | 1/4 | 1.3393 | 1.4292 | 0.0899 | 0.1520 |
| Magicoder | frozen | 1/10 | 1.5634 | 1.5987 | 0.0353 | 0.0829 |
| Magicoder | nogram | 1 | 0.7225 | 0.8762 | 0.1537 | 0.2189 |
| Magicoder | nogram | 1/4 | 0.8403 | 0.9102 | 0.0699 | 0.1299 |
| Magicoder | nogram | 1/10 | 0.9589 | 0.9839 | 0.0250 | 0.0791 |
| OpenR1 | trainable | 1 | 0.6870 | 1.3726 | 0.6855 | 0.7312 |
| OpenR1 | trainable | 1/4 | 0.9798 | 1.2452 | 0.2654 | 0.2940 |
| OpenR1 | trainable | 1/10 | 1.1791 | 1.2974 | 0.1184 | 0.1347 |
| OpenR1 | frozen | 1 | 1.0425 | 1.2011 | 0.1587 | 0.1811 |
| OpenR1 | frozen | 1/4 | 1.2014 | 1.2810 | 0.0796 | 0.0860 |
| OpenR1 | frozen | 1/10 | 1.3481 | 1.3933 | 0.0452 | 0.0496 |
| OpenR1 | nogram | 1 | 0.8254 | 0.9820 | 0.1566 | 0.1666 |
| OpenR1 | nogram | 1/4 | 0.9322 | 1.0099 | 0.0777 | 0.0859 |
| OpenR1 | nogram | 1/10 | 1.0340 | 1.0710 | 0.0370 | 0.0438 |
| RLVR-MATH | trainable | 1 | 0.0115 | 0.5108 | 0.4993 | 0.5020 |
| RLVR-MATH | trainable | 1/4 | 0.0110 | 0.4583 | 0.4473 | 0.4482 |
| RLVR-MATH | trainable | 1/10 | 0.0300 | 0.4058 | 0.3758 | 0.3755 |
| RLVR-MATH | frozen | 1 | 0.0137 | 0.4357 | 0.4220 | 0.4178 |
| RLVR-MATH | frozen | 1/4 | 0.0215 | 0.3787 | 0.3571 | 0.3537 |
| RLVR-MATH | frozen | 1/10 | 0.0623 | 0.3002 | 0.2380 | 0.2339 |
| RLVR-MATH | nogram | 1 | 0.0183 | 0.3291 | 0.3108 | 0.3035 |
| RLVR-MATH | nogram | 1/4 | 0.0145 | 0.2894 | 0.2748 | 0.2705 |
| RLVR-MATH | nogram | 1/10 | 0.0247 | 0.2603 | 0.2356 | 0.2315 |

## Changes relative to original LR

Negative validation-loss change is an improvement. Negative gap change alone can also reflect slower training. Comparisons hold total token budget fixed, not training-loss convergence.

| Source | Arm | LR fraction | Change in train | Change in validation | Change in online gap | Gap / original |
|---|---|---:|---:|---:|---:|---:|
| id | trainable | q4 | +0.6764 | -1.0011 | -1.6774 | 0.319 |
| id | trainable | q10 | +0.9671 | -1.1646 | -2.1317 | 0.134 |
| id | frozen | q4 | +0.0478 | -0.1080 | -0.1558 | 0.543 |
| id | frozen | q10 | +0.1164 | -0.1253 | -0.2417 | 0.291 |
| id | nogram | q4 | -0.0414 | -0.1423 | -0.1009 | 0.681 |
| id | nogram | q10 | +0.0250 | -0.1827 | -0.2077 | 0.343 |
| magicoder | trainable | q4 | +0.3674 | -0.2084 | -0.5758 | 0.379 |
| magicoder | trainable | q10 | +0.6483 | -0.1277 | -0.7760 | 0.163 |
| magicoder | frozen | q4 | +0.2294 | +0.1274 | -0.1019 | 0.469 |
| magicoder | frozen | q10 | +0.4535 | +0.2969 | -0.1566 | 0.184 |
| magicoder | nogram | q4 | +0.1179 | +0.0340 | -0.0838 | 0.455 |
| magicoder | nogram | q10 | +0.2365 | +0.1077 | -0.1287 | 0.162 |
| openr1 | trainable | q4 | +0.2928 | -0.1274 | -0.4202 | 0.387 |
| openr1 | trainable | q10 | +0.4921 | -0.0751 | -0.5672 | 0.173 |
| openr1 | frozen | q4 | +0.1589 | +0.0799 | -0.0790 | 0.502 |
| openr1 | frozen | q10 | +0.3056 | +0.1922 | -0.1135 | 0.285 |
| openr1 | nogram | q4 | +0.1067 | +0.0279 | -0.0788 | 0.496 |
| openr1 | nogram | q10 | +0.2086 | +0.0890 | -0.1196 | 0.236 |
| rlvr | trainable | q4 | -0.0005 | -0.0525 | -0.0520 | 0.896 |
| rlvr | trainable | q10 | +0.0185 | -0.1049 | -0.1235 | 0.753 |
| rlvr | frozen | q4 | +0.0078 | -0.0570 | -0.0649 | 0.846 |
| rlvr | frozen | q10 | +0.0486 | -0.1355 | -0.1840 | 0.564 |
| rlvr | nogram | q4 | -0.0038 | -0.0398 | -0.0360 | 0.884 |
| rlvr | nogram | q10 | +0.0063 | -0.0688 | -0.0752 | 0.758 |

## Excess gap over matched no-gram

| Source | LR fraction | Trainable minus no-gram | Frozen minus no-gram |
|---|---:|---:|---:|
| ID | 1 | +2.1470 | +0.0251 |
| ID | 1/4 | +0.5705 | -0.0297 |
| ID | 1/10 | +0.2230 | -0.0089 |
| Magicoder | 1 | +0.7739 | +0.0382 |
| Magicoder | 1/4 | +0.2819 | +0.0201 |
| Magicoder | 1/10 | +0.1267 | +0.0103 |
| OpenR1 | 1 | +0.5290 | +0.0021 |
| OpenR1 | 1/4 | +0.1877 | +0.0019 |
| OpenR1 | 1/10 | +0.0814 | +0.0082 |
| RLVR-MATH | 1 | +0.1885 | +0.1112 |
| RLVR-MATH | 1/4 | +0.1724 | +0.0823 |
| RLVR-MATH | 1/10 | +0.1402 | +0.0023 |

## Interpretation limits

The LR reduction tests sensitivity to the size of optimization updates. It does not measure pretraining/downstream n-gram overlap, uniquely identify distribution shift, or establish that familiar n-grams suffice to induce a gap. The table and no-gram models start from different pretrained checkpoints with different baseline validation loss. All new jobs use the same H200 host/code/software; historical LR=1 ran on ophis with another torch version. Numerical and single-seed limits remain.

A100 batch72 no-gram smoke failed with OOM before the first update; it is retained as engineering evidence and excluded from scientific endpoints. Formal jobs preserve batch72 and use idle 360-2 H200s. Inputs live in /data/home/guoshaoyang/ngram-gap-lab/data; new outputs in /data/shared/guoshaoyang/ngram-gap-lab/posttraining_lr_20260925/data/runs_fixed. The shared location avoids the home-directory storage quota. No old artifacts are deleted.

## Artifacts and reproduction

- `assignments.json`: run IDs and settings; actual GPU/host/commands are in raw launch.json files.
- `inputs_manifest.json`, `*-input-verification.json`, `code_hashes.json`: immutable input and code provenance.
- `artifact_manifest.json`: remote paths, byte counts and SHA256, including final checkpoint and saved probe tokens. Large checkpoint files remain remote.
- `raw/<run_id>/`: verified logs, summary, launch and completion status.
- `endpoints.csv`, `curves.csv`: regenerated from verified sources.
- `baseline_uniform4.json`: original-LR endpoints and initial-checkpoint diagnostic on exactly the same fixed probe, when available.
- Figures: `docs/figs/posttraining_lr_sweep_20260925/`, including `all_source_lr_curves.pdf` and `lr_endpoint_overview.png`.

From repository root:

```sh
python3 docs/plot_scripts/collect_posttraining_lr_sweep_20260925.py
.venv/bin/python docs/plot_scripts/plot_posttraining_lr_sweep_20260925.py
python3 docs/plot_scripts/report_posttraining_lr_sweep_20260925.py
```


## Validation deterioration after the best logged point

Descriptive values from the recorded curve; these are not checkpoint selection claims. Only final weights were saved in this sweep.

| Source | Arm | LR | Best logged validation | Step | Final minus best |
|---|---|---:|---:|---:|---:|
| id | trainable | base | 3.2851 | 60 | 1.0911 |
| id | trainable | q4 | 3.1956 | 660 | 0.1795 |
| id | trainable | q10 | 3.1914 | 1010 | 0.0201 |
| id | frozen | base | 3.2958 | 360 | 0.0758 |
| id | frozen | q4 | 3.2526 | 980 | 0.0109 |
| id | frozen | q10 | 3.2460 | 1660 | 0.0002 |
| id | nogram | base | 2.7040 | 30 | 0.1596 |
| id | nogram | q4 | 2.6702 | 370 | 0.0510 |
| id | nogram | q10 | 2.6646 | 680 | 0.0163 |
| magicoder | trainable | base | 1.3326 | 730 | 0.2137 |
| magicoder | trainable | q4 | 1.3318 | 1510 | 0.0061 |
| magicoder | trainable | q10 | 1.4186 | 1665 | 0.0000 |
| magicoder | frozen | base | 1.2968 | 1640 | 0.0051 |
| magicoder | frozen | q4 | 1.4292 | 1665 | 0.0000 |
| magicoder | frozen | q10 | 1.5987 | 1665 | 0.0000 |
| magicoder | nogram | base | 0.8701 | 1420 | 0.0060 |
| magicoder | nogram | q4 | 0.9102 | 1665 | 0.0000 |
| magicoder | nogram | q10 | 0.9839 | 1665 | 0.0000 |
| openr1 | trainable | base | 1.2214 | 710 | 0.1512 |
| openr1 | trainable | q4 | 1.2402 | 1400 | 0.0050 |
| openr1 | trainable | q10 | 1.2974 | 1665 | 0.0000 |
| openr1 | frozen | base | 1.1985 | 1620 | 0.0026 |
| openr1 | frozen | q4 | 1.2797 | 1660 | 0.0013 |
| openr1 | frozen | q10 | 1.3933 | 1665 | 0.0000 |
| openr1 | nogram | base | 0.9709 | 1580 | 0.0111 |
| openr1 | nogram | q4 | 1.0087 | 1610 | 0.0011 |
| openr1 | nogram | q10 | 1.0710 | 1665 | 0.0000 |
| rlvr | trainable | base | 0.2726 | 170 | 0.2382 |
| rlvr | trainable | q4 | 0.2875 | 290 | 0.1708 |
| rlvr | trainable | q10 | 0.2945 | 520 | 0.1113 |
| rlvr | frozen | base | 0.2419 | 300 | 0.1938 |
| rlvr | frozen | q4 | 0.2443 | 460 | 0.1344 |
| rlvr | frozen | q10 | 0.2492 | 700 | 0.0510 |
| rlvr | nogram | base | 0.1905 | 250 | 0.1386 |
| rlvr | nogram | q4 | 0.1861 | 280 | 0.1033 |
| rlvr | nogram | q10 | 0.1858 | 370 | 0.0745 |

## Initial-checkpoint fixed gap and growth

These use the identical four uniform training-probe batches and fixed validation set. Initial gap is a train/val sample-difficulty offset before downstream optimization. Growth is final fixed gap minus this initial value, a separate diagnostic from the primary online gap. A negative net change does not imply no replay effect: rapid initial adaptation can reduce the offset before later gap growth. The full fixed-gap curves and min_logged_fixed_gap/fixed_gap_growth_after_min in endpoints.csv preserve this distinction.

| Source | Arm | Initial fixed gap | Final growth at LR 1 | At LR 1/4 | At LR 1/10 |
|---|---|---:|---:|---:|---:|
| ID | trainable | -0.0312 | 2.8098 | 0.9127 | 0.3856 |
| ID | frozen | -0.0312 | 0.3957 | 0.2072 | 0.1019 |
| ID | nogram | -0.0264 | 0.3821 | 0.2608 | 0.1329 |
| Magicoder | trainable | -0.0629 | 1.0922 | 0.5042 | 0.2906 |
| Magicoder | frozen | -0.0629 | 0.3384 | 0.2149 | 0.1458 |
| Magicoder | nogram | 0.0007 | 0.2182 | 0.1292 | 0.0785 |
| OpenR1 | trainable | 0.2241 | 0.5071 | 0.0700 | -0.0894 |
| OpenR1 | frozen | 0.2241 | -0.0429 | -0.1381 | -0.1745 |
| OpenR1 | nogram | 0.0354 | 0.1312 | 0.0505 | 0.0084 |
| RLVR-MATH | trainable | 0.0319 | 0.4702 | 0.4164 | 0.3436 |
| RLVR-MATH | frozen | 0.0319 | 0.3860 | 0.3219 | 0.2021 |
| RLVR-MATH | nogram | -0.0288 | 0.3323 | 0.2992 | 0.2603 |

## Domain-distribution diagnostic

Token histograms use the entire unique pretraining pool (993,853,440 target tokens) and each full downstream packed train pool. JS divergence is in nats (0 means identical unigram distributions). Exact n-gram coverage uses the same 4,096 uniformly sampled pretraining rows and 288 uniformly sampled downstream rows per source; it is a token-occurrence-weighted sampled lower bound on full-pretraining coverage, not a hash-table hit rate.

| Source | Unigram JS from pretraining | Sampled bigram coverage | Sampled trigram coverage |
|---|---:|---:|---:|
| id | 0.000081 | 89.32% | 50.49% |
| magicoder | 0.218077 | 80.59% | 35.16% |
| openr1 | 0.309210 | 87.45% | 46.37% |
| rlvr | 0.413521 | 88.98% | 43.46% |

ID is much closer in unigram distribution; OOD nevertheless reuses almost exclusively previously seen vocabulary and many sampled contexts. This measures marginal distribution and sampled identity overlap, not conditional continuation similarity, causal explanation of the gap ranking, or suitability of the pretraining corpus. Artifacts: domain_overlap.json and domain_overlap_provenance.json; CPU-only script code/tools/audit_posttraining_domain_overlap.py.


## Hash-table address coverage

Using the actual input-trigram hash, the first 15872 pretraining rows visit all 1048576 table addresses (within-row contexts only). A previously unseen exact context therefore need not address an unused parameter row: collisions share rows. This confirms address coverage, not exact-context familiarity, useful learned content, or a nonzero net gradient for every row. Source: table_row_coverage.json; script code/tools/audit_posttraining_table_row_coverage.py.
