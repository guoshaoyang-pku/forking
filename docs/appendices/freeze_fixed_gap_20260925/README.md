# Freeze intervention: online, fixed-head, and full-pool gaps

Audit date: 2026-09-25. User hypothesis: the first pass may already create a large fixed gap, while subsequent online growth merely reveals it; perhaps fixed gap stops growing when tables freeze. This audit recovers historical fixed-head curves and evaluates two existing final checkpoints on the entire training pool. No training or architecture change was performed.

## Findings

- The online endpoint overstates the **full-pool** endpoint in pass-1 freeze runs: about 3.43 online versus 2.52 full-pool fixed gap at step 1000 (seeds 43/44).
- The historical fixed probe is the first four batches only. It **understates** that full-pool endpoint: about 0.29 versus 2.52. Neither population is interchangeable with the other.
- Freezing after pass 2 does not stop fixed-head gap growth: seed 42, step 674 to 3370, 1.2484 to 3.8651; fixed validation rises 4.7698 to 7.4534. Freezing both table and backbone leaves the same fixed-head gap unchanged at 1.24447. Continued table updates are not necessary for further fixed-gap growth in this protocol.
- The pass-1 frozen checkpoint's **full-pool gap at the freeze boundary is unavailable**. We cannot quantify how much of its final full-pool gap was inherited from pass 1 versus acquired afterward. Historical fixed-head curves cannot replace that missing quantity.

## Reconstruction and provenance

All sources are canonical _fixed runs. Historical fixed_train_probe_batches=0 does not mean there is no fixed diagnostic: the exact-frequency path captures the first four training batches separately and evaluates them after optimization. Historical code at git commit b976c71, code/train.py, explicitly assigns exact_freq_train_batches from four calls to the independent diagnostic iterator when no explicit probe is enabled. It does not advance training. Old runs used FP32 diagnostic forwards; newer seeds use BF16 diagnostics.

For each saved step and split, scalar loss = sum of all frequency-bin loss_sum / sum of all frequency-bin token_count. All bins are included, including validation-only novel tokens. This is the difference of complete train/validation marginal means, not an average of matched-context gaps. Bigram and trigram partitions independently recover the same mean to below 4e-14. Every split has 589824 tokens (4 × 72 × 2048), with identical per-bin counts at all saved times. Diagnostic versus logged validation differs by at most 0.000238 in old runs and below 4e-7 in newer seeds.

The reduced JSONs include source paths, full summaries, SHA256 values of the source summary/train/exact-frequency logs, all reductions, and validation checks. They were computed on the source hosts to avoid transferring large raw logs. Reproduction tool: docs/plot_scripts/extract_freeze_fixed_gap.py.

| Host and root | Runs |
|---|---|
| ophis-gpu:/data4/guoshaoyang/ngram-gap-lab/data/runs_fixed | causalv5c_none_128x_fixed; causalv5c_freeze_table_e1_128x_fixed; causalv5c_freeze_table_e1_128x_s43_fixed; causalv5c_freeze_table_e1_128x_s44_fixed; causalv5c_freeze_backbone_e1_128x_fixed |
| 360-2:/data/home/guoshaoyang/ngram-gap-lab/data/runs_fixed | causalv5m3_freeze_table_e2_fixed; ffqv5_freeze_table_e2_10ep_fixed; ffqv5_freeze_backbone_e2_10ep_fixed; ffqv5_freeze_both_e2_10ep_fixed |

There are 733 recovered observations in recovered_curves.csv. No averaging across FP32/BF16 histories is applied.

## Pass-1 freeze, seed 42

Run causalv5c_freeze_table_e1_128x_fixed: input bigram+trigram, table LR 128×, freeze at step 338, endpoint 1000. Step 330 is the last logged pre-intervention point, not the exact boundary checkpoint.

| Step | Fixed-head train | Fixed validation | Fixed-head gap | Online gap |
|---:|---:|---:|---:|---:|
| 330 | 4.746295 | 4.769805 | 0.023510 | -0.065261 |
| 340 | 4.668034 | 4.757262 | 0.089228 | 0.053906 |
| 350 | 4.574364 | 4.747273 | 0.172909 | 0.183891 |
| 670 | 5.220860 | 5.403553 | 0.182693 | 3.156878 |
| 1000 | 5.287107 | 5.563197 | 0.276090 | 3.452124 |

The head-probe gap stays relatively small and grows modestly. It does not show a large inherited **head-probe** gap before freezing, but says nothing decisive about the full-pool gap then. Fixed validation itself deteriorates by about 0.79 between steps 330 and 1000, independently of the training denominator.

The online sawtooth can reflect sample position as well as learning over time. Frozen table contents retain a nonuniform history across the deterministic stream. Recency, hash interference, and backbone co-adaptation are plausible contributors; their separate causal contributions are not isolated here.

## Full-pool final-checkpoint evaluation

Seed 42 has no saved checkpoint in the audited directory. Later seed-43/44 runs have final models but no boundary/intermediate checkpoints. Both were evaluated on idle ophis GPU 6 with strict checkpoint loading, BF16, batch 72, sequence 2048, no gradients, and all 337 original training batches. The original four validation batches were also evaluated; both validation reproduction errors are exactly zero.

| Seed, step 1000 | Full train mean | Fixed val | Full-pool fixed gap | Head-four fixed gap | Online endpoint gap |
|---|---:|---:|---:|---:|---:|
| 43 | 3.024648 | 5.544495 | 2.519847 | 0.287720 | 3.425178 |
| 44 | 3.001207 | 5.519091 | 2.517885 | 0.288533 | 3.427931 |

Same-checkpoint first-four versus last-four training loss is 5.256775 versus 2.169315 (seed 43), and 5.230558 versus 2.120177 (seed 44). The figure shows every batch without smoothing. A fixed head is a poor estimator of the full pool for these models. This establishes a large final gap, not its full-pool growth rate after freezing.

Evaluator: code/tools/audit_freeze_full_train.py, committed before execution as 6a7b1d8. Local and remote code/train.py SHA256: cf6ef3e78cad67ced506dd8d480d450995a9bf91b6a519d7326ba4a1975d1369. Evaluator hash, checkpoint sizes, and all batch losses are in full_train.json. Remote diagnostic: /data4/guoshaoyang/ngram-gap-lab/data/diagnostics/freeze_full_train_audit_20260925/full_train.json. Checkpoints stayed remote and were unchanged.

## Pass-2 freeze: temporal counterexample

Runs ffqv5_freeze_{table,backbone,both}_e2_10ep_fixed, seed 42, intervention step 675. Each arm uses the same fixed samples throughout. Arms are separate training runs with small pre-intervention numerical differences, not bit-identical shared-checkpoint forks.

| Arm | Fixed-head gap at 674 | At 3370 | Change | Fixed val, 674 → 3370 |
|---|---:|---:|---:|---:|
| Freeze table | 1.24836 | 3.86509 | +2.61673 | 4.76980 → 7.45338 |
| Freeze backbone | 1.24337 | 2.72401 | +1.48064 | 4.75769 → 4.96271 |
| Freeze both | 1.24447 | 1.24447 | 0 | 4.76562 → 4.76562 |

The dense shorter causalv5m3_freeze_table_e2_fixed run independently gives fixed-head gap 1.23464 at step 670 and 2.86771 at step 2022. The long result is not an online last-batch effect. Freezing backbone still permits table fitting to reduce training loss while validation changes modestly; memorization and validation damage are different outcomes.

In the freeze-both control, online gap jumps from 1.17966 at step 674 to 3.07589 at the next logged pass endpoint, although fixed-head gap stays 1.24447. That compares the last batch before its last update to the same batch after that update. It is not further learning after all weights freeze. This directly supports caution about interpreting online gap as same-checkpoint generalization gap.

## Interpretation and injection locations

Freezing a table does not freeze the input-output function while backbone parameters remain trainable. A fixed table supplies context-dependent features; the backbone can learn to use them differently. Models with and without a fixed table have different representations and gradients, so training only their backbones need not behave alike. Even arbitrary fixed features may provide context identifiers; there is no zero-gap guarantee.

Seeing common vocabulary in pretraining is weaker than learning the same ordered n-gram keys, table rows, and conditional continuation statistics. Overlap can make frozen features reusable, but overlap alone does not imply gap: continuation coverage, distributions, replay dose, collisions, and adaptation also matter. Conversely, finite-data fitting can create gap without n-gram tables or explicit multi-pass replay. These experiments do not measure downstream/pretraining context overlap and do not establish that downstream combinations were unseen.

For injection before zero-indexed blocks 0 and 4, the same principle holds twice: frozen table outputs interact with changing hidden states and subsequent layers. If gates or projections remain trainable, freezing embeddings alone does not freeze the complete memory pathway. Current main pretraining evidence uses single input injection and does not determine the new route's effect size. Local input mode injects once before the block loop; inject_layers applies to v/y/engram routes and changes the computation. The official DeepSeek V4.1 Flash report section 2.4.2 specifies **zero-indexed layers 1 and 14**, not 0 and 4, and includes context-aware gating. Treat 0/4 as a proposed small-model ablation, not the exact production layout.

The missing direct discriminator for the pass-1 hypothesis is the same full-pool or predeclared uniform fixed metric at step 337 and subsequent checkpoints, ideally from a shared boundary checkpoint with all four freeze arms. This audit did not launch that training experiment. The old head probe and endpoint-only full-pool evaluation cannot substitute for it.

## Figures and reproduction

Outputs under docs/figs/freeze_fixed_gap_20260925 (PNG/SVG):

- fig_e1_online_vs_fixed_head: seed-42 control/table-freeze/backbone-freeze fixed-head and online curves.
- fig_e2_long_fixed_gap: pass-2 freeze, fixed-head gap and validation over ten passes; dots are measured endpoints.
- fig_full_pool_position_audit: all batch losses and three gap definitions at step 1000, seeds 43/44.

Run .venv/bin/python docs/plot_scripts/plot_freeze_fixed_gap_audit.py locally. It reads archived reductions and independent full-pool results. All rendered figures were inspected. artifact_sha256.json records evidence/source/figure hashes and excludes itself. Authoritative paper/blog files were not modified.
