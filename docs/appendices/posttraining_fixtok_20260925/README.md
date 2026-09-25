# Corrected-tokenizer post-training curves and gap audit

Audit date: 2026-09-25. This packet supersedes the numerical interpretation of post-training gaps in `docs/handoff/plan7-handoff-20260925.md`, section 2.1. It contains completed training logs, a read-only final-checkpoint diagnostic, unified plots, and an official-source check of DeepSeek-V4.1-Flash. No training was performed by this audit.

## Findings that change the interpretation

1. The plotted gap is **online gap**: fixed-validation loss after update minus the current training batch's loss before update. It is not a fixed-train estimate or a whole-pass training mean.
2. The no-gram gap survives a same-checkpoint diagnostic on 33 deterministic training batches. All 13 re-evaluated final checkpoints reproduce their original four-batch fixed-validation loss exactly. The observed endpoint gaps are therefore not explained away by the online measurement convention. This does not certify every data preparation choice.
3. Frozen-table models retain positive and growing gaps. The reduction in **extra gap relative to no-gram** must not be described as the disappearance of gap or as a proof that trainable tables are necessary for forking. Magicoder retains a modest positive excess; RLVR retains a larger excess.
4. The nominal five-pass RLVR proxy includes approximately **42.7 raw-source token passes**: its 333-batch pool already cycles the 7,125 training examples about 8.54 times. Loss includes the fixed exemplars, question, and appended reference answer. This is a strong memorization stress test, not an on-policy RL result or a response-only SFT result.
5. The previous pretraining freeze result remains valid. It differs from this downstream experiment in corpus, initialization, table architecture, freeze timing, and optimizer history. A single causal explanation cannot be isolated from this comparison.
6. The checked official DeepSeek V4.1 materials do **not explicitly disclose** whether Engram is frozen during SFT, RL, or OPD. GPU residency during RL rollout does not establish freezing.

## Sources and coordinates

These are historical continued-LM/static-RLVR stress tests registered in experiment-log section 66. Their remote run namespace does not use the canonical `_fixed` suffix; they are not relabeled as main pretraining runs. Actual configurations establish fixed replay and fixed validation. The separate pretraining reference is drawn from `data/runs_fixed/` with `_fixed` run IDs.

Remote root: `ophis-gpu:/data4/guoshaoyang/ngram-gap-lab/data`. There are 30 runs and 5,010 loss records: three OOD sources × three arms × three downstream seeds, plus three ID arms at seed 42.

| Source | Run ID / remote parent | Seeds | Endpoint |
|---|---|---|---|
| Magicoder | `runs_sft_fixtok/sft_magicoder_fixtok_e5_{trainable,frozen,nogram}_s{seed}` | 42, 43, 44 | 1665 |
| OpenR1 | `runs_sft_fixtok/sft_openr1_10shards_fixtok_e5_{trainable,frozen,nogram}_s{seed}` | 42, 43, 44 | 1665 |
| RLVR-MATH | `runs_sft_fixtok/sft_rlvr_math_fixtok_e5_{trainable,frozen,nogram}_s{seed}` | 42, 43, 44 | 1665 |
| ID table arms | `runs_sft_freq10/sft_id_tri_e5_freq10_{trainable,frozen}_s42` | 42 | 1665 |
| ID no-gram | `runs_sft/sft_id_e5_nogram_s42` | 42 | 1665 |

Within every source: 333 batches per replay pass, five passes, batch 72, sequence 2048, BF16, backbone LR 0.0006, 100-step warmup then constant LR, data seed 42, zero dropout, validation every ten steps plus step 1665, four fixed validation batches. No fixed-train probe was enabled during training. Table models use input-injected trigram only, unlike the bigram+trigram main pretraining reference. See the archived configurations for the complete setting.

Table trainable/frozen arms initialize from `runs_scaling/sftbase_ep20_tri_ckpt_fixed/final_model.pt`. No-gram initializes from `runs_scaling/s1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed/final_model.pt`. These are different pretrained models. Seeds 42/43/44 are downstream seeds from the same initialization per arm, not independent pretraining seeds. Trainable-versus-frozen compares the same table checkpoint; table-versus-no-gram additionally changes pretrained initialization and architecture.

`source_manifest.json` stores remote paths and remote SHA256 values for every archived summary, training log, and table-norm log. The generator checks all these hashes, expected steps, endpoint agreement, gap arithmetic, fixed settings, freeze flags, and constant RMS values throughout each frozen-table run. Source logs are preserved under `runs/`, not reconstructed from figures.

## Tokenizer correction

The old OOD encoding used a different ID-to-token mapping from the pretrained model. Legal IDs below 8192 did not make those inputs semantically compatible with the pretrained embeddings and n-gram addressing. ID data were already correctly encoded. The three OOD sources were re-encoded and all three arms rerun.

The correct tokenizer recorded in each `TOKENIZER.json` is `/data2/ncpl-pathA/work/vbird_autoresearch/cache/tokenizer/tokenizer.pkl`, SHA256 `f6e7dad994ed5f8bc2da99d3eba8de3e466d9a0a61b99e209c3b7fd24ef0012e`. `tokenizer_and_data_manifests.json` archives the metadata directly from the corrected data directories. The generic tokenizer string in the OOD `manifest.json` is hardcoded and is not authoritative.

Old logs under `old_tokenizer_runs/` are **invalid evidence for post-training claims** and appear only in the explicitly labeled error-diagnosis plot. `old_tokenizer_manifest.json` preserves their origin and hashes. Re-encoding also changes token packing and the text extent covered by a fixed token budget; old/new differences are not a pure matched-text intervention.

At seed 42, the frozen/trainable online-gap ratio changes from 21.0% to 20.7% for Magicoder, from 13.5% to 23.1% for OpenR1, and from 82.0% to 84.5% for RLVR. Thus the relative increase is especially clear for OpenR1; it is not a universal increase in absolute frozen gap.

## Corrected online endpoints

Values are nats/token at step 1665. Mean ± sample standard deviation across downstream seeds; ID has only seed 42. Exact per-run values are in `endpoints.csv`.

| Source | Table trainable | Table frozen | No-gram | Frozen minus no-gram |
|---|---:|---:|---:|---:|
| ID | 2.462920 | 0.341044 | 0.315922 | 0.025122 |
| Magicoder | 0.923971 ± 0.005973 | 0.197767 ± 0.009700 | 0.131553 ± 0.019171 | 0.066214 |
| OpenR1 | 0.678133 ± 0.014742 | 0.158156 ± 0.001749 | 0.149219 ± 0.007926 | 0.008937 |
| RLVR-MATH | 0.509021 ± 0.008475 | 0.433748 ± 0.012436 | 0.295042 ± 0.035179 | 0.138707 |

Trainable/no-gram ratios of the final means are 7.80, 7.02, 4.54, and 1.73 respectively. These ratios do not measure absolute validation quality or a matched-initialization architecture effect. The corrected Magicoder frozen excess of 0.066 should not be rounded away by saying that all three non-RLVR frozen models are identical to no-gram.

## No-gram audit

The training code records the batch loss before `optimizer.step`, then evaluates the same four fixed validation batches after the update. The final-checkpoint diagnostic instead measures both losses with one set of weights: 33 deterministic training batches equally spaced across all 333 batches, including first and last, and the original four validation batches. It does no optimization and does not change weights or training output.

`checkpoint_audit.json` contains the twelve source × arm combinations at seed 42 and one extra RLVR no-gram seed 43. Strict state-dict loading and the original BF16 forward are used. All thirteen validation reproduction errors are exactly 0.0. The model/evaluation source files matched the remote host before execution; `train.py` SHA256 is `cf6ef3e78cad67ced506dd8d480d450995a9bf91b6a519d7326ba4a1975d1369`. Evaluator source was committed as `7848106`.

| No-gram source, seed 42 | Logged online gap | Same-checkpoint sampled gap | Initial-model sampled gap |
|---|---:|---:|---:|
| ID | 0.315922 | 0.367393 | -0.011997 |
| Magicoder | 0.153690 | 0.215804 | -0.011256 |
| OpenR1 | 0.156552 | 0.171382 | 0.031061 |
| RLVR-MATH | 0.310811 | 0.303118 | -0.043076 |

These sampled fixed-train results are an additional diagnostic, not replacements for the primary online curves. They are neither all-training-set averages nor historical fixed-probe trajectories. Intermediate checkpoints were not available to reconstruct such trajectories.

No-gram gap grows across passes, not only at the last point. At seed 42, the averages of logged gap observations within successive passes are approximately ID: -0.015, 0.074, 0.139, 0.221, 0.271; Magicoder: 0.000, 0.044, 0.085, 0.120, 0.170; OpenR1: 0.011, 0.038, 0.061, 0.093, 0.130; RLVR: 0.032, 0.152, 0.219, 0.266, 0.291. `epoch_logged_means.csv` contains the exact values. These are means of logged observations, not full-pass loss estimates.

Gap growth and validation deterioration are separate observations. No-gram ID fixed-val rises from 2.736874 at initialization to 2.863557. Corrected RLVR no-gram seed 42 initially adapts rapidly, reaches minimum fixed-val 0.190493 at step 250, and ends at 0.329128. Magicoder and OpenR1 validation mostly improve and then flatten, with only small late increases from their minima. Their gaps can grow while validation still improves overall.

The previous handoff called Magicoder no-gram validation around 1.80 its starting value. That is step 10, after adaptation. The actual initial checkpoint has corrected-Magicoder validation 3.707078, dropping to 0.876150 at step 1665. Plotted training trajectories start at step 10; they do not include the step-zero model evaluation.

RLVR no-gram seed 43 has a final online gap 0.254738 and sampled fixed gap 0.268953; its sampled training loss is 0.087715. This last-point deviation cannot be attributed solely to a hard last batch because the final checkpoint also has higher average sampled training loss. Local optimization fluctuation is possible; the exact cause is unresolved without intermediate checkpoints.

## Data and repetition audit

`data_row_audit.json` contains hashes and full-row comparisons for each actual fixed pool: 23,976 training rows and 288 validation rows, each 2,049 tokens. All IDs are within the vocabulary; no exact validation row equals a training row. OpenR1 has two duplicate training rows; other pools have unique complete rows. This audit does not establish example-level semantic deduplication, and repacking a repeated text stream can produce unique complete rows.

For the RLVR source, the preparation script splits 7,500 source examples into 7,125 train and 375 validation examples, then cycles each split until its target token pool is full. With the corrected tokenizer:

- One raw training-source cycle: 5,749,963 tokens.
- One prepared training pool: 49,126,824 stored tokens = 333 × 72 × 2,049.
- Pool/source ratio: 8.543850; five replays: 42.719252 source-token cycles.
- One raw validation-source cycle: 306,787 tokens; prepared fixed pool: 590,112 tokens.

These are source-token cycle ratios, not claims that every example has exactly the same exposure count. The pool truncates its final cycle. The effective training target count is 333 × 72 × 2,048 per nominal pass; using target tokens changes the ratio only slightly. `rlvr_repetition.json` preserves the accounting. Repeated fixed few-shot exemplars and all-token loss plausibly contribute to memorization, but the present audit does not isolate their causal contribution.

## Relation to the pretraining freeze result

The reference figure reads the original canonical logs, with source paths and hashes in `pretrain_reference_manifest.json`:

- `causalv5c_none_128x_fixed`, seed 42, step 1000: gap 2.723778.
- `causalv5c_freeze_table_e1_128x_fixed`, seed 42, step 1000: gap 3.452124. The recorded intervention occurs at step 338, with table parameters, optimizer state, and hash identity preserved.
- `nglab1x_nogram_v5_128x_freq10_fixed`, seed 42: plot truncated to step 1000 for the same displayed horizon.

That experiment trains input bigram+trigram from random initialization, then freezes the table after one pass while replaying the same corpus. This downstream experiment uses a long-pretrained trigram-only checkpoint, freezes before adaptation, and starts a fresh optimizer and warmup on the selected downstream pool. The former shows that continued table updates are not necessary for a large gap once the model has adapted to the replay corpus. The latter shows that freezing attenuates downstream excess gap in a source-dependent way. A frozen table may still supply structured features that a trainable backbone learns to use; this remains a mechanism hypothesis, not a proof that freeze timing alone explains the difference.

## DeepSeek V4.1 official-source check

Checked official model: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), revision `dba1be0a40aa45a94ad051997016db3960a90277`. The official [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf), model card, and inference README are archived under `literature/`; URL, revision, and PDF hash are in `literature/source.json`.

- Report sections 2.4.2 and 2.5 describe trainable Engram embeddings and their optimizer, without establishing a stage-specific post-training freeze policy.
- Section 3.1.3, page 18: “Embedding gradients are buffered during backward and returned to their owning ranks after the backbone backward pass.” This describes training infrastructure, without separating all SFT/RL/OPD policies.
- The same section says: “During RL rollouts, Engram embedding tables remain resident in GPU memory.” Residency concerns memory placement during generation; it is not a statement about trainer updates or `requires_grad`.
- Section 5.1, page 25 gives SFT → RL → OPD, but no explicit Engram freezing rule. Other mentions of frozen components concern the DSpark backbone or vision encoder.

Therefore the defensible answer is **not disclosed in the checked official materials**. Neither “DeepSeek freezes Engram during post-training” nor its opposite has been established here. A claim that freezing is their standard production choice should not cite these passages.

## Figures, captions, and reproduction

Outputs are under `docs/figs/posttraining_fixtok_20260925/`.

| Output | Contents |
|---|---|
| `fig_sft_rl_combined.{pdf,png,svg}` | Publication overview: all four sources, train/val above and online gap below; table-trainable and table-frozen arms only |
| `fig_posttraining_combined.{pdf,png,svg}` | Backward-compatible alias of the two-arm four-column publication overview |
| `fig_sft_unified.{pdf,png,svg}` | ID, Magicoder, OpenR1 subset in the same style |
| `fig_rlvr_unified.{pdf,png,svg}` | RLVR static proxy in the same style |
| `fig_nogram_audit.{png,svg}` | No-gram-only curves for all sources |
| `all_seeds_<source>.{png,svg}` | Three arms separately with shared y axes and individual seeds |
| `fig_tokenizer_before_after.{png,svg}` | Explicitly invalid old encoding versus corrected encoding, seed 42 |
| `fig_checkpoint_gap_audit.{png,svg}` | Online versus sampled fixed-train gap at one checkpoint |
| `fig_pretraining_freeze_reference.{png,svg}` | Canonical pretraining reference with freeze at step 338 |
| `posttraining_all_seed_curves.pdf` | 17-page atlas: 4 source summaries, 10 individual source/seed pages, 3 diagnostics |

Suggested combined caption: **Continued-LM and static-RLVR replay stress tests with corrected tokenization.** Columns show ID, Magicoder, OpenR1, and the static RLVR-MATH proxy. Top: current-batch online training loss (solid) and four-batch fixed-validation loss (dashed); bottom: their difference. Curves compare a trainable trigram table with a frozen trigram table; the no-gram arm is retained only in the separate diagnostic figure. Curves show raw logged-point means and ±1 sample SD across downstream seeds 42/43/44, with only seed 42 for ID, through step 1665. No temporal smoothing is applied. Each nominal pass replays the fixed 333-batch pool; for RLVR this pool already contains approximately 8.54 raw-source token cycles. These are all-token continued-LM tests, not end-to-end SFT/RL replications. Run IDs and settings are given in this packet.

The separate SFT-source and RLVR figures use the same color mapping, line conventions, fonts, and metric. The source-specific atlas uses one source color across separate arm panels, preserving the earlier SFT plotting layout. All 17 PDF pages were rendered and visually inspected; legends, axis labels, and footer text are visible. Main overview figures were also inspected individually.

From the repository root:

```bash
.venv/bin/python docs/plot_scripts/plot_posttraining_fixtok.py
```

The generator verifies the 30-run sources and archived old/pretraining hashes before plotting, and writes the CSVs directly from logs. The checkpoint diagnostic can be rerun on a host holding those checkpoints with an available GPU, using a **new output path**:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> python code/tools/audit_posttraining_gap.py \
  --data-root /data4/guoshaoyang/ngram-gap-lab/data \
  --out /path/to/new/checkpoint_audit.json
```

The diagnostic refuses to overwrite existing output. Its completed archived JSON is sufficient to regenerate the diagnostic figure locally without GPU access. `artifact_sha256.json` records the delivered source/evidence/figure bytes; it excludes itself. The packet and figures are local deliverables; this audit does not imply that the live Overleaf manuscript has been updated.
