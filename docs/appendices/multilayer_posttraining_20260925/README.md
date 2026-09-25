# Multilayer frozen-memory post-training and verified mathematics data

Status: data prepared and hashes verified on360-2; eight runs queued for idle GPUs; no scientific endpoint yet.

The question is whether freezing pretrained lookup embeddings still permits
post-training gap growth substantially greater than an independently pretrained
no-gram backbone. Input-trigram negative results in §§69–71 do not answer this
multilayer question. This is a minimal gated Engram-like proxy, not a reproduction
of a production DeepSeek model.

## Registered comparison

Eight runs: `ptml_{id,math}_{tableonly,module,trainable,nogram}_L14_q4_s42_fixed`.
Each run owns one directory in the isolated remote `data/runs_fixed/` namespace.
The run matrix and exact command generator are `code/cluster/run_multilayer_posttraining.py`.

| Arm | Lookup embeddings | Gates | Backbone |
|---|---|---|---|
| tableonly | frozen before downstream step 1 | trainable | trainable |
| module | frozen before downstream step 1 | frozen | trainable |
| trainable | trainable | trainable | trainable |
| nogram | absent; independently pretrained baseline | absent | trainable |

Vanilla nanoGPT 8L/6H/768D, 8192 vocabulary, learned absolute positions, LayerNorm,
sequence 2048, dropout 0, bf16, no compile. Memory arms use bigram AND trigram,
each R=2^20 and width768, independently at zero-based blocks1 and4 (layers2 and5).
This is the existing checkpoint's layout; it is not silently relabeled as0,4.
Backbone AdamW(0.8,0.95), wd0.1, LR0.00015; warmup100 from0.25× then constant.
Memory optimizer RMSProp(0,0.99), ratio128. Trainable gates retain the existing
memory optimizer group and LR0.0192; this is explicitly a gate-policy ablation,
not a claim about official Engram post-training practice.

Device batch72, total147456 target tokens/update,333 updates/pass,5passes,1665
updates,245514240 target exposures. Both sources have23976 train packed rows
and288 val rows. Fixed val and uniform288-row train probe every10updates plus
exact replay boundaries. Initial fixed gap is measured before any update.
No compatible frequency index is used; the dense measurements are raw LM losses.

ID: first23976 rows of pretraining-validation shard21, held-out first288 rows of
shard22, from `tokenized_epoch20_ophis`. Both were excluded from pretraining
optimization. Math: fresh OpenR1 verified Algebra pool described below.

Bases, on360-2 `/data/home/guoshaoyang/ngram-gap-lab/data/runs_scaling/`:

- `s1v5_128_ep20_engram_L14_20p0xL4_3ep_3602_nockpt_fixed/final_model.pt`,
  SHA256 `2c8acc84fa25114cf80db6f401b2f5c8d2a2acaab2127791ca0955dbe88c1c25`.
- `s1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed/final_model.pt`.

Both are20× data-pool bases trained for3passes/20220updates. This is not a
no-replay pretraining comparison. One base realization, seed42; downstream
seed relabeling would not constitute independent pretrained replications.

The primary metric is fixed-val after update minus current-batch online train
loss before update. Report raw loss, online gap, fixed gap, and
`[fixed_gap_memory(t)-fixed_gap_memory(0)] - [fixed_gap_nogram(t)-fixed_gap_nogram(0)]`.
Retain all negative results. Correlated curve points are not statistical seeds.
No success criterion may be replaced by trainable-table success.

Acceptance: strict checkpoint compatibility, route/layers/table/gate construction,
2-step full-sized smoke, idle GPU admission, source/data hashes, complete dense
logs, finite metrics, exact final step, saved model/probes, identical source probe
hashes, bitwise frozen embeddings and (module arm) gates, changed backbone.
tableonly must additionally show changed gate weights.

## Data correction

`allenai/RLVR-MATH` at revision
`bd2a93551b503a395fadd1a740d957559cfe6f3c` contains7500 user prompts and no assistant
rollouts. All prompts include the same four worked few-shot examples and a fifth
query. The previous preparation appends the query ground truth and cycles the
source to fill the packed budget. Its label “verifier-selected” is unsupported;
the experiment is prompt-plus-reference-answer all-token replay, not RFT rollouts
or on-policy RLVR. The repeated prefix also appears in validation, so the
all-token average is strongly template-weighted. This does not by itself prove
the direction of the measured memory/no-gram contrast.

The old remote `code/prepare_ood_sft.py` reads OpenR1 `problem` + `solution`,
not `generations` or assistant messages. Thus the old OpenR1 line used reference
solutions, not generated R1 reasoning. Correcting the tokenizer did not correct
this separate data-semantic distinction. Historical artifacts are preserved.

New source: [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k),
revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`, Apache2.0.
Use existing raw `all/default-*.parquet` files on ophis. Predeclare Algebra; for
each unique normalized problem and uuid, select the first generation whose
Math-Verify flag and reasoning-complete flag are true. No LLM-judge-only samples.
Split normalized problem hash before packing (10% validation), one trace per
question, no internal cycling, same tokenizer as pretraining. Stop each split at
the fixed packed budget; the last selected record may be truncated. Save each
selected uuid/generation index, source hashes and token offsets. This guarantees
exact normalized-text/uuid isolation, not absence of semantic paraphrases.

This is static verified reasoning data suitable for a rejection-sampled
fine-tuning proxy. Our all-token LM objective is still not response-only SFT or
on-policy RLVR. Algebra and one teacher narrow the source distribution; prefix
concentration is measured rather than assumed to imply a large fork.

Other public candidates: DeepScaleR provides about40k problem/reference-answer
pairs (not saved policy rollouts); DAPO-Math-17k exposes prompt and reward_model.ground_truth, with no response
column in the verified dataset-viewer schema; it is not a saved-rollout corpus. OpenThoughts-114k provides verified synthetic
reasoning, but OpenR1 exposes per-generation verification flags most directly.

## Execution

```text
python3 <isolated_root>/code/cluster/run_multilayer_posttraining.py --root <isolated_root> --data-root /data/home/guoshaoyang/ngram-gap-lab/data --pool-root <isolated_root>/data/pools --out-root <isolated_root>/data/runs_fixed --source <id|math> --arm <tableonly|module|trainable|nogram> --gpu <idle_gpu> [--smoke]
```

Output root is under `/data/shared/guoshaoyang/ngram-gap-lab/` to avoid the home
quota (43GB remaining versus13GB per memory checkpoint). Existing workloads and
completed checkpoints are preserved. No large checkpoint transfer or push.


## Preparation audit and current execution state

New pool:9237 train problems and114 validation problems;23976/288 packed rows,
49,102,848 train targets/pass. Tokenizer SHA256 f6e7dad994ed5f8bc2da99d3eba8de3e466d9a0a61b99e209c3b7fd24ef0012e.
All selected examples and token offsets are recorded in math_examples.jsonl.
Counts refer to source records contributing tokens; final records can be cut by
packing. The fixed validation set is small and is not an independent-seed test.
The first8-word response prefix occurs1334/9351 times (14.27%); the10most common
prefixes cover4696/9351 (50.22%). These are lexical-template diagnostics, not a
measurement of mathematical solution diversity or proof of excess overfitting.

The old RLVR four-example prefix has1520characters and is identical in all7500
records. Its estimated token mass is87.175% (prefix tokenized independently).
The remaining query/answer tokens can dominate differences despite low all-token
CE. Do not infer that the template alone caused the old arm ordering.

The original .git was temporarily renamed to .git_disabled by an external action
while preparing this line; it was subsequently restored externally. No restoration,
reset or index repair was performed here. Source was first committed independently
as dc21b28731a1d8bdf465650dc9253f615c8c4bf8; primary repo commit is recorded separately.
The original staged79-item migration diff was preserved.

Idle-only dispatcher: code/cluster/dispatch_multilayer_posttraining.py. It checks
all8GPUs, admits only memory<150MiB/utilization0, runs a2step full-sized smoke per
source/arm, and launches1665steps only after its QCpasses. Admission expires after
8hours; already running jobs finish. A failed smoke stops that arm and is retained.
Queue state and logs are under <isolated_root>/queue/. This queue has no authority
to stop other jobs. Gate-only update and full checkpoint freeze checks occur in
smoke and again at completion. Training curves remain unavailable until a GPU is
admitted; queued is not running or done.
