---
name: ngram-gap-settings
description: Define, review, or change an ngram-gap-lab training setting. Use for a new baseline or ablation, for checking whether a launcher matches the minimal contract, or before running an experiment; do not use for historical-result narration alone.
---

# Ngram Gap Settings

Read `agents.md` §0–1 before proposing a setting. Treat it as the setting SSOT;
use `docs/experiment-lines.md`, `docs/experiment-log.md`, and
`docs/claims-ledger.md` only for current experiment state and evidence.

## Establish the contract

State the full contract once for a new line, then state only the bold
differences from it in each run entry.

| Coordinate | Standard value |
|---|---|
| backbone | vanilla nanoGPT, 8L/6H/768D, vocab 8192, learned absolute position, LayerNorm, tied embeddings |
| n-gram path | bigram + trigram at `input`; unigram and fourgram off |
| table architecture | clean single full-width table, one layer and one hash per enabled branch |
| clean-table R | outside a table-size experiment, `R_bigram = R_trigram = 2^20 = 1,048,576` |
| backbone optimizer | AdamW `(0.8, 0.95)`, weight decay `0.1`, LR `0.0006` |
| table optimizer | RMSProp without momentum, `--table_betas 0.0,0.99`, `--table_lr_scale 128.0`; effective table LR `0.0768` |
| data | fixed-order replay, data seed `42`, non-overlapping train and validation shards |
| standard run | seed `42`, 1000 steps, bf16, no `torch.compile` |
| primary measurement | online training-batch loss and fixed validation-batch loss |

Always record `--lr_schedule warmup_constant --warmup_steps 100` explicitly.
All new experiments linearly warm from 0.25×LR (`0.00015` at the standard base
LR) at step 1 to 1×LR (`0.0006`) at step 100,
then hold LR fixed; this boundary does not move with epochs or total steps.
They may not use warmdown. `warmdown` exists only to rerun a registered
historical run. Zero-warmup `constant` is an optimizer diagnostic, not a
main-line setting unless a registered convergence check justifies it.

## Do not leave table capacity implicit

`R` is scientifically material because it controls collisions. A runnable
clean-table run must name every enabled branch's capacity, for example
`--bigram_clean_table <R_bigram>` and `--trigram_clean_table <R_trigram>`, or
name a perfect-map condition. Record the values in the run ID, setting table,
and summary. Do not silently fall back to `--table_mult`: that flag selects the
deprecated multi-layer/two-hash table framework and is historical-only.

If table capacity itself is the question, make `R` the sole variable. If it is
not the question, use the locked `2^20` clean-table capacity for every enabled
branch. Do not derive it from an old `1M` legacy-table label: the numerical
coincidence does not make the architectures equivalent.

## Preserve the measurement definition

Use the value written to `train_log.jsonl`:

```text
online gap(step) = fixed_val_loss(step) - online_train_loss(step)
```

The online training loss is the current optimization batch, evaluated before
that step's update; the fixed validation loss is evaluated on the fixed
validation batch set after the update. They share a logged optimizer step but
are not the same examples or an interchangeable held-out train probe. A fixed
train probe is diagnostic-only and must not replace the online gap in a main
claim or plot.

For a curve-producing run, use fixed validation and frequency evaluation every
10 steps. A sparse final-only run must pass matching `--val_steps` and
frequency settings, and its result must not be presented as a full curve.

## Change one coordinate

Before changing a setting, list every differing CLI flag and explain the one
scientific variable. Do not introduce current shell, Muon, RoPE, RMSNorm,
fourgram, gating variants, momentum on the table, moving validation windows,
or a legacy table. Use a new `run_id` whenever code or measurement semantics
change; do not overwrite completed artifacts.

End with an executable command, the expected output directory, the intended
metric/step, and the acceptance check. Hand the run to
`$ngram-gap-experiment-registration` before launch. For an unchanged main-line
baseline, start from `code/cluster/run_baseline.sh`; it locks both clean-table
capacities to `2^20` and refuses to overwrite a run directory.
