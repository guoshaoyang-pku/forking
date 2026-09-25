# Three context-only continuation examples

User-requested figure revision, 2026-09-25. No model evaluation or training.

Layout: petting your, introducing your, there is a; three rows, no-gram left
and bigram-plus-trigram right. Only negative natural log probability bars
remain, with zero baseline and shared 0–16-nat scale. Epochs 1 and 5 are
light and solid. Details belong in the LaTeX caption, not a figure footer.

Archived logits: existing ep5ckpt_v5_{nogram,both}_s42_fixed runs, seed42,
steps337/1685, input injection, backbone LR0.0006, RMSProp(0,0.99), table
LR128x, R=2^20, warmup_constant(100), replay shard1. Export used fp32 CPU
forward with BOS plus exactly3 context tokens. Softmax is recomputed from
all8192 logits without displayed-token renormalization. Maximum discrepancy
from archived named log probabilities: below1.6e-6nats.

Sources/exporter were recovered from committed lab work cfd6a9b; its old
PDFs were byte-identical to shared paper PDFs. source_manifest.json records
original paths/hashes. Copies here remove dependencies on another worktree
or temporary directory. Manuscript edits occur only in shared paper/.

Classes use the independent valid-position recount in context_counts.json:
train shard1, heldout shards2,3,4; context and target inside2049-token chunks.
Do not conflate these with earlier candidate-selection metadata counts.
Labels are per exact context-token pair, not token-global. Counts are saved.

Others is negative log of arithmetic mean of top20 remaining probabilities,
excluding dog/cat/fish/horse/bird. Membership is reselected per distribution.
It is a rank statistic, not fixed-group or remaining mass, nor mean token NLL.

Reproduce from repository root:

    .venv/bin/python docs/plot_scripts/plot_context_examples_nll.py

Outputs: docs/figs/context_examples_20260925 (PDF/PNG/SVG, values/classes CSV,
manifest). Paper supplement contains corresponding CSV and checkpoint hashes.
Raw export metadata retains checkpoint, summary, tokenizer and code hashes.

Changes: Chapter4 references and AppendixF example presentation, epoch
terminology. walk-their-dog is no longer included. The five-epoch aggregate
fig09_crowd_ce_seen_novel.pdf is preserved byte-for-byte; its aggregation ID
is retained in paper/supplement/context_examples_manifest.json. AppendixE
and masking limitations are unchanged. Before-edit TeX copies are in before/.

Fable supplied drafts; unsupported 'before and after memorization' wording
and unavailable cref macro were not adopted.

Validation: hashes/configs checked;72 bars recomputed;15 classes from counts;
full LaTeX compile without undefined refs, overfull boxes or oversized floats.
Pages7,32,34 rendered/inspected. Existing hyperref PDF-string warnings remain.
Examples are Figure21,p34 in the validated37-page build.
