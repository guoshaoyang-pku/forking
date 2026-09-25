# Context examples: corpus membership and moving points

User-requested correction to the three-context figure, 2026-09-25.
The previous log-probability bars hid the checkpoint change and encoded word
identity by color. This version restores the earlier open-to-filled point
segments, with color identifying train-seen (blue), validation-only/novel
(orange), neither (grey), and the top-20 rank statistic (charcoal diamonds).
Train and Val count rows show exact context-token occurrences and distinguish
train-only from membership in both splits. A context-only probe has one
prediction distribution per model/checkpoint/input, not two distributions for
the corpus splits. Counts are not model probabilities.

All 72 plotted values and all 15 token memberships are unchanged from the
verified export in ../context_examples_20260925/. Source hashes, run IDs,
seed 42, epochs 1/5 (steps 337/1685), and full-softmax semantics are retained.
Counts use the independent valid-position recount, not candidate-selection
metadata. Input is BOS plus exactly three context tokens. Others remains the
log of the mean top-20 remaining probability, reselected per distribution.
Only two endpoints are displayed: connecting them does not establish a
monotone trajectory, and the horizontal offset just separates the markers.

Reproduce from the repository root:

    .venv/bin/python docs/plot_scripts/plot_context_examples_points.py

Outputs are in docs/figs/context_examples_points_20260925/. The reader is
shared with plot_context_examples_logp.py so the existing source/config/hash
checks apply. The previous bar version is retained as an archived alternative.
No training, checkpoint evaluation, or remote compute was performed.

Manuscript changes use only the shared paper/ checkout and preserve the
collaborator's new subsection organization and scientific discussion. The
old three probability/log-probability figures and walk-their-dog figure were
replaced by the unified 3x2 point figure; the five-epoch aggregate CE PDF is
unchanged byte-for-byte. Historical walk-their-dog rerun prose is retained.
The main-text schematic links to the measured there-is-a panels. Replay
terminology in Chapter 4 and Appendix F is epoch, as previously requested.
Before-edit TeX is retained here for review of concurrent manuscript edits.

Fable provided a caption draft. Its unsupported “n-gram reference” wording
was replaced with the actual no-gram / bigram-plus-trigram comparison; all
metric, membership, endpoint, and input claims were checked against sources.
