# Engram-style baseline decision record · 2026-09-13

## Decision state

**NO-GO for an immediate direct reproduction in this overnight queue.** This is a scheduling decision, not evidence that Engram is ineffective.

## Evidence

- The local study has a token-remapping proxy and an over-encoding literature survey, but no matched Engram/DeepSeek reproduction.
- The article's direct logit-sharpening evidence is from the local clean-table protocol (`ls20ep_input_v5_128x_fd`, `ls20ep_nogram_v5_128x_fd`, seed 42, step 6740); it cannot be relabelled as an Engram result.
- The direct reproduction would require architecture/code agreement, model/checkpoint access, and potentially large cross-cluster transfers.

## Conditional re-open criteria

Re-open only after all of the following are available: (1) a pinned public implementation and license, (2) a declared checkpoint/data path, (3) an approved compute and storage budget, (4) a registered one-variable protocol, and (5) a frequency-conditioned evaluation plan.

## Article wording

Use: “Engram-style direct reproduction is left as future work; our current evidence is a clean-table mechanism study plus a token-remapping proxy.” Do not write that Engram itself was validated or falsified.

## Approval boundary

No download, checkpoint movement, architecture branch, or `git push` follows from this record. Any such action remains an `agents.md` P5 approval node.
