# Read-only post-training gap audit, 2026-09-25

User requested all corrected-tokenizer training curves and investigation of the no-gram gap. This task reuses completed continued-LM stress-test runs registered in experiment-log section 66; it does not train a model or introduce a new training setting. The historical post-training namespace has no `_fixed` suffix; fixed data and validation semantics must be verified from actual configurations and code, and figures must be labeled as continued-LM/static-RLVR proxies rather than canonical main pretraining runs.

The raw evidence packet contains 30 runs: 27 corrected OOD runs (three sources, three arms, seeds 42/43/44), plus three ID arms at seed 42. Every downloaded file is checked against a SHA256 calculated remotely.

Diagnostic ID: `posttraining_gap_audit_20260925`. Host: ophis-gpu; one idle GPU reserved only for forward evaluation. The evaluator loads the final checkpoint of each source/arm at seed 42, plus RLVR no-gram seed 43 to investigate its last-batch spike. It evaluates 33 deterministic training batches spanning all 333 batches, including head and tail, and the exact four original validation batches. The no-gram initial checkpoint is also evaluated on the same batches. Configuration, weights and optimizer are not changed.

Acceptance: strict state-dict loading; exact source model config; BF16, batch 72, sequence 2048; reproduce logged final validation within numerical precision; retain per-batch results, code hash and checkpoint size; label sampled fixed-train diagnostic separately from online gap. A discrepancy triggers investigation and is never silently substituted into the paper curves. The absence of intermediate checkpoints prevents a fixed-probe training trajectory from being reconstructed.

Evaluator: `code/tools/audit_posttraining_gap.py`; output is a new JSON under this packet. Main training sources `code/train.py` and `code/tools/eval_logits_rank.py` match ophis byte for byte before evaluation.
