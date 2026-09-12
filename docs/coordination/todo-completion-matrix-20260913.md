# Unified TODO completion matrix · 2026-09-13

This matrix reconciles the two Feishu pages, `experiment-lines.md`, `plan-3-fix-and-backfill.md`, and the overnight queue. “Done” means a reviewable artifact and validation exist; “partial/stalled” is kept explicit.

| source item | local queue | current state | evidence / blocker |
|---|---|---|---|
| Feishu writing + symbols | D1 | **audit complete / prose open** | d1-paper-language-symbol-audit-20260913.md inventories open markers; paper v2 still needs prose patch. |
| Feishu shanbin/haoran handoff | D2 | **done locally** | `handoff-task-cards-20260913.md`; cloud page still has historical “未启动” wording. |
| Feishu Engram/overencoding decision | D4 | **decision recorded locally** | `engram-decision-record-20260913.md`; immediate direct reproduction NO-GO, conditional reopen criteria recorded. |
| Feishu logit-sharpening panel | D20 | **done as current evidence** | §53 20-epoch artifacts and figures; strict 2×2 remains optional. |
| Feishu Appendix C.4 | D21 | **done locally** | `c4_cross_axis_coverage.md` generated from 62/12/12 CSV rows; unmeasured interaction stated. |
| Feishu Appendix H.1.3 | D22 | **done locally** | `run_id_index.md` generated from registry/log/lines (329 IDs). |
| Feishu figure inventory | D23 | **done locally** | figure index: 290 visualizations, 79 scripts; no files moved/deleted. |
| Feishu untracked resources | D24 | **done locally** | `untracked-resource-audit-20260913.md`; PDFs have 30-entry SHA-256 manifest. |
| Feishu v2 ↔ local report diff | D25 | **done read-only** | `feishu-local-diff-20260913.md`; pages not modified. |
| Local T1 `_fixed` backfill | D10 | **stalled** | local `data` symlink target unavailable; zero local summaries. |
| Local T2 β₂ history cleanup | D11 | **stalled** | requires the same authoritative `_fixed` data mount; no unsupported rewrite made. |
| Local T3 figure regeneration | D12 | **queued / fixture gate** | scripts/index audited; data-backed regeneration waits for data mount. |
| Local T4 log structure | D13 | **audit complete / cleanup partial** | duplicate §10 repaired to §5; duplicate review finds no title collisions, numeric/table review remains. |
| Local T5 model/fallback audit | D14 | **partial** | source binding reviewed; remote import passes; pytest unavailable. |
| Local T6 M6 4x–8x | D15 | **needs user decision** | choose four new runs or limit claim to tested range. |
| Local T8 no-ngram long baseline | D16 | **queued / code-sync gate** | 360-2 is idle, but remote code MD5 differs; see remote-preflight-20260913.md. |
| Local T9 fixed-train probe | D17 | **queued / code-sync gate** | same MD5 mismatch and registration gate; no run started. |
| Local T10 shorter epoch | D18 | **blocked by T8** | may start only after T8 result. |
| Local T12 causal recovery | D19 | **queued / remote-check** | no new intervention run started. |
| Paper §6 predictions | D5–D8 | **audit complete / evidence open** | d5-d8-prediction-claim-audit-20260913.md separates replay, LR, data/epoch and capacity claims. |
| Paper §7.1 DeepSeek-like | D9 | **gated** | no direct reproduction without pinned implementation/model/compute authorization. |
| Paper §7.2 SFT stress test | D3 | **gated** | protocol can be designed; execution is not authorized or complete. |
| Publication/push | D27 | **gated** | local `main` is ahead of `origin/main`; push remains P5 approval. |

## Decision points for the next work block

1. Restore or mount the external data volume before T1/T2/T3 numerical work.
2. Decide T6: run the missing 4x/5x/6x/8x points, or narrow the written claim.
3. Decide whether the two Feishu pages should receive a cloud-side status update; this is a separate write operation.
4. Provide/enable a dependency-complete Python environment for pytest, or accept a remote environment setup task.
