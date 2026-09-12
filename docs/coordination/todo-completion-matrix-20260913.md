# Unified TODO completion matrix · 2026-09-13

This matrix reconciles the two Feishu pages, `experiment-lines.md`, `plan-3-fix-and-backfill.md`, and the overnight queue. “Done” means a reviewable artifact and validation exist; “partial/stalled” is kept explicit.

| source item | local queue | current state | evidence / blocker |
|---|---|---|---|
| Feishu writing + symbols | D1 | **local review draft ready / cloud open** | `docs/report/paper-v2-review-draft.md` supplies claim-safe §6–§7/B prose; Feishu v2 still retains historical placeholders. |
| Feishu shanbin/haoran handoff | D2 | **done locally** | `handoff-task-cards-20260913.md`; cloud page still has historical “未启动” wording. |
| Feishu Engram/overencoding decision | D4 | **decision recorded locally** | `engram-decision-record-20260913.md`; immediate direct reproduction NO-GO, conditional reopen criteria recorded. |
| Feishu logit-sharpening panel | D20 | **done as current evidence** | §53 20-epoch artifacts and figures; strict 2×2 remains optional. |
| Feishu Appendix C.4 | D21 | **done locally** | `c4_cross_axis_coverage.md` generated from 62/12/12 CSV rows; unmeasured interaction stated. |
| Feishu Appendix H.1.3 | D22 | **done locally** | `run_id_index.md` regenerated from registry/log/lines (294 identifier-like IDs; asset names/glob patterns filtered). |
| Feishu figure inventory | D23 | **done locally** | figure index: 290 visualizations, 79 scripts; no files moved/deleted. |
| Feishu untracked resources | D24 | **done locally** | `untracked-resource-audit-20260913.md`; PDFs have 30-entry SHA-256 manifest. |
| Feishu v2 ↔ local report diff | D25 | **done read-only** | `feishu-local-diff-20260913.md`; pages not modified. |
| Local T1 `_fixed` backfill | D10 | **partial / conflict audit** | Remote summaries recovered; 33 duplicate IDs all have endpoint metric conflicts across hosts, so no blind overwrite is allowed. |
| Local T2 β₂ history cleanup | D11 | **partial** | Historical β₂ claims are marked; remaining numeric cleanup depends on host-scoped config/log evidence. |
| Local T3 figure regeneration | D12 | **queued / fixture gate** | scripts/index audited; data-backed regeneration waits for data mount. |
| Local T4 log structure | D13 | **active/history audit complete; numeric review open** | duplicate §10 repaired to §5; 23 status-like lines classified; historical/current labels are explicit, while numeric/table review remains. |
| Local T5 model/fallback audit | D14 | **done** | source binding reviewed; isolated local pytest suite passes 26 tests. |
| Local T6 M6 4x–8x | D15 | **historically bounded** | Existing decision limits the claim to measured points; no new historical runs are currently required. |
| Local T8 no-ngram long baseline | D16 | **covered by existing evidence; protocol review open** | `nglab1x_nogram_long_v5_fixed` is an 8000-step seed-42 endpoint with 800 log rows and disjoint shards; historical table scale 2.0 is inert because n-gram branches are disabled. Code/md5 and curve review remain. |
| Local T9 fixed-train probe | D17 | **covered by existing endpoints; conflict review open** | Both hosts explicitly record 4 fixed probe batches, the same hash, `train_probe_mode=first`, and interval 10. 360-2 1x log is incomplete (12 rows); 2x endpoints differ and both use historical 2× table LR. |
| Local T10 shorter epoch | D18 | **still blocked by protocol choice** | Existing S1 epoch-length array supersedes the old shrink-data proposal; no new run should start until the exact question is re-registered. |
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
