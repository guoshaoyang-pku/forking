# Paper TODO coverage · Feishu v2 → local review draft

This table is the current read-only reconciliation of the unresolved markers in Feishu paper v2 (revision 1082). “Local ready” means a claim-safe replacement exists in the repository; “cloud open” means the Feishu page still contains the original placeholder and has not been edited.

| Feishu marker | Local replacement | Evidence / boundary | Status |
|---|---|---|---|
| §6.1 multi-epoch analytical prediction and alignment | `docs/report/paper-v2-review-draft.md` §6.1 | §8.6; `ffqv5_freeze_*_e2_10ep_fixed`; no fixed recovery rate or observed plateau claimed | local ready / cloud open |
| §6.2 LR–step–coherent-update relation | draft §6.2 | §42 backbone-LR and §45 freeze evidence; table-LR saturation described conditionally | local ready / cloud open |
| §6.3 weighted integral and fixed-step/fixed-pass distinction | draft §6.3 | S1 table-size/dose/epoch arrays; no universal `F→kF` law claimed | local ready / cloud open |
| §6.4 capacity/mapping/intervention boundary | draft §6.4 | S1 clean-table slopes; collision owner-mass measurements; net validation must be reported with train benefit | local ready / cloud open |
| §7.1 DeepSeek-like reproduction | draft §7 | No pinned model, module, data, run, or evaluation packet; protocol only | design only / gated |
| §7.2 SFT pressure test | draft §7 | Requires frozen/trainable arms, target/alternative continuations, unrelated contexts and whole-validation metrics | design only / gated |
| Appendix B setting and measurement contract | draft Appendix B | `agents.md` §1 SSOT; novel has no train loss and no standard novel gap | local ready / cloud open |
| Appendix C–H structure | `docs/plans/plan-6-appendix-structure.md` plus generated C.4/H.1.3 artifacts | C.4 has separate axes only; H.1.3 index filters asset/glob noise | local ready / cloud open |
| Figure numbering and placement | `docs/notes/figure-inventory-0912.md`; `docs/figure-index.html` | Current generated index is machine-derived; Feishu §3.3 numbering still needs cloud edit | review needed |

## What remains genuinely open

1. Copying this wording into the authoritative blog source and then syncing `docs/report/index.html` requires a deliberate publication edit.
2. D10/D11 numerical overwrite requires host-scoped config/train-log evidence; 33 duplicate IDs currently conflict across hosts.
3. T8/T9 have candidate endpoints, but their historical coordinates and probe semantics still need protocol review.
4. DeepSeek-like and SFT experiments remain gated by model/data/compute choices; the ~189 GiB table transfer is a separate P5 approval point.
