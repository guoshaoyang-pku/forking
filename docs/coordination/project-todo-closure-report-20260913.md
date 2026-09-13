# Project TODO closure report · 2026-09-13

This report is the current handoff point for the article phase. It reconciles the two Feishu pages, local paper draft, experiment log, coordination queue, resource index, figure index, remote summaries, and remote code identity. It is read-only with respect to Feishu, the blog repository, and GPU hosts.

## What is closed locally

- README navigation and article-phase entry points are current.
- Resource index, figure index, run-id index, appendix C.4/H.1.3 outputs, and untracked-resource classification are present.
- Historical status snapshots in docs/experiment-log.md that falsely implied active work were clarified; historical numbers and settings were retained.
- The machine-assisted active/history audit classifies all 23 status-like log lines; no historical `planned`/`running` marker is used as current execution evidence.
- Local paper review draft contains claim-safe replacements for §6.1–§6.4 and Appendix B.
- D2 handoff cards, D4 Engram decision, D20–D24 appendix/resource work, D14 model audit, and D26 local QC have reviewable artifacts.
- Repeatable checks exist for Feishu markers, navigation links, and overnight dispatch metadata.

## What is evidence-backed but not promoted

- T8: 360-1 nglab1x_nogram_long_v5_fixed, 8000 steps, seed 42, 800 log rows, disjoint shards; host-scoped endpoint only.
- T9: fixed probe is explicit in both host summaries (4 batches, shared probe hash, interval 10), but 360-2 1x logging is incomplete and 2x endpoints disagree; both use historical table scale 2.0.
- 33 duplicate _fixed IDs across hosts have endpoint metric differences. No averaging or blind overwrite was performed.
- Remote code identities differ: ophis matches local Python core files; 360-1/360-2 share a different Python revision.

## What remains open or gated

- D1 cloud write-back: completed as append-only status snapshots on the preserved pages; TODO revision 2335→2336 and paper v2 revision 1082→1083. Existing blocks, comments, images, and resource blocks were retained.
- Blog publication: local report and blog target are byte-identical, but the review draft is not published.
- D3/D9: DeepSeek-like / V4.1-Flash / SFT lines require pinned resources and approval; the approximately 189 GiB transfer remains a P5 gate.
- D15/D18: M6 and shorter-epoch questions require an explicit protocol decision.
- D16/D17/D19: candidate review or new run registration after code/data preflight.
- D27: git push remains a P5 gate. Current local main is ahead of origin/main; no push was attempted.

## Approval boundary

The Feishu write-back was authorized and completed as two append operations. Pre-write protection: TODO revision 2335/history_version_id 164864; paper revision 1082/history_version_id 114688. Post-write: TODO revision 2336; paper revision 1083. This authorization does not cover blog publication, git push, code synchronization, GPU execution, deletion, or large-file transfer.

## Verification at handoff

Current local snapshot: figure index reports 290 visualizations and 79 plotting scripts; the dispatch manifest contains 27 unique IDs (`D1..D27`); Feishu post-write snapshot is TODO rev 2336 / 25 comments and paper rev 1083 / 11 comments. Full fetch comparison shows each pre-write XML is an exact prefix of the post-write XML, and comment IDs are unchanged.

    python3 docs/coordination/check_navigation_links.py  # checked=32 missing=0
    python3 -m py_compile code/train.py code/diag_worker.py  # current-path compile
    find code tasks docs/coordination docs/plot_scripts -name '*.py' -print0 | xargs -0 -n1 .venv/bin/python -m py_compile
    find code tasks -name '*.sh' -print0 | xargs -0 -n1 bash -n
    git diff --check

The five untracked resources remain intentionally untouched and are listed in untracked-resource-audit-20260913.md.
