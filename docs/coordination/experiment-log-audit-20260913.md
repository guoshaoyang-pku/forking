# experiment-log hygiene audit · 2026-09-13

This is a read-only audit. It records the structural repair completed so far and the remaining content review; it does not rewrite experiment evidence.

## Verified

- The two top-level `## 10` headings were distinct experiments: “基础实验统计与图表归档” and “shard 大小扫描”.
- The first was renumbered to `## 5` and the registry row for `nglab_plot_baseline` now points to §5.
- The shard scan remains §10, so existing §10 cross-references for the shard/dose experiment remain meaningful.
- `git diff --check` passes after the repair.

## Remaining review

- The log still has historical sections written before the current `_fixed` cleanup. T1/T2 must reconcile their numerical claims against `data/runs_fixed/*_fixed/summary.json`; this audit does not declare that work complete.
- A paragraph-level duplicate scan and a full top-of-file table of contents are still pending. A duplicate-looking paragraph may be an intentional historical warning, so deletion requires manual review.
- Missing low-level section number 5 is now repaired; later `§21`/`§24` aliases are historical aliases and should not be renumbered without changing citations.

## Evidence

- Source: `docs/experiment-log.md`
- Repair commit: recorded in the current worktree after the preceding documentation commits
- Validation: heading search and `git diff --check`
