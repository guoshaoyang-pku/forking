# Shared paper checkout and completed Overleaf merge — 2026-09-25

The user's current instruction makes paper/ the sole shared local paper Git
repository. All agents should use this directory. Do not create an independent
paper clone or worktree, or revive the deprecated external checkout. Earlier
read-only-mirror descriptions are superseded for authorized manuscript edits.

## Current sync

- Shared checkout: /Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/paper.
- Branch: main, tracking Overleaf origin/main.
- Remote project: git.overleaf.com/6a9b071075d832e243e895a2.
- Merged the five remote commits through 34d12a3 with local narrative through
  ed4205d; final merge commit 2a9873f was pushed normally and verified by
  a fresh remote ref read. Both heads matched, and the paper working tree was
  clean at completion. Live status must be checked before the next edit.
- Seven previously uncommitted Appendix B/C, table and supplement files were
  byte-identical to the fetched remote versions. They were preserved in local
  commit fa0ef70 before merging; no stash or destructive reset was used.
- Remote discovery prose and figures are retained. Chapter 3 combines the
  remote layout with local coverage text. The single Appendix E conflict
  combines removal of a repeated definition with the local mathematical
  qualifications. Figure 4 wrapping changed from 25 to 28 lines to prevent
  caption/text overlap after integration.
- main.tex compiled to 34 pages with no undefined references or overfull
  boxes. Affected pages were rendered and checked. paper/main.pdf is the
  current shared build (ignored by Git).

## Coordination

Inspect status and recent commits before changing files. Preserve other
agents' unfinished changes; do not reset, clean or blanket-stash the checkout.
Use path-scoped commits for ordinary edits. During merge, inspect the index
and existing merge state first. Fetch before push and use a normal push;
if the remote advanced, merge and verify again. Do not rebase shared history
or force-push. Push only under the user's authorization for the active task.

A merge-operation marker is .git/codex-sync.lock inside the paper repo;
when present, read its owner information before attempting Git mutations.
The temporary marker from this operation was removed after verification.
Backup bundle, dirty-file archive and manifests are kept outside the paper
checkout in the lab Git directory, under paper-sync-backups/.
