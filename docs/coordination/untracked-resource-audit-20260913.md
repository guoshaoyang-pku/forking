# Untracked resource audit · 2026-09-13

This is a classification record for files present in the shared worktree but not yet tracked. It intentionally does not delete, move, or add large binary files.

| path | size | disposition | reason / next gate |
|---|---:|---|---|
| `docs/notes/literature/overencoding-practical-implications.md` | 18.9 KB | **retain and review** | Article-phase synthesis; it contains evidence and extrapolation labels, but external claims still need source-owner review before tracking or article citation. |
| `docs/plot_scripts/analyze_v5_seen_vs_novel_cpu.py` | 3.1 KB | **retain locally; do not adopt yet** | Script compiles, but executes against hard-coded `data/runs_fixed` and `data/freq_index.npz` paths; the local `data` symlink is dangling, so no numbers were regenerated. Its header also says “2020” while the run names are 2026-era, requiring owner review. |
| `docs/notes/literature/pdfs/` | ~31 MB | **retain locally, do not stage yet** | Citation source PDFs; `pdfs-manifest.tsv` now records 30 SHA-256 entries. Binary collection still needs a storage decision. No cross-cluster transfer. |
| `_tmp_script_mmd.mmd` | 670 B | **temporary; deletion requires review** | Scratch Mermaid source; no current index/reference found. |
| `lark_qr.png` | 895 B | **temporary; deletion requires review** | Feishu login artifact; unrelated to reproducible project evidence. |

## Safe next actions

1. Review the two text artifacts and add them to the resource/citation index if they pass source and path checks.
2. Create a checksummed PDF manifest before deciding whether PDFs belong in this repository, an external archive, or `.gitignore`.
3. Delete or move the two temporary artifacts only after explicit confirmation; this audit keeps them recoverable in place.
4. Do not stage or push any of these files as part of the current documentation commits.

## Verification update · 2026-09-13 continuation

- `overencoding-practical-implications.md` was read in full. It is useful as an article-phase synthesis, but it mixes `[实验证据]`, `[机制推断]`, and `[推测]`; it must remain visibly separate from the claims ledger until a citation pass is complete.
- `analyze_v5_seen_vs_novel_cpu.py` passes `py_compile` only. Because its data mount is unavailable and its inputs are hard-coded, compilation is not evidence that its numerical output is valid. It remains untracked and no plot or article number was derived from it.
