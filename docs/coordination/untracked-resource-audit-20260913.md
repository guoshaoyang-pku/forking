# Untracked resource audit · 2026-09-13

This is a classification record for files present in the shared worktree but not yet tracked. It intentionally does not delete, move, or add large binary files.

| path | size | disposition | reason / next gate |
|---|---:|---|---|
| `docs/notes/literature/overencoding-practical-implications.md` | 18.9 KB | **retain and review** | Article-phase synthesis; link from literature index after citation/core-credit review. |
| `docs/plot_scripts/analyze_v5_seen_vs_novel_cpu.py` | 3.1 KB | **retain and review** | Analysis script for §53; verify its input paths and claim ceiling before tracking. |
| `docs/notes/literature/pdfs/` | ~31 MB | **retain locally, do not stage yet** | Citation source PDFs; `pdfs-manifest.tsv` now records 30 SHA-256 entries. Binary collection still needs a storage decision. No cross-cluster transfer. |
| `_tmp_script_mmd.mmd` | 670 B | **temporary; deletion requires review** | Scratch Mermaid source; no current index/reference found. |
| `lark_qr.png` | 895 B | **temporary; deletion requires review** | Feishu login artifact; unrelated to reproducible project evidence. |

## Safe next actions

1. Review the two text artifacts and add them to the resource/citation index if they pass source and path checks.
2. Create a checksummed PDF manifest before deciding whether PDFs belong in this repository, an external archive, or `.gitignore`.
3. Delete or move the two temporary artifacts only after explicit confirmation; this audit keeps them recoverable in place.
4. Do not stage or push any of these files as part of the current documentation commits.
