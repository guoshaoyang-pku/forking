# Feishu ↔ local backlog audit · 2026-09-13

## Sources read

| source | revision | fetched content | checksum (SHA-256) |
|---|---:|---:|---|
| 0912 TODO wiki `G4KKwHMWriuHDOkARHCcHWBenwc` | 2335 | 34,544 bytes | `b41a2d89aeb225753f7e9608204d088fbe9a2f62ff3073257416989c1edfaae3` |
| paper v2 wiki `Lyr3wfH8FiIWT6kdQmrc6dGUnEf` | 1082 | 35,333 bytes | `2677d783f1602716fc12a0350d2cd3e9692f40d0ba714c163ae08f6531708b38` |

Fetched with `lark-cli docs +fetch --doc-format markdown --detail simple` using verified user identity.

## Confirmed open work

- The TODO wiki has 9 unchecked boxes and 17 unresolved markers (`待补`, `待核`, `未启动`, `进行中`, or `预留`).
- The paper v2 has no Markdown checkbox syntax, but it has 17 unresolved markers and explicit placeholders in §§6–8 / appendices.
- Both documents still require: human-readable prose and symbol cleanup, gap(i,j)/frequency and missing-mass definitions, formula and epoch-length audits, cross-model/SFT protocol decisions, figure placement/length review, and final evidence review.
- The TODO wiki still contains historical status text from before the local handoff work: it calls shanbin/haoran handoff “未启动”, C.4/H.1.3 extraction “待生成”, and the logit panel “已由待办 1b 的图 H-1/H-2 填上” in different places. Local evidence now supersedes those historical status lines for coordination, but the cloud page itself has not been edited.

## Local alignment

- Local queue D2/D4/D20–D24 records handoff cards, Engram decision, logit-sharpening material, C.4 coverage table, H.1.3 index, figure audit, and untracked-resource audit as completed or evidence-backed.
- Local queue D10/D11 are stalled because the `data` symlink points to an unavailable external SSD; no numerical backfill was guessed.
- Local queue D14/D26 are partial because model runtime needs dependencies and pytest is unavailable.
- D3/D9/D15/D18 remain gated by P5, user choice, or upstream dependency.

## Cloud update boundary

This audit is read-only. No Feishu page was modified. The source pages still need a deliberate cloud-side status update if you want the historical checkboxes and local evidence to be reconciled in place; that is a separate write operation.

## Immediate article-writing priorities

1. Resolve the six paper TODO lines before prose polish: formula correctness, gap(i,j) object, F/f notation, epoch-length versus pass count, and conditions for cross-model/SFT claims.
2. Attach the local evidence paths and run coordinates to the corresponding cloud sections.
3. Decide whether cloud pages should be updated or remain the immutable meeting record.
