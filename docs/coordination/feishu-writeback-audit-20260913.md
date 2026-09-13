# Feishu write-back audit · 2026-09-13

## Scope

Two authorized append operations were applied to the existing Wiki-backed Docx pages. No existing block was replaced or deleted.

| page | Wiki token | protected pre-write revision / history | post-write revision | post-write comment count |
|---|---|---:|---:|---:|
| TODO / 0912 handoff | `G4KKwHMWriuHDOkARHCcHWBenwc` | 2335 / 164864 | 2336 | 25 |
| paper v2 | `Lyr3wfH8FiIWT6kdQmrc6dGUnEf` | 1082 / 114688 | 1083 | 11 |

## Preservation checks

- `docs +fetch --detail full` before and after: the complete pre-write XML is an exact prefix of the post-write XML for both pages; only the intended status snapshot was appended.
- `drive +list-comments --solved-status all --need-relation`: comment IDs are unchanged page-by-page (TODO 25/25; paper 11/11). Existing comment text, authors, replies, and relation payloads were not touched.
- Append-only updates did not rebuild the document, so existing images, whiteboards, citations, and other resource blocks remain in the preserved prefix.
- The appended paragraphs contain no comment anchors. Existing comments therefore remain attached to their original blocks; no comment was deleted or silently re-anchored.

## Limitations

The API does not expose a command to attach an existing comment to a newly appended block. Because this write-back intentionally preserved the original reviewed text and did not replace any commented block, no anchor migration was necessary. Any future replacement of a commented paragraph must use a block-level mapping and a comment-capable anchor operation before deleting or replacing that paragraph.

## Evidence files

- `/tmp/todo_full.json`, `/tmp/todo_after.json`, `/tmp/todo_comments.json`, `/tmp/todo_comments_after.json`
- `/tmp/paper_full.json`, `/tmp/paper_after.json`, `/tmp/paper_comments.json`, `/tmp/paper_comments_after.json`
- `docs/coordination/project-todo-closure-report-20260913.md`
