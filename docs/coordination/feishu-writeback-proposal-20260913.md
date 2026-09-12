# Feishu write-back proposal · 2026-09-13

This is a reviewable, append-only proposal. It does not modify either Feishu page. The intended operation is block_insert_after at the existing section anchors below; historical checkboxes and meeting notes remain unchanged.

## Target pages and anchors

| page | revision read | document anchor | proposed operation |
|---|---:|---|---|
| TODO wiki G4KKwHMWriuHDOkARHCcHWBenwc | 2335 | doxcnDEqL7Pu6ccZEUJajDO3mpe (未启动项与交接坐标) | insert an article-phase status note after the existing handoff paragraph |
| paper v2 Lyr3wfH8FiIWT6kdQmrc6dGUnEf | 1082 | doxcnJquY9M2VzWvZId41HEvmGg (本次修改记录与待核项) | insert a current evidence boundary note after the revision record |

## Proposed TODO-page insertion (XML)

<h3>文章阶段状态同步（2026-09-13，本地审计）</h3>
<p>以下状态来自仓库当前文章阶段文档、active/history 审计和对应 evidence packet；具体提交以写回前重新核验的本地 HEAD 为准。原有会议 checkbox 保留为历史记录；当前排程以 docs/coordination/overnight-queue-20260913.md 和机器可读的 overnight-dispatch-manifest.json 为准。</p>
<table>
  <thead><tr><th>工作项</th><th>当前状态</th><th>证据与边界</th></tr></thead>
  <tbody>
    <tr><td>D1 写作与符号</td><td>本地 ready，云端 open</td><td>docs/report/paper-v2-review-draft.md；需审阅后再替换正文</td></tr>
    <tr><td>D2 handoff</td><td>本地 done</td><td>docs/coordination/handoff-task-cards-20260913.md</td></tr>
    <tr><td>D4 baseline 调研</td><td>本地 decision recorded</td><td>engram-decision-record-20260913.md；直接 Engram 复现维持 NO-GO</td></tr>
    <tr><td>D10/D11 数值与 beta2</td><td>partial / conflict audit</td><td>33 个重复 run ID 有 host-level 差异；不合并、不取平均、不覆盖 experiment-log</td></tr>
    <tr><td>D16/T8</td><td>candidate review open</td><td>360-1 有 8000 行完整 no-gram 日志；仅作 host-scoped endpoint</td></tr>
    <tr><td>D17/T9</td><td>candidate conflict review</td><td>fixed probe 已记录；360-2 1x 日志不完整，2x 两端冲突且为历史 2x table LR</td></tr>
    <tr><td>D3/D9/D15/D18/D27</td><td>gated</td><td>分别受 P5、大文件/新架构、协议选择或发布审批约束</td></tr>
  </tbody>
</table>
<p>完整机器可读审计：feishu-marker-audit-20260913.json、candidate-run-protocol-audit-20260913.md、todo-completion-matrix-20260913.md。本段不改变原始会议记录，也不宣称后台 multi-agent 已启动。</p>

## Proposed paper-page insertion (XML)

<h2>当前证据边界（2026-09-13，本地审阅）</h2>
<p>本地 claim-safe review draft 已覆盖 §6.1–§6.4 与 Appendix B，但尚未写回本页或发布源。DeepSeek-like / Engram-style 复现和小数据 SFT 压力测试仍是 protocol，不是实验结果；T8/T9 候选保留 host-scoped，不能升级为当前 128× 主线证据。实验日志、图索引和资源索引的当前入口见 docs/coordination/overnight-queue-20260913.md、docs/resource-index.md 和 docs/report/paper-v2-review-draft.md。</p>

## Safe execution sequence after approval

1. Re-fetch both anchors with --detail full and verify revision IDs are still 2335 and 1082.
2. Run lark-cli docs +update --command block_insert_after once per page, using the exact XML above.
3. Re-fetch each page and verify the new blocks appear after the requested anchors; record new revision IDs and block IDs.
4. Do not use overwrite, delete existing blocks, mark gated experiments done, or modify the blog source in the same operation.

The proposal intentionally leaves blog publication, git push, large-file transfer, and new GPU runs outside the write-back operation.
