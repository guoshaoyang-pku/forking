# 2026-09-12 handoff review

## 判定

这份 handoff 足以让我接管仓库的开发、实验登记、图表生成和证据审计主线。它给出了当前 setting SSOT、权威文档层级、主要 run 与 commit 链、集群坐标、已完成的 benefit/logit/literature/appendix 工作，以及未启动任务。

它还不是“全项目已闭环”的 handoff，原因是有几项工作仍明确等待决策或外部输入：

- 全文中文初稿与符号统一尚未启动；
- shanbin / haoran 的任务边界和交付格式尚未形成正式 handoff；
- Engram-style baseline 是 NO-GO 建议，仍待拍板；
- V4.1-Flash SFT 攻击未启动，且约 189 GiB 表搬运涉及 P5 授权；
- 飞书 §3.3 图号、非 `_fd` exact-f 图、旧图归档、orphan 清理、SVG/PNG 标准五项待决策。

## 本次新增的接管入口

- `docs/plot_scripts/build_figure_index.py`：从仓库内容重建索引；
- `docs/figure-index.html`：可搜索、按上下文/状态筛选、带缩略图的合作者入口；
- `docs/figure-index.json`：机器可读映射，记录图文件、脚本、脚本声明的数据路径、集群提示和本地文档引用；
- `docs/plot_scripts/README.md`：补充生成命令、认领流程和证据纪律。

生成时间的实际扫描结果是 **289 个可视化文件**（191 PNG、89 SVG、9 HTML）和 **77 个绘图脚本**。这与 handoff 中的“278 张”不同；278 应视为 2026-09-12 早先盘点的历史快照，不能继续作为当前文件数。

## 接管时仍需保留的边界

1. `data/` 整体 gitignored；图索引不会把大体积 JSONL/ckpt 提交进仓库。跨集群原始产物仍按 `docs/notes/data/cluster-infra.md` 查找。
2. 图的 `current/outdated/orphan-review` 是整理层状态。结论有效性仍以 `data/runs_fixed/`、`experiment-lines.md`、`experiment-log.md` 和 `claims-ledger.md` 为准。
3. 脚本没有声明原始集群时，索引显示“需按 run 核对”；这比根据文件名猜集群安全。
4. 当前工作区仍有若干未跟踪文件（包括文献 PDF、临时 Mermaid、二维码和一个未提交的分析脚本）；它们不应被自动纳入本次图索引提交。
5. 交接列出的 commit 均为本地 commit，尚未 push。按 `agents.md` P5，push、删除旧图或跨集群搬运大文件仍需单独决定。

## 最小补充信息

若要让我把“接管整个项目”推进到完全可执行状态，最有价值的补充不是重新解释现象，而是确认三件事：

- 是否接受把本地扫描的 289 文件作为新统计，并先不移动/删除任何旧图；
- Engram baseline 是否按 NO-GO 结论关闭；
- shanbin 与 haoran 的联系方式/交付渠道，以及希望我直接生成哪种 handoff（仓库 Markdown、飞书文档或两者）。
