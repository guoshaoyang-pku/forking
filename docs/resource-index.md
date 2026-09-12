# Resource index

> 文章阶段的资源总表。这里记录去哪里找和能不能作为当前证据，不复制实验数字。

更新时间：2026-09-13。权威 setting、冲突处理和安全边界仍以 agents.md 为准。

## 先按问题找入口

| 问题 | 入口 | 备注 |
|---|---|---|
| 当前 setting / 禁用项 / 集群规则 | agents.md | 唯一 SSOT |
| 实验属于哪条线 | experiment-lines.md | 全景、状态、图和 run 前缀 |
| 某个数字从哪里来 | experiment-log.md | 登记簿，按 section 查 |
| 一句话能否写进文章 | claims-ledger.md | SUPPORTED / PROXY / UNRUN 边界 |
| 论文正文 | report/index.html | 本地只读副本；发布源在 blog repo |
| 本地正文审阅稿 | [report/paper-v2-review-draft.md](report/paper-v2-review-draft.md) | 已将飞书 v2 的待补段落改成 claim-safe 文字；不覆盖发布源 |
| 论文 TODO 覆盖表 | [coordination/paper-todo-coverage-20260913.md](coordination/paper-todo-coverage-20260913.md) | 逐项区分 local ready、cloud open、design only 和 gated |
| appendix 章节规划 | plans/plan-6-appendix-structure.md | C–H 结构与图片分配 |
| 多 agent 协作 | coordination/README.md | owner、状态、回收和证据包 |
| experiment-log 目录 | coordination/experiment-log-toc.md | 由日志顶层 headings 生成的导航 |
| 飞书↔本地 TODO 差异 | coordination/feishu-local-diff-20260913.md | 最新 revision、未解决项与云端写入边界 |
| TODO 完成矩阵 | coordination/todo-completion-matrix-20260913.md | 飞书与本地 backlog 的逐项当前状态 |
| 整晚 TODO 队列 | coordination/overnight-queue-20260913.md | 飞书 + 本地 TODO 的统一排程 |
| 远端 summary 证据 | coordination/remote-summary-recovery-20260913.md、remote-summary-cross-host-audit-20260913.md | 仅小型 summary.json；跨主机重复 ID 保留原样，未合并 |
| 远端冲突 manifest | [coordination/remote-summary-conflicts-20260913.csv](coordination/remote-summary-conflicts-20260913.csv) | 33 个重复 ID、66 条 host-scoped endpoint 记录；不能取平均或覆盖登记簿 |
| 候选 run 协议审计 | [coordination/candidate-run-protocol-audit-20260913.md](coordination/candidate-run-protocol-audit-20260913.md) | T8/T9 字段级 probe、日志完整性与历史 2× 口径核对 |
| 机器可读 dispatch manifest | [coordination/overnight-dispatch-manifest.json](coordination/overnight-dispatch-manifest.json) | 每项 owner、依赖、并行性、验收和 P5 门槛；不表示后台 agent 已启动 |
| 飞书 marker 审计脚本 | [coordination/refresh_feishu_backlog_audit.py](coordination/refresh_feishu_backlog_audit.py) | 只读刷新两份 wiki 的 revision、checksum 和 unresolved marker；不写回云端 |
| 飞书写回草案 | [coordination/feishu-writeback-proposal-20260913.md](coordination/feishu-writeback-proposal-20260913.md) | 两页 append-only 局部插入方案；需用户批准后执行 |
| blog source audit | [coordination/blog-source-audit-20260913.md](coordination/blog-source-audit-20260913.md) | 本地 report 与 blog target 当前字节级一致；review draft 尚未发布 |
| D1 prose audit | [coordination/d1-paper-language-symbol-audit-20260913.md](coordination/d1-paper-language-symbol-audit-20260913.md) | 符号统一、禁用强断言和本地 draft 状态 |
| 导航链接检查 | [coordination/check_navigation_links.py](coordination/check_navigation_links.py) | 扫描 5 个导航文件的相对链接；当前 `checked=24 missing=0` |
| launcher 口径审计 | coordination/launcher-setting-audit-20260913.md | canonical baseline 与历史/变体 launcher 分开；不自动执行 |

## 代码入口

| 路径 | 角色 | 状态 |
|---|---|---|
| code/train.py | 主线 nanoGPT、优化器、fixed-val、诊断 | current；改动需新 run |
| code/diag_worker.py | fast-diag 异步 CPU 聚合 | current；_fd 运行使用 |
| code/ngram_freq.py | context frequency index | current |
| code/make_ngram_blocks.py | 极简 n-gram block 数据路径 | current |
| code/cluster/run_baseline.sh | 非 table-size 新实验入口 | current；只改一个变量 |
| code/cluster/ | 历史 wave launcher 与环境工具 | 混合；逐个脚本审计后再运行 |
| ngram5_freq_gap/ | alpha 数据干预 + 独立 trainer | current 独立实验维度 |
| tasks/ | 自包含的 toy / theory / synthetic 验证任务 | 各任务自包含 |

## 实验与数据命名

| 名称空间 | 位置 | 用途 |
|---|---|---|
| data/runs_fixed/*_fixed/ | gitignored，本地/远端 | 唯一权威主线 run 产物 |
| data/runs_scaling/ | gitignored | S1 三轴 scaling 产物 |
| data/runs_theory/ | gitignored/远端较大 | replay snapshot 与理论后处理 |
| tasks/*/results/ | tracked | toy / exact-enumeration 结果 |
| docs/appendices/ | tracked | 可审阅的 CSV、报告和附录素材 |

原则：data/runs/ 的旧频率 bug 结果不作为证据；novel（hit=0）没有 train loss，不定义标准 gap；远端原始结果没有回收到本地时，只能标记 remote-only。

## 当前论文素材

| 主题 | 入口 | 证据状态 |
|---|---|---|
| benefit-side CE / PPL | notes/theory/benefit-side-measurements-0912.md | current；R=1 proxy 已明确作废 |
| logit sharpening | experiment-log.md §53；appendices/ls20ep_logit_stats/ | current；context hit-count 分组 |
| sharpening vs CE 图 | figs/main/fig_ls20ep_sharpening_vs_ce.png；figs/main/fig_ls20ep_exemplar_evolution.png | current |
| over-encoding baseline survey | notes/literature/engram-benefit-baseline-survey-0912.md | current；Engram-style reproduction NO-GO 已记录 |
| citation libraries | notes/literature/citations/ | current；core / credit 分开 |
| over-encoding practical implications | notes/literature/overencoding-practical-implications.md | untracked article-phase synthesis；混合证据/机制推断/推测，citation review 后再纳入主引用 |
| citation PDF provenance | notes/literature/pdfs-manifest.tsv | 30 files with SHA-256；PDF collection remains review-only |
| appendix C–H | plans/plan-6-appendix-structure.md | structure ready；C.4/H.1.3 已生成，正文缺口仍单列 |
| figure inventory | [figure-index.html](figure-index.html) / [figure-index.json](figure-index.json) | 当前扫描 290 个可视化文件，区分 referenced/outdated/orphan-review；referenced 仅表示被文档或命名引用，不等于科学验证 |

## 主线实验地图

| 线 | 主要问题 | 权威导航 |
|---|---|---|
| M2 / V5-refresh | injection position、主现象和统一 v5 口径 | experiment-lines.md M2 / V5-refresh |
| M3b / X1 / X2 | table optimizer、LR、β2、row width | experiment-lines.md M3b / X1 / X2；experiment-log.md §27–§35 |
| M8–M10 | replay dynamics、freeze、混匀与长期增长 | experiment-lines.md M8–M10；experiment-log.md §38–§45 |
| S1 | epoch / frequency / clean table-size 三轴 | tasks/s1_scaling_three_axis/、docs/appendices/s1_scaling_three_axis/ |
| §48–§53 | 理论审计、margin、seen/novel、logit probe | experiment-log.md §48–§53 |

## 图表、文献和远端

- 作图规则和数据源：docs/plot_scripts/README.md。
- 当前图：docs/figs/main/、docs/figs/theory/、docs/figs/epoch_scale/。
- 机器可读图片索引：[docs/figure-index.html](figure-index.html) 与 [docs/figure-index.json](figure-index.json)；生成/更新时需保留源脚本与 run_id。
- 附录报告：docs/appendices/README.md。
- 文献核实：docs/notes/literature/engram-benefit-baseline-survey-0912.md；实际影响分析：docs/notes/literature/overencoding-practical-implications.md。
- 主开发仓库：/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab。
- 发布博客仓库：/Users/guoshaoyang/Desktop/workdir/guoshaoyang-pku.github.io。
- ophis-gpu：`/data4/guoshaoyang/ngram-gap-lab`（`/data/home` 为别名；以远端实际挂载为准）。
- 360-1 / 360-2：/data/home/guoshaoyang/ngram-gap-lab。
- 集群细则：docs/notes/data/cluster-infra.md。

## 清理与待决策

- `docs/coordination/overnight-queue-20260913.md` 已提供统一队列；历史 T1–T12 仍保留在 experiment-lines 以维护溯源。
- README、code/cluster/README.md 和部分旧 experiment-log section 曾保留旧 setting；历史数字不自动改写。
- docs/figs/ 约 35 张 outdated、约 216 张 orphan，先归档/引用审查，再决定是否移动。
- 本地 main 持续领先 origin/main（用 `git rev-list --count origin/main..HEAD` 获取实时数量）；citation PDF、临时脚本等未跟踪文件仍遵守 agents.md P5，未经决定不删除或 push。
