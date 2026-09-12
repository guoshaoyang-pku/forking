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
| appendix 章节规划 | plans/plan-6-appendix-structure.md | C–H 结构与图片分配 |
| 多 agent 协作 | coordination/README.md | owner、状态、回收和证据包 |
| experiment-log 目录 | coordination/experiment-log-toc.md | 由日志顶层 headings 生成的导航 |
| 飞书↔本地 TODO 差异 | coordination/feishu-local-diff-20260913.md | 最新 revision、未解决项与云端写入边界 |
| TODO 完成矩阵 | coordination/todo-completion-matrix-20260913.md | 飞书与本地 backlog 的逐项当前状态 |
| 整晚 TODO 队列 | coordination/overnight-queue-20260913.md | 飞书 + 本地 TODO 的统一排程 |

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
| over-encoding practical implications | notes/literature/overencoding-practical-implications.md | article-phase synthesis；机制推断与实验证据分开 |
| citation PDF provenance | notes/literature/pdfs-manifest.tsv | 30 files with SHA-256；PDF collection remains review-only |
| appendix C–H | plans/plan-6-appendix-structure.md | structure ready；C.4/H.1.3 已生成，正文缺口仍单列 |
| figure inventory | figure-index.html / figure-index.json | 当前扫描 290 个可视化文件，含 current/outdated/orphan-review 分类 |

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
- 机器可读图片索引：docs/figure-index.html 与 docs/figure-index.json；生成/更新时需保留源脚本与 run_id。
- 附录报告：docs/appendices/README.md。
- 文献核实：docs/notes/literature/engram-benefit-baseline-survey-0912.md；实际影响分析：docs/notes/literature/overencoding-practical-implications.md。
- 主开发仓库：/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab。
- 发布博客仓库：/Users/guoshaoyang/Desktop/workdir/guoshaoyang-pku.github.io。
- ophis-gpu：/data3/guoshaoyang/ngram-gap-lab。
- 360-1 / 360-2：/data/home/guoshaoyang/ngram-gap-lab。
- 集群细则：docs/notes/data/cluster-infra.md。

## 清理与待决策

- experiment-lines.md 的历史 T1–T12 与新论文 backlog 仍需合并成一张统一队列。
- README、code/cluster/README.md 和部分旧 experiment-log section 曾保留旧 setting；历史数字不自动改写。
- docs/figs/ 约 35 张 outdated、约 216 张 orphan，先归档/引用审查，再决定是否移动。
- 本地 main 持续领先 origin/main（用 `git rev-list --count origin/main..HEAD` 获取实时数量）；citation PDF、临时脚本等未跟踪文件仍遵守 agents.md P5，未经决定不删除或 push。
