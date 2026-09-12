# 多 agent 协作手册

这是文章阶段的协作控制面。它补充 agents.md 的不变规则，不替代实验登记簿和 claim ledger。

## 一个任务必须留下什么

每次分工都要先写清楚：问题（只验证一个问题）、输入（commit、run_id、数据和文献范围）、交付路径、验收条件和状态。

状态统一为 planned → running → done；缺少证据时标 stalled，不要用“已分析”代替完成。一个 agent 只负责一个清晰交付单元；共享文档和同一代码文件的写入由主 agent 顺序合并。

## 推荐工作包

| 工作包 | 输入 | 交付位置 | 边界 |
|---|---|---|---|
| 证据回收 | runs_fixed、experiment log | docs/appendices 或 docs/notes | 不重写原始 run，不补造数字 |
| 作图 | 已登记 run + canonical script | docs/plot_scripts + docs/figs | 不从 prose 手填数值 |
| 文献 | 一手论文、官方代码/页面 | docs/notes/literature | core 与 credit 分开，无法核实时标注 |
| 正文编辑 | claims ledger、已审核图表 | blog repo → copy 到 docs/report | 不把 proxy/observational 升级成因果结论 |
| 数学附录 | theory notes、exact toy results | docs/appendices 或 docs/report/theory.html | 写明适用条件与 claim ceiling |
| 远端实验 | 已登记 setting、固定 commit | runs_fixed + experiment log | 先做 GPU、md5、数据不重叠检查 |

## 实验交接模板

    任务：
    科学问题（唯一变量）：
    run_id：
    代码 commit / md5：
    setting：
    机器 / GPU / 启动时间：
    输入数据与 hash：
    预期产物：
    验收命令或检查：
    当前状态：planned | running | done | stalled
    结论（附 step / seed）：
    claim ceiling：
    下一步：

## 回收标准

主 agent 必须检查文件存在、路径可复现、run_id/step/seed/setting 齐全、数据来源是 _fixed 或明确标为历史/remote-only、脚本读取真实 artifact、图表报告与 experiment log 互相链接、结论没有越过 claims ledger。agent 返回文本只能算待审阅输入，通过这些检查并写入登记簿后才算 done。

## 运行与资源纪律

- 新实验前先用 ngram-gap-settings 审计，再用 ngram-gap-experiment-registration 登记。
- 修改 code/train.py 或测量语义必须新起 run_id，并在同一 commit 后同步到集群。
- 一台 GPU 同时只给一个 agent；启动前 nvidia-smi，跨机后核对 md5sum。
- data、远端 checkpoint 和大 PDF 不通过聊天传递；只传路径、hash、摘要和证据包。
- 不使用 rsync --delete；不删除已完成 run；不在未授权时 push、改分支或搬运超过 1 GB 的文件。

## 当前论文队列

1. 正文人话化、符号统一和 claim ceiling 对齐。
2. Appendix C–H 缺口：C.4 三轴表、H.1.3 run index、G.2.1 已完成素材回填。
3. 图注编号、fd 数据源和 outdated/orphan 图片的分批整理。
4. Engram-style baseline 的 NO-GO 决策记录。
5. V4.1-Flash SFT 攻击线保持低优先级，189 GiB 表搬运需单独授权。
