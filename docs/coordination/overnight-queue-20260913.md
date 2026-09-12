# Overnight execution queue · 2026-09-13

> 由飞书两份 handoff、本地 experiment-lines、experiment-log、plans、figure index 和资源索引合并而成。本文是排程，不把 planned/running 误报成 done。

来源：G4KKwHMWriuHDOkARHCcHWBenwc（rev 2335，0912 待办）；Lyr3wfH8FiIWT6kdQmrc6dGUnEf（rev 1082，论文 v2）。

## 调度原则

- 先执行零 GPU、可逆、不会改变科学口径的任务；它们可以并行。
- GPU 任务只有在登记 run_id、确认 GPU 空闲、同步 commit 和 md5、检查 train/val 不重叠后启动。
- 新 architecture、跨集群大文件、删除已完成 run、改分支和 push 是审批节点。
- 每项必须回收文件、命令、commit、run_id、step、seed 和 claim ceiling；agent 文本不能直接算完成。

## 全部已知 TODO 的统一状态

| ID | 来源 | 工作项 | 今晚动作 | 状态 |
|---|---|---|---|---|
| D1 | 飞书 0912-3 | 全文写人话、符号统一 | `docs/report/paper-v2-review-draft.md` 已补齐 §6–§7/B 的 claim-safe 正文；符号/禁用强断言清单见 `d1-paper-language-symbol-audit-20260913.md`；云端原稿仍保留待补标记 | local draft ready / cloud open |
| D2 | 飞书 0912-4 | shanbin / haoran handoff | 生成两份任务卡，锁定输入、交付、验收和边界 | done (e8f3f5e) |
| D3 | 飞书 0912-6、v2 §7.2 | V4.1-Flash SFT 攻击 | 只做方案和风险矩阵；不搬运 189 GiB 表 | blocked-by-P5 |
| D4 | 飞书 0912-2 | Engram baseline go/no-go | 把 NO-GO 建议写入 backlog，保留 revision contingency | done (e8f3f5e; decision record) |
| D5 | v2 §6.1 | Replay 次数与相干更新 | §48–§53 与预测缺口已审计 | audit complete / evidence open |
| D6 | v2 §6.2 | 学习率与 table/backbone 更新 | 整理 blrabs、table LR、wd/SGD 预测 | ready, GPU gate |
| D7 | v2 §6.3 | 数据量与 epoch 长度 | S1/M5/M6 覆盖与 fixed-step/pass 缺口已列出 | audit complete / decision open |
| D8 | v2 §6.4 | 容量、映射与干预收益 | table-size、mask、reseed、net-val claim 边界已列出 | audit complete / evidence open |
| D9 | v2 §7.1 | DeepSeek-like / Engram 复现 | 只做协议设计和 no-go 依据，不自动下载/训练大模型 | blocked-by-P5 |
| D10 | local T1 | _fixed 数值回填 experiment-log | 远端 summaries 已回收；字段级 config/log 仍需核对后再覆盖 | partial (remote evidence) |
| D11 | local T2 | β₂ bug 记录清理 | 已完成历史口径标记与替代线说明；数字覆盖仍依赖 D10 核对 | partial |
| D12 | local T3 | _fixed 图重生成 | 审计脚本 roots 和 fixtures，只重跑可验证图 | ready, fixture gate |
| D13 | local T4 | experiment-log 结构整理 | 编号修复、标题审阅、目录刷新和 active/history 状态审计完成；数字/表格 review 仍开放 | audit complete / numeric review open |
| D14 | local T5 | ngram5 fallback / bug audit | model.py 绑定已审阅；隔离环境 pytest 26 项通过 | done (runtime/unit QC) |
| D15 | local T6 | M6 大 shard 或限定结论 | 写补跑和限定范围两种方案，不擅自执行 | needs-user-decision |
| D16 | local T8 | 8k–10k no-ngram 长程基线 | 360-1 候选有 8000 行完整 train_log、train/val 不重叠、seed 42；可作为 host-scoped 长程 endpoint，但仍需代码 md5 与曲线人工复核 | evidence candidate / review open |
| D17 | local T9 | 固定 train probe 测 rho | ophis 与 360-2 summary 均明确 4-batch fixed probe、同 hash、每 10 步；360-2 的 1x 日志仅 12 行，2x 两端 gap 不同且均为历史 2× table LR | evidence candidate / conflict review |
| D18 | local T10 | 缩短 epoch 放大 gap | S1 epoch-length 阵列已覆盖相邻科学问题；重新定义问题前不启动新 run | protocol review required |
| D19 | local T12、v2 §6 | causal 剩余回填 | M10/ffq/mix 远端 summaries 已核对并在 experiment-lines §45 回填；其余 causal 小文件仍待检查 | partial / remote evidence |
| D20 | appendix plan | G.2.1 logit sharpening 回填 | 更新 plan-6 的旧待实验标记和交叉链接 | done (f65ab29) |
| D21 | appendix plan | C.4 三轴交叉总结表 | 从现有 CSV 生成提取脚本和首版表 | done (f65ab29) |
| D22 | appendix plan | H.1.3 run_id 索引 | 从 registry/log 生成机器可读索引，并过滤 asset/glob 噪声 | done (9798db9) |
| D23 | figure inventory | 图号、fd、outdated/orphan、SVG/PNG | 生成决策报告，不移动/删除图片 | done (d1656ee / f65ab29) |
| D24 | codebase | 未跟踪 PDF、临时脚本、二维码 | 生成分类和 gitignore 建议，不删除 | done (f65ab29) |
| D25 | docs | 飞书 v2 与本地 report 差异 | 生成 revision/checksum 与待核项差异报告；不写云端 | done (Feishu audit) |
| D26 | tests | pytest、shell、plot smoke、link check | pytest 26 passed；shell/py_compile/figure-index passed；README/resource links now exercised by path audit | done (local QC) |
| D27 | publication | 本地 commit 与 blog 状态 | 已核对本地 branch ahead；不 push，发布仓库差异仍待审计 | partial (P5 gate) |

## 整晚波次

### Wave 0 · 零 GPU并行

D1、D5、D7、D8、D10、D11、D13、D14、D25、D26、D27。D2、D4、D20–D24 已有 evidence packet（见 overnight-execution-status）。每项写独立审计文件或 patch，主 agent 最后顺序合并共享索引。

### Wave 1 · 远端 preflight 与小文件回收

D6、D12、D16、D17、D19。检查三台机器的 SSH、GPU、进程、磁盘、代码 md5 和已有 run；只回收日志、JSON、CSV，不搬运大 checkpoint。

### Wave 2 · 条件 GPU 运行

D16/T8 与 D17/T9 通过 preflight 后并行；D19 只运行已登记且不改变 setting 的补点。run 结束后立即生成 evidence packet 并回填登记。

### Wave 3 · 依赖与审批

D18/T10 等待 T8；D15/M6 等待用户选择补跑或限定范围；D3/D9 等待大文件、模型资源和跨集群授权。

## 最小 evidence packet

queue_id、owner、status、input_commit、run_id/source artifact、commands、output_paths、validation、claim ceiling、next dependency。

## 完成判定

`done` 只表示产物、验证和登记都齐全；`stalled` 表示外部阻塞；`blocked` 只用于明确的用户授权或 P5 门槛。没有 evidence packet 的“完成”退回 planned 或 stalled。
