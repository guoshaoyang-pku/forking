# Stale status inventory - 2026-09-13

This inventory is a review queue; historical entries remain for provenance. Current coordination status is maintained in the queue/matrix files linked below.

Refresh note (2026-09-13 continuation): D14/D26 are now locally complete (26 pytest tests pass); D10/D11 are partial because 33 cross-host duplicate IDs have conflicting endpoint metrics; the local paper review draft is ready.

Found 36 lines containing status markers across docs. This review pass resolved the stale markers that were misleading inside the active experiment log; remaining entries below are intentionally historical snapshots, explicit queue states, or cloud/archive copies. The duplicated §10 historical snapshot is now explicitly labelled as superseded, and its later v3/s43 result is the authoritative local record for that episode.

- `docs/experiment-log.md:111` 状态约定：`planned` 已登记未开跑 / `running` 运行中 / `done` 已回填 / `stalled` 超期未回填。
- `docs/experiment-log.md:253` 可复用的干净基线。该 section 不启动新训练，不覆盖其他 Agent 的 running
- `docs/experiment-log.md:402` resolved: both historical injection runs are now labelled completed historical runs.
- `docs/experiment-log.md:446` resolved: both historical injection runs are now labelled completed historical runs.
- `docs/experiment-log.md:463` resolved: optimizer wave1/wave2 are now labelled completed historical setting.
- `docs/experiment-log.md:1156` ### 登记（planned → running → done；seed 42 首轮）
- `docs/experiment-log.md:1524` ### 登记（planned → running）
- `docs/experiment-log.md:3368` resolved: the old running snapshot now points readers to §42.6 for the 14/14 terminal backfill.
- `docs/experiment-log.md:3978` resolved: §53 already has a done backfill section; the earlier transition is retained as provenance.
- `docs/coordination/feishu-local-diff-20260913.md:14` - The TODO wiki has 9 unchecked boxes and 17 unresolved markers (`待补`, `待核`, `未启动`, `进行中`, or `预留`).
- `docs/coordination/feishu-local-diff-20260913.md:17` - The TODO wiki still contains historical status text from before the local handoff work: it calls shanbin/haoran handoff “未启动”, C.4/H.1.3 extraction “待生成”, and the logit panel “已由待办 1b 的图 H-1/H-2 填上” in different places. Local evidence now supersedes those historical status lines for coordination, but the cloud page itself has not been edited.
- `docs/coordination/experiment-log-duplicate-review-20260913.md:19` - L869: ## 12. epoch 对齐批（同 epoch 数 × 同 LR-per-epoch 轨迹，2026-08-07 进行中）
- `docs/coordination/experiment-log-duplicate-review-20260913.md:65` - L3947: ## §53 · 置信度极化探针 + 20-epoch logit-sharpening 直接验证（2026-09-12，planned→running）
- `docs/coordination/README.md:9` 状态统一为 planned → running → done；缺少证据时标 stalled，不要用“已分析”代替完成。一个 agent 只负责一个清晰交付单元；共享文档和同一代码文件的写入由主 agent 顺序合并。
- `docs/coordination/README.md:33` 当前状态：planned | running | done | stalled
- `docs/coordination/d1-paper-language-symbol-audit-20260913.md:23` - L459: - v1 中 MiniMax M2.5 事件的具体机制归因未获本轮核实，不能作为本文机制的证据；新 Section 8 未采用这一外部事件。历史“实验进行中”也不代表本轮核准了实时进度。
- `docs/coordination/todo-completion-matrix-20260913.md:8` | Feishu shanbin/haoran handoff | D2 | **done locally** | `handoff-task-cards-20260913.md`; cloud page still has historical “未启动” wording. |
- `docs/coordination/overnight-queue-20260913.md:3` > 由飞书两份 handoff、本地 experiment-lines、experiment-log、plans、figure index 和资源索引合并而成。本文是排程，不把 planned/running 误报成 done。
- `docs/coordination/handoff-task-cards-20260913.md:58` status: planned | running | done | stalled
- `docs/plans/plan-3-fix-and-backfill.md:6` **状态**：2026-08-23 立。代码侧两个 bug 已修复，污染数据已删除；路径与标准口径收口工作进行中。
- `docs/notes/handoff-review-0912.md:5` 这份 handoff 足以让我接管仓库的开发、实验登记、图表生成和证据审计主线。它给出了当前 setting SSOT、权威文档层级、主要 run 与 commit 链、集群坐标、已完成的 benefit/logit/literature/appendix 工作，以及未启动任务。
- `docs/notes/handoff-review-0912.md:9` - 全文中文初稿与符号统一尚未启动；
- `docs/notes/handoff-review-0912.md:12` - V4.1-Flash SFT 攻击未启动，且约 189 GiB 表搬运涉及 P5 授权；
- `docs/skills/skills.md:47` - 按 `planned → running → done` 或 `stalled` 更新生命周期；
- `docs/skills/skills.md:105` ngram-gap-experiment-registration（running → done 回填）
- `docs/_archive/docs/closure-status-20260808.md:18` | `ngram5_freq_gap` full-163 | `DATA_GEN_RUNNING / PREFLIGHT_DONE` | 本地 smoke 只有 20 steps、order=3、vanilla fallback；远端 full-163 data generation 已在运行；另有一个 4-GPU 120-step preflight 已完成，但 contract 指向旧的 `aligned_v2_20260806` 数据，不是 full-163 | 只监控生成日志并等待 `meta.json`；用 full-163 数据目录重新做 DDP smoke，再决定正式长跑 |
- `docs/_archive/docs/closure-status-20260808.md:27` - 注入点 observable 重跑：`injpos_obs_summary.json` 已覆盖 input/y/v 三组，step 10–1000 的 norm/grad 点和 gap 曲线已有本地交付物，不再按“进行中/排队”统计。
- `docs/_archive/docs/ophis-manual-worklog.md:52` - v10 标准重跑（nglab1x_v10_v/y/input/nogram，2000 步）由 ngram-gap-lab 并行 agent 推进中（截至 20:35：v 在 GPU2 ~step220，input/y/nogram 未启动）；blog 数据 `injpos_ablation_data.json` 待 v10 input 完成后用 `build_injpos_data_json.py` 重建（自动变 10 步档）。
- `docs/_archive/docs/ophis-manual-worklog.md:79` - 过程踩坑：上一轮遗留的旧启动器（high 4000 steps 版）曾并发抢写 run_meta/启动 4000-step run，已杀掉 `t5_on_high_s43/44` 旧实例并以 2500 steps 重跑；toy5_launch.sh 的 wait_gpu 已修（只统计 running 的 GPU）。所有 12 个 run 最终均为正确配置（low 2000 / high 2500 steps）。
- `docs/notes/literature/overencoding-survey-0912.md:41` | `zhao2025speedrun` | Automated LLM Speedrunning Benchmark 基于 NanoGPT speedrun 评估 agent 复现能力，当前 reasoning LLM 仍难以复现已知改进 | **credit-assignment**：NanoGPT speedrun 是本课题使用的基线框架（nanoGPT）的来源社区；致谢性引用 | credit |
- `docs/notes/theory/hypothesis-dilution-amplification.md:176` 标定 A 动力学——running（§38）；backbone-LR 判决批 running（§39）；混匀（pass 交错）实验需
- `docs/notes/theory/markov-unigram-exact-gap.md:128` running estimate sees ~tN samples, gap ~ gamma(M-1)/(tN) -> 0). The
- `docs/notes/data/full-corpus-full163.md:49` - **数据集从未生成完成**。卡在数据生成阶段（closure 状态 `DATA_GEN_RUNNING`），
- `docs/notes/data/full-corpus-full163.md:52` - 原计划训练是 4-GPU DDP、70000 步持续预训练，**从未启动**。
- `docs/plans/toy/plan-causal-interventions.md:63` | `nglab1x_input_freeze_table_e1` | ⏸ 未启动 | — | 等 GPU 释放 |
- `docs/plans/toy/plan-causal-interventions.md:64` | `nglab1x_input_freeze_backbone_e1` | ⏸ 未启动 | — | 等 GPU 释放 |
