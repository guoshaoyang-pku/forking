# Handoff: Verdent `绘制logits图` session

日期：2026-09-17  
项目：`/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab`  
原 Verdent session：`1860eb1b-0826-4c0a-8706-716697fbec5d`  
原 session 标题：`绘制logits图`

## 先看结论

这个 session 的本地聊天记录仍然存在，至少 1084 条 `conversation_messages` 可读；问题不是聊天内容被清空，而是 Verdent 的“Retry/复制”依赖的旧 terminal checkpoint/run 已经无法解析。

截图中的错误：

```text
复制失败
resolveTerminalCheckpointByRunId:
unknown run in session 1860eb1b-0826-4c0a-8706-716697fbec5d: 9b2e89fc-...
```

随后单独发送 `Retry` 时，Verdent 将它当成了一个没有前置动作的新请求，因此返回：

```text
This session has no prior action to retry — "Retry" alone doesn't tell me what you want re-executed.
```

不要继续单独点击或发送 `Retry`。请新建一个正常请求，把下面的“可直接粘贴的继续请求”连同本 handoff 一起交给新 run。

## 为什么旧 run 不能继续

- 原 session 本身存在，标题和项目路径正确，状态曾为 `completed`，后续状态出现过 `NETWORK_CONNECTION_ERROR` / `fetch failed`。
- 本地 `checkpoints` 表当前为空；虽然部分消息带有 checkpoint 字段，但对应的 terminal checkpoint 元数据不在本地可解析状态中。
- 旧 run 的 seed 为 `9b2e89fc-366b-4519-8e7f-2dd99ed1d635`；截图中的 truncated id 与它相符。
- Verdent 日志在故障时段记录过：
  - `container probe failed: reason=socket-drop ... result=fetch failed`
  - `ws proxy error: Error: getaddrinfo ENOTFOUND cloudapi.verdent.ai`
  - `container ensure failed (socket-drop): fetch failed`
- 这说明当时是桌面端到 Verdent cloud-container/WebSocket 控制链路的网络/DNS 故障。现在直接访问 `cloudapi.verdent.ai` 已能得到 HTTP 响应，但这不会恢复已经丢失的旧 run checkpoint。

因此，恢复策略是“保留本地聊天记录，使用明确的新请求从已有项目文件和结果继续”，而不是修复 `Retry` 本身。

## 用户原始目标

在 `ngram-gap-lab` 中完成 logits 排名/分布图，展示 native、bigram、trigram、both 四个 arm 在 epoch 1/2/3 的变化，并解释 ngram 约束如何让输出空间变窄、logits 变尖锐，以及 gap 主要来自哪些 continuation/context。

用户已明确的偏好和约束：

- 第一张 logits 图使用线性 y 轴，不加 log；重点看较小 rank 区间。
- 展示 epoch 1、2、3，并保留 train/validation 区分。
- rank 1/rank 2/rank 3 要能看出差异，后续 rank 应逐步被压制。
- 图例和标题必须清楚，caption 不放进图片主体。
- 保存完整 logits 信息，供后续分析；不要只保留截图。
- 需要区分 seen/novel continuation，以及 high-frequency/low-frequency context。
- 主要解释应围绕“见过的 continuation 对其他 continuation 的效应”，并把效果分成 memory、crowding-out、erosion 等分支。
- 本地机器不要重新跑昂贵训练/评估；优先使用已有 JSON、图和远端/基础设施结果。

## 已完成或已形成的结果

### Logits 主图

脚本已经存在：

```text
docs/plot_scripts/plot_marm_logits_rank.py
```

它读取现有 `data/runs_scaling/` JSON，生成：

- `docs/figs/theory/fig_marm_logits_rank.png`
- `docs/figs/theory/fig_marm_logits_rank.svg`
- `docs/figs/theory/fig_marm_logits_rank_share.png`
- `docs/figs/theory/fig_marm_logits_rank_share.svg`
- `docs/figs/theory/fig_marm_true_rank_cdf.png`
- `docs/figs/theory/fig_marm_true_rank_cdf.svg`

当前主图的设计是：4 arms × train/val × epoch 1/2/3，展示 top-10 rank 的平均 softmax 概率，使用线性 y 轴。已有解释性证据显示 replay sharpening 本身是全局现象；ngram arms 还会带来额外的 train sharpening，而 validation 的 rank-1 形状更接近 native，差异更多体现在概率质量被放到哪些 continuation 上。

### Section 4 图表决策

完整图表计划：

```text
docs/notes/theory/section4-figure-plan-0917.md
```

目前确定的结构：

- F4-1：`fig_marm_logits_rank`，主文 logits/rank 证据。
- Appendix E.3：`fig_marm_logits_rank_share` 与 `fig_marm_true_rank_cdf`，作为归一化 head shape 和 true-token rank 的补充/负结果。
- F4-2：保留 `fig_v5_good_turing_kernel.png`。
- F4-3：`fig_marm_gap_by_ctxfreq`，按 context frequency 分箱，展示 paired per-position excess NLL 及 95% CI。
- F4-4：保留 `fig_v5_val_damage_vs_missing_mass_bigram.png`。
- F4-5：`fig_marm_three_effects`，四列分别解释 memory、crowding-out、erosion、safe case，并保留 native control。
- F4-6：保留 `fig_v5_f_by_pass_accounting.png`。
- F4-7：保留 `fig_v5_table_cosine_and_margin.png`。

一个需要在继续工作时确认的对齐事项：F4-3/F4-5 早期使用 trigram-context 轴，而 F4-4/F4-6 使用 bigram-context 轴。已有建议是用 `code/tools/context_freq_split.py` 的 `groups_bigramctx` 重算 F4-3，使主文大部分图统一到 bigram 轴；不要未经检查直接把旧轴标成新轴。

### 已有解释线索

- gap 更可能集中在 novel/low-frequency context，而不是 high-frequency context；这一点需要以已有 JSON/图中的实际值为准，不要只按假设写成结论。
- both@e3 的 novel bin 已有“约 31% token、贡献约 57%”的候选统计，继续时应从原始 artifact 复核。
- “novel continuation ppl 下降而 CE 上升”在数学上不直观，必须检查定义、分母、聚合方向和图中标签后再写进正文；不要默认接受这个方向。

## 关键文件

优先读取：

```text
docs/notes/theory/section4-figure-plan-0917.md
docs/notes/theory/logits-architecture.md
docs/plot_scripts/plot_marm_logits_rank.py
docs/plot_scripts/plot_marm_three_effects.py
docs/plot_scripts/plot_gap_by_ctxfreq.py
```

相关评估/数据工具：

```text
code/tools/eval_logits_rank.py
code/tools/logits_subspace_stats.py
code/tools/context_freq_split.py
code/tools/pick_sharpen_examples.py
code/tools/topk_share_stats.py
```

已有输出常见位置：

```text
data/runs_scaling/
docs/figs/theory/
```

不要丢弃当前工作树改动。诊断时看到的已有未提交修改包括：

```text
docs/experiment-log.md
docs/figs/theory/fig_marm_subspace_both_novel.png
docs/figs/theory/fig_marm_subspace_both_novel.svg
docs/figs/theory/fig_marm_subspace_both_seen_appendix.png
docs/figs/theory/fig_marm_subspace_both_seen_appendix.svg
docs/plot_scripts/plot_marm_subspace_mass.py
```

## 推荐的下一步

1. 先读本 handoff、`section4-figure-plan-0917.md` 和 `logits-architecture.md`。
2. 检查现有 `fig_marm_*` 图和对应 JSON 的 legend、epoch、train/val、seen/novel 标签。
3. 优先修正/补齐 F4-1、F4-3、F4-5；先复用现有 artifact，不重新训练。
4. 若要重算 F4-3，先确认 bigram-context 分组与旧 trigram 分组的定义，再在远端/基础设施运行；不要在本地执行昂贵实验。
5. 对“novel continuation ppl/CE”方向做一次独立数值核对，再决定正文表述。
6. 最后展示最终图的绝对路径，并在 `docs/experiment-log.md` 记录 run/config/artifact/结论边界。

## 可直接粘贴到新 Verdent session 的继续请求

```text
请接着 ngram-gap-lab 的 logits 图任务继续。原 Verdent session 1860eb1b-0826-4c0a-8706-716697fbec5d 的 Retry/checkpoint 已不可恢复；不要依赖旧 run，也不要只发送 Retry。请先读取 docs/handoff-2026-09-17-logits-session.md、docs/notes/theory/section4-figure-plan-0917.md 和 docs/notes/theory/logits-architecture.md，再检查现有 data/runs_scaling/ JSON 与 docs/figs/theory/fig_marm_* 图。不要重新跑训练，也不要在本地重跑昂贵实验。优先完成 F4-1、F4-3、F4-5 的图例/标签/轴对齐和数值核对；若 F4-3 需要重算，先确认 bigram-context 分组定义并给出远端运行命令。特别核对 novel continuation 的 ppl 与 CE 方向，不要把未经验证的候选统计写成结论。完成后展示最终图片绝对路径，并说明每张图对应的数据 artifact、epoch、train/val 与 seen/novel 定义。
```

这段明确请求就是新的可执行动作；它可以替代旧 session 的 Retry。

## 2026-09-17 更新：cell 图文件夹已交付（本 session 完成）

在上述恢复 session 中，用户手绘 3×3（context × continuation）草图已落地为
**文件夹交付**，并经多轮迭代定稿（rank 直方图线性刻度、每 epoch 一张、
novel cont/ctx 也可画、第三列 = 总 ppl/ce 随 epoch 变化）：

```text
docs/figs/theory/marm_cell_figs/
├── A_train_seenctx_seencont/   e{1,2,3}_A_rank.{png,svg} + A_ppl_epoch.{png,svg} + caption.txt
├── B_val_seenctx_seencont/     （同结构；e3 时 +both 挤出已溢出到 seen continuation）
├── C_val_seenctx_novelcont/    （挤出；e3 直方图：+both top-1 预测本身是 train-seen 候选 0.376/0.287 vs native 0.134）
└── D_val_novelctx_novelcont/   （侵蚀；D 无 train-seen 候选故直方图全浅色）
```

绘图脚本（绘图人直接用这个）：

```text
docs/plot_scripts/plot_marm_cell_folder.py
```

关键数值口径（写 caption / 正文前必读）：本组图 = bigram (t-1,t) 轴 +
逐位置 ce_mean（loss 口径）+ 全部 context 频数 f 合并；excess ce 梯子
A −0.9 → B +0.7 → C +3.4 → D +7.1 nats。与三效应图（trigram 轴 +
mean-of-p 口径）不混用。数据源与口径详见 `docs/experiment-log.md`
「Context×continuation cell 图」段与 F4-8 条目。

## 给合作者/绘图人的坐标（代码与数据在哪）

- **仓库（已 push）**：`git@github.com:guoshaoyang-pku/forking.git`，`main`
  分支。权威源 = 已 commit 版本；跨机使用先 `md5sum` 核对（P4）。
- **绘图脚本**（本地仓库内，跑图不需要集群）：
  - `docs/plot_scripts/plot_marm_cell_folder.py` —— cell 图文件夹（本节上图）
  - `docs/plot_scripts/plot_marm_subspace_mass.py` —— 核心逻辑图 `fig_marm_subspace_*`
  - `docs/plot_scripts/plot_marm_three_effects.py` —— F4-5 三效应图
  - `docs/plot_scripts/plot_marm_logits_rank.py` —— F4-1 logits rank 主图
- **统计/评估工具**（只在集群跑）：`code/tools/logits_subspace_stats.py`、
  `code/tools/eval_logits_rank.py`、`code/tools/context_freq_split.py`
- **数据（ophis-gpu）**：`/data4/guoshaoyang/ngram-gap-lab/data/runs_scaling/logits_rank/`
  - `marm_subspace_mass.json` —— cell 图 + 核心逻辑图的数据（4 arm × e1-3 ×
    train/val × bigram 分组：`rank_prob`/`rank_seen_prob`/`ppl`/`ce_mean`/
    `true_prob_mean`，top-10）
  - `marm_seen_novel.json` —— 三效应图数据
  - `s1v5_128_marm_logits_rank{,.json,_epochs.json}` —— F4-1 数据
- **重画 cell 图命令**（任一有 repo + 该 JSON 的机器）：

```bash
python3 docs/plot_scripts/plot_marm_cell_folder.py \
  --json /data4/guoshaoyang/ngram-gap-lab/data/runs_scaling/logits_rank/marm_subspace_mass.json \
  [--cell A|B|C|D]   # 缺省 all
```

- **图表规范**：改图前先读 `.agents/skills/ngram-gap-plotting/`（或 repo 内
  `docs/plot_scripts/README.md`）。结论必须附 run_id/step/seed（P7）。
- 后续许愿由用户手动提给绘图人；新需求改 `plot_marm_cell_folder.py` 或新开
  脚本，不要覆盖已交付图。
