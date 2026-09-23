# Plan 7 · ICLR 2027 投稿审阅与修改方案（Forking）

> 审阅对象：刘子鸣老师批注版 `Forking (8)-comment.pdf`（= Overleaf `776bf5a`，24 页，9 条批注）+ Overleaf 最新 `a8dbc5c`（09-23 13:13，Related Work 扩写到 39 条 bib）。
> **硬截止：2026-09-25 23:59 AOE（北京时间 09-26 19:59）**。正文 ≤ 9 页，超页 desk reject。讨论期可改稿到 11-18，届时上限 10 页；12-16 出结果。
> 本文所有数字附 run_id / step / seed（均为 seed 42）；标「假设」的要验证后才能写进正文。
> 配套：`plan-7-assets/refs_existing_corrections.bib`（现有条目修正版）、`plan-7-assets/refs_candidates_verified.bib`（39 条已核验候选文献）。

---

## 0. 总判断

- 题材本身够 oral 级别：一个有名字的现象（forking）；一个正在进入生产的模块（DeepSeek-V4.1-Flash 的 Engram memory 196B 参数，modded-nanogpt、Parameter Golf 的 BigramHash）；一个能定量预测的机制（Good–Turing 缺失质量给出 gap ∝ F^-α，指数只由语料统计决定，与模型实测相差 ≤0.05）；一个实际后果（大语料预训练里看不见，post-training 里重新出现）。
- 离 oral 的差距主要不在"缺大实验"，而在三点：
  1. 叙事像教科书，读者到第 5 页才知道这件事为什么重要。
  2. Figure 1 没有传达核心反差：同一个模块，数据不重复时是好东西，一重复就出事。
  3. 有几处审稿人一眼能抓住的证据漏洞：Section 5 没有 no-gram 对照；全文单 seed；表冻结曲线有锯齿；online train loss 的口径没交代。
- 另有两个 P0 合规风险会直接 desk reject：匿名性（OPHIS 博客），以及缺少 AI use statement。
- 时间只剩约 2.5 天。提交前只做四类事：P0、叙事与 Figure 1、Related Work、几条便宜但关键的实验。其余放到讨论期（到 11-18，可改到 10 页）。

---

## 1. P0：不处理会被 desk reject 或严重扣分

| # | 问题 | 证据 | 改法 |
|---|---|---|---|
| 1 | **匿名性** | OPHIS 博客（meta-circle.com，2026-07-23）公开列出 14 位作者，包括 Shaoyang Guo 和 Ziming Liu，并有一节 "On the discovery of the 'forking' phenomenon"。最新 Related Work 写 "We discovered forking during OPHIS-guided autoresearch" 并引该博客，等于自曝身份。附录 A.2.2 描述的 OPHIS 内部统计量是博客里标为 proprietary 的内容，也暴露了内部人身份。附录 F 还提到 "Chinese draft / revision 2633"。 | 全文用第三人称："Forking was first observed during an automated architecture search on NanoGPT (Yu et al., 2026)"；发现过程用 "the agent / the search" 叙述，不写 "we discovered"。A.2.2 只保留博客公开过的内容，或改为泛化描述。删掉附录 F 的迁移说明。代码用匿名仓库。ICLR 规则：自己的公开预印本可以引，但必须第三人称。引用方式请和刘老师确认。 |
| 2 | **AI use statement 必须有** | ICLR 2027 Author Guide 规定它是必需项，不计页数，缺失可 desk reject；提交表单里也要勾选。本文的现象本身就是 AI agent 发现的，写作也用了 LLM。 | 在正文之后、参考文献之前加 `\section*{AI Use Statement}`，写清三点：(a) 现象由 LLM agent 的自动架构搜索首次暴露；(b) LLM 在代码、分析、写作中的具体用途；(c) 所有数字和引用都经作者核验。另加一段 Reproducibility Statement（推荐项，不计页数）。 |
| 3 | **参考文献** | 已逐条核验：39 条全部真实存在，URL 全部返回 200。问题如下：<br>• `huang2024mlagentbench` 被引用却不在 bib 里，PDF 会显示 "?"<br>• `cheng2026engram` 的 author 写成 "and others"<br>• `lambert2024tulu3` 只有 eprint 字段，iclr 的 bst 不识别，打印出来没有任何出处<br>• 15 条标成 arXiv，实际已正式发表<br>• 3 处描述与原文不符（见 §7）<br>ICLR 2026 对幻觉引用一律 desk reject。 | 用 `refs_existing_corrections.bib` 按 key 替换；按 §7 改描述句。 |
| 4 | **附录草稿痕迹** | 附录 B、C、F、G 里有 "Draft note"、"Figure pending"、"the Chinese draft"、"source manuscript" 和长 run_id。 | 提交版全部删除或改成正式表述。来不及写的节宁可删掉，也不要留占位。 |
| 5 | **投稿资格** | abstract 截止日 09-18 已过。 | 郭绍阳今天确认：OpenReview 上已注册、作者列表已定（之后不能加人），且至少一名作者已登记审稿 ≥3 篇（reciprocal reviewing）。 |

---

## 2. 故事主线（为 oral 服务）

### 2.1 一句话 pitch（全文围绕它写）

一个在"数据只看一遍"时有效的模块（n-gram / Engram 查表记忆，已进入 DeepSeek-V4.1-Flash），一旦数据被重复，就会让训练和验证曲线像叉子一样分开，而且验证损失在绝对值上变差。原因有三层：
- 查表把优化切成了互不干扰的"每个上下文一个小问题"；
- 重复训练在这些小问题里不断把概率压到见过的续写上；
- 稀有上下文恰恰最缺续写覆盖。

大语料单遍预训练把问题掩盖了，SFT、RL 这种小数据多遍的阶段又会把它重新打开。

两组已有数据可以直接写进摘要和引言（run 为 `s1v5_128_marm_{both,bigram,trigram,nogram}`）：

| 验证损失 | no-gram | both-table | bigram-only | trigram-only |
|---|---:|---:|---:|---:|
| step 330（第一遍，未重复） | 5.035 | **4.762** | 4.907 | 6.202 |
| step 1000（约 3 遍） | **3.650** | 5.274 | 4.610 | 6.675 |

- both-table 在第一遍比 no-gram 低 0.27 nats（先帮忙），重复后反过来高 1.6 nats（再伤害），这就是 Figure 1 要讲的反差。注意：trigram-only 在第一遍就比 no-gram 差，见 §6-8。
- 表的大小不是唯一解释：表参数与 backbone 相当时（R=76k 行约 58M 参数，backbone 为 64.6M），step-1000 gap 已经是 bigram 0.33、trigram 0.47，而 no-gram 只有 0.02（`s1v5_128_tbl_{bi,tri}1_R76000`，`s1v5_128_marm_nogram`）。

### 2.2 标题（回应刘老师"标题是不是太弱"）

现标题 "Over-Encoding Causes Over-Memorization Under Data Replay" 有两个问题：over-encoding 是小众术语；"Causes Over-Memorization" 读起来像局部技术结论，看不出生产相关性，也看不出 post-training 后果。候选：

1. Forking: Lookup Memory Turns Data Replay into Overfitting
2. Forking: N-gram Memory Makes Language Models Overfit When Data Repeats
3. Forking: Why Conditional Memory Overfits Repeated Data
4. Hidden in Pretraining, Exposed in Post-Training: Forking in Language Models with N-gram Memory

倾向 1 或 2：保留 "Forking" 作钩子，副标题点出"查表记忆 × 重复数据 → 过拟合"这个普适关系。3 借用 DeepSeek 的 "conditional memory" 术语，传播性强；但正文主体是 hashed n-gram table，需要掂量是否 overclaim。

### 2.3 摘要重写结构（6 句）

1. 钩子：n-gram / Engram 查表记忆正在进入大模型（DeepSeek-V4.1-Flash、speedrun 纪录），在单遍训练中有效。
2. 现象：数据一旦重复，train 和 val 像叉子一样分开，并按 epoch 台阶式扩大，验证损失在绝对值上变差，我们称之为 forking。在 NanoGPT 和 DeepSeek-style + Engram 上都出现。
3. 定位：由 AI 自动架构搜索首次暴露，经 observable 筛查和组件消融定位到 n-gram 表。
4. 规律：gap 集中在低频上下文，随 F 呈幂律，指数可以由语料的 Good–Turing 缺失质量预先算出；表越大越严重，语料越大越轻。
5. 机制：查表给每个上下文一个私有坐标，重复训练在每个上下文内部把概率挤到见过的续写上（crowding-out），而上下文之间几乎互不影响（随机 mask 实验）。
6. 后果：所以常规大语料预训练看不见 forking，但小数据多遍的 SFT 和 RL 式训练会把它重新打开，冻结表只能缓解。这一句依赖 §8 E2 的 no-gram 对照。

注意：现摘要写 "larger training budgets ... suppress it" 是错的。抑制 forking 的是更大的训练**语料**（每个 token 过的遍数更少）；在同一语料上加训练步数反而更严重。改成 "larger training corpora"。

### 2.4 新章节结构与页数预算

目标 ≤ 9 页，写到 8.5–9 页即可，不要为了压到 8 页砍掉故事。

| 节 | 内容 | 页数 | 相对现稿 |
|---|---|---:|---|
| Abstract | 按 2.3 重写 | — | 重写 |
| 1 Introduction | 钩子 → 意外 → Figure 1 → 命名解释 → "Our findings are summarized as follows" 4 条 → 1 段 related work 摘要并指向附录 | 1.3 | 删去教科书式的机制预告段 |
| 2 Tracking Down Forking | 悬疑叙事，见 §4 | 3.0 | 合并现 §3 和附录 A 的精华 |
| 3 Why Forking Happens | 现 §4：coverage、crowding-out、context subspaces | 2.3 | 基本保留，修正数字 |
| 4 Forking in Post-Training | 现 §5；Fig 8 和 Fig 9 合成一张 gap-only 图，加 no-gram 对照 | 1.0 | 压缩 |
| 5 Discussion and Conclusion | 与 MoE 同期工作的关系、实践建议、局限 | 0.5 | 替换 "Future Work" |
| AI Use / Reproducibility Statement | — | 不计 | 新增 |
| Appendix A Related Work（完整版） | 见 §7 | 不计 | 从正文移出 |

腾空间的具体动作：
- Related Work 移到附录：约 −0.9 页。
- Figure 2（注入路线示意）移附录，或缩成 2.3 里的小图：约 −0.3 页。
- Figure 4a 与 Figure 1b 重复，删掉；Figure 4b 并入 Figure 3 作第 4 个 panel：约 −0.4 页。
- Figure 8 只留 gap 行，Figure 9 并入作第 4 列：约 −0.6 页。

新增的 Figure 1 故事版、结论列表和侦探叙事约 +1.0 页，净结果约 8.7–9 页。

建议的 Discussion 实践段（证据已有）：
- post-training 阶段要监控 train/val gap。
- 冻结表只能缓解。
- 屏蔽稀有上下文能去掉大部分超额损失：F<200 屏蔽后 step-1000 val 为 3.79，对照 5.30，no-gram 3.65（`causalv5c_mask_low_f200_e1_128x`，旧 f<200 语义）。但它不比 no-gram 更好，不要写成免费午餐。
- 设计原则层面可以引 SCONE 式的频率阈值 n-gram 词表。

---

## 3. Figure 1 方案

刘老师给了两个方向：极简（只画一条 forking 线，参考 Grokking 的图），或故事版（左边示意，右上不重复得到漂亮曲线，右下重复出现 forking 加震惊脸）。建议做故事版，但右侧两个 panel 必须用真实数据，并且是严格对照。

**左：over-encoding 示意（不是架构图）**
- 内容：最近 n 个 token → hash → 表的一行 → 加到 token embedding。
- 可直接沿用刘老师 ChatGPT 草图的结构，按他的建议用 Keynote/PPT 手搓，字体与正文一致。

**右上：Single pass (no replay)**
- 同一模型在新数据上训练 2000 步，n-gram 的 val 应始终低于 no-gram（over-encoding helps）。
- 旁注："the regime where Engram / speedrun recipes are adopted"。

**右下：Replay**
- 同一模型在 337 个 batch 的语料上重复训练 2000 步（约 6 遍）。
- n-gram 的 train（实线）和 val（虚线）在第一个 epoch 边界分叉，并台阶式扩大；no-gram 的 val 作为淡灰参考线；标注 "forking"。
- 震惊脸可以保留成小图标，也可以只用颜色和注释表达，由你们按审美决定。

**两个右侧 panel 的对照设计**
- 共用 x 轴（0–2000 step）。前 337 步数据完全相同（同一个 shard），之后一条走新数据，一条重复。"a fork in the road" 本身就是叙事。

**需要新跑的 run（§8 E1）**
- 必须用 **both-table**，不能用 trigram-only：trigram-only 在第一遍就比 no-gram 差（见 2.1 的表）。
- 如果新 run 显示 2000 步单遍中 both-table 并不持续优于 no-gram，右上 panel 就不要写 "helps"，改成"新数据上不 fork"，收益改为引用文献。

**Caption（回应"Caption 需要展开"）**：3 句，依次是画了什么、两个 regime 的结论、为什么叫 forking 并指向 Section 2。示例：

> Figure 1: **The same n-gram memory helps without replay and forks with it.** Left: over-encoding adds a hashed embedding of the last n tokens to the token representation. Top right: on fresh data, the model with n-gram memory (blue) reaches lower validation loss than the same model without it (grey). Bottom right: when the same 337-batch corpus is replayed, training (solid) and validation (dashed) losses split at the first replay boundary (dotted) like the tines of a fork, and validation loss rises above the no-memory model. Section 2 traces this to the n-gram table.

**其他要求**
- v/y 注入路线**不进 Figure 1**，放到 §2.3 或附录，用来说明"三种注入方式都会 fork"。
- 保底方案（新 run 来不及时）：用现有 `s1v5_128_marm_both` 和 `s1v5_128_marm_nogram` 的 0–1000 步画极简版。一张图里放 n-gram 的 train/val，加 no-gram val 淡灰线，第一个 epoch 的背景标 "single pass"，同样能讲"先帮后害"。
- Figure 1、4、5、8 在 `paper/FIGURE_PROVENANCE.md` 里是 not found。按仓库规则，新图先在 `docs/figs/` 和 `docs/plot_scripts/` 落地，再上传 Overleaf。

---

## 4. Section 2 悬疑叙事提纲

回应刘老师的建议，以及善斌提出的"在 Section 3 开头加 intro"。

原则：每一小节回答上一节留下的问题，再留下下一个问题，让读者跟着"破案"。定量规律放在"已经知道嫌疑人是谁"之后出场。

**开场段**（善斌的想法）
- 一句话复述现场，加路线图："We first describe what the search saw (§2.1), how the cause was localized (§2.2), then rebuild the phenomenon in a minimal model (§2.3) and map what makes it worse or better (§2.4–2.5)."

**2.1 The scene**
- agent 的目标是最小化 val BPB；在一次小数据实验里，曲线一过 epoch 边界就分开（附录 A.1 的 discovery run）。
- 反常之处：多遍训练本应基本无害（Muennighoff et al. 2023：重复约 4 遍以内，效果约等于新数据）。
- 留下的问题：触发它的是"重复"本身，还是某个时间点？

**2.2 Suspects**
- 四种 replay 调度对照：TR1、TR2（延迟 replay）、有放回随机抽样、50-new-50-replay。分叉跟着"是否在重复"走，而不跟着 step 走。
- observable 筛查（第三人称写成 "an observable screen over parameter, gradient and update statistics"）指向 n-gram 表的 update RMS。
- 去掉 bigram/trigram 路径后 forking 消失：val BPB 从 1.677 降到 1.077。
- 留下的问题：这是那套复杂搜索配置（Muon、门控、2-hash、v 注入）的特殊组合，还是 n-gram 本身的问题？

**2.3 Rebuilding it in a minimal model**
- 设置：vanilla NanoGPT + 单 hash clean table，AdamW/RMSProp；定义 L_tr、L_va、G。
- 必须主动交代 online train loss 的口径：第一遍里它就等于 held-out loss，所以分叉必然从第一次重复开始，台阶高度 = 一次曝光带来的记忆量（见 §6-3）。
- input、v、y 三种注入都会 fork，no-gram 不会；DeepSeek-style + Engram 也能复现（附录）。
- 留下的问题：什么决定 gap 的大小？

**2.4 What makes it worse**
- 四个因素：频率 F（幂律）、表行数 R、语料大小、遍数，对应合并后 Figure 3 的四个 panel。
- 本节 punchline 是"低频上下文贡献最大"，由此自然引出"为什么是低频"。

**2.5 Two interventions**
- 低频屏蔽：F≤8 去掉约 72% 的净 gap（附录 Table 2 口径）。
- 表冻结：冻结后仍然 fork。
- 收尾段列出三个未解问题（为什么是幂律、为什么越重复越糟、为什么非要有表），交给 Section 3。

**章节名**：现 "Forking and Its Empirical Structure" 改为 "Tracking Down Forking"（或 "What Controls Forking"）；现 §4 "Mechanism of Forking" 改为 "Why Forking Happens"。两者仍保持对偶。刘老师读到最后才明白这个对偶，不应该让读者去猜。

---

## 5. 刘老师 9 条批注逐条回应

| # | 位置 | 批注 | 改法 |
|---|---|---|---|
| 1 | p1 Fig 1 | 看不懂 v/y 两条线；只保留一条 forking 线更清楚；简洁比完整重要；caption 要展开 | 按 §3 重做；v/y 移到 §2.3 或附录；caption 写 3 句。 |
| 2 | p1 摘要（高亮） | "larger training budgets … forking goes unnoticed in conventional pretraining" | 这是全文最重要的推论之一，应前移并写准：抑制 forking 的是语料规模，不是训练预算。"heavily crowded tables" 改成 "small tables shared by many contexts"。 |
| 3 | p1 摘要末句 | "那这个很 General 啊，标题是不是太弱？" | 按 §2.2 改标题。末句 "verify even when the table is frozen" 在补 no-gram 对照前不要写这么强（§6-1）。 |
| 4 | p2 | 解释为什么叫 forking | 在引言首次出现处加一句："We call this forking: the training and validation curves, which coincide during the first pass, split apart like the tines of a fork at the first replay boundary." |
| 5 | p2 | 核心结论要醒目，单独一段 1. 2. 3. | 引言末尾加 "Our findings are summarized as follows:"，列 4 条，每条 1–2 句并标节号：(1) 现象与定位；(2) 四条规律（F 幂律、R、语料、遍数）；(3) 机制（coverage × crowding-out × 私有子空间，指数由语料统计预测）；(4) post-training 后果与冻结表。 |
| 6 | p3 | "Empirical Structure" 是什么意思 | 改章节名（见 §4 末），并在两节开头各用一句话点明对偶关系。 |
| 7 | p5 Fig 5b | Tables frozen 为什么突变 | 见下方详解。 |
| 8 | p9 §5 / Fig 9 | Forking 指 overfitting，还是强调 stage-wise？Fig 9 没有明显台阶 | 统一定义：forking = 查表记忆放大的、由重复触发的 train/val 分叉；台阶只是"按 epoch 顺序重放"时的指纹，不属于定义。Fig 9 中每个 prompt 在一个 epoch 内出现约 8.7 次，位置分散，重复与 epoch 边界不同步，所以曲线平滑增长。这与附录 A 的"有放回随机抽样"对照一致（gap 渐增、没有边界事件）。在 §4 加一句解释；摘要 "often grows in a staircase" 里的 "often" 保留。 |
| 9 | p1 摘要 | 高亮 "Low-frequ…" | 同 #2。 |

### 批注 7 详解

已从集群日志核对：`causalv5c_freeze_table_e1_128x` 对比 `causalv5c_none_128x`，seed 42。

| step | 冻结组 train | 冻结组 val | 对照组 train | 对照组 val |
|---:|---:|---:|---:|---:|
| 330 | 4.84 | 4.77 | 4.83 | 4.76 |
| 670 | 2.25 | 5.40 | 3.59 | 4.78 |
| 680 | **5.26** | 5.49 | 3.47 | 4.79 |
| 1000 | 2.11 | **5.56** | 2.57 | 5.30 |

- 锯齿全部来自 online train loss，val 是平滑的。
- 冻结组 step 1000 的 val（5.56）比不冻结（5.30）还差。所以"冻表不能阻止 forking"用 val 来讲，比用 gap 更有力。
- 解释（**假设**，与 §45.3 记录的"单次 self-write 让自身 batch 下降 1.90 nats"属于同一类效应）：
  - 表在第 1 个 epoch 末冻结。此时 epoch 早段 batch 用到的行，已经被后面 336 个 batch 通过共享行覆盖或稀释了（R=2^20 时 bigram/trigram context 碰撞率约 71.5% / 94.5%，§46）；epoch 末段 batch 的行还是"新鲜"的。
  - epoch 2 中 backbone 学会信任表，所以越靠后的 batch 拟合得越好，train 一路降到 2.25。
  - loader 回卷到最早的 batch 时，表里的内容对它们已经是错的，train 跳回 5.26。
  - 不冻结时，每个 batch 每遍都会重写自己的行，所以没有这种位置效应。
- 论文改法：Fig 5b 改画 validation loss（或 fixed-probe gap），caption 加一句说明 online 口径下锯齿的来源。
- 验证方法：用同一 checkpoint 在 epoch 早段和末段的训练 batch 上做 fixed-probe 评估；没有 checkpoint 的话，重跑冻结 run 并加 fixed train probe，约 6 分钟（§8 E4）。
- 附录 Table 2 的 freeze 描述来自另一组实验：M8，在 epoch 2 边界冻结，保留 94% gap（`causalv5m3_freeze_table_e2`）。它和 Fig 5b（epoch 1 边界冻结）不是同一个设置，正文和附录都要写清楚。

---

## 6. 其他审稿人会抓的问题（按严重程度）

1. **Section 5 没有 no-gram 对照（最严重）**
   - frozen-table 的残余 gap（ID 0.341、Magicoder 0.212、OpenR1 0.098、RLVR 0.502）可能只是任何模型做小数据多遍 SFT 都会有的普通过拟合。
   - 摘要里的 "which we verify even when the n-gram table is frozen" 目前没有被支撑。
   - 需要补：no-gram 在同一 20× 语料上预训练 3 遍（20,220 步，按日志约 1.3–1.5 h），再跑同样 4 个 continued-pretraining 源各 5 遍（每个 7–11 分钟）。
   - RLVR-MATH 的 train loss 只有 0.009 / 0.020，而且 4-shot 前缀在所有样本中逐字相同，没有 no-gram 对照很难解读。
   - 如果 no-gram 也出现同等 gap，结论要改为"冻结表去掉了大部分超额 gap"。
2. **单 seed**
   - 几乎所有结论都只有 seed 42。
   - 至少为以下几项补 seed 43/44（每个 1000 步 run 约 6 分钟）：Figure 1 两臂、低频屏蔽、表冻结、Section 5。
   - 来不及可以放到讨论期，但正文要如实写明单 seed。已有的三 seed 证据（X1 optimizer、S1 历史三 seed）要引用上。
3. **online train loss 口径**
   - 第一遍中 online train loss 本质上就是 held-out loss，所以分叉必然从第一次重复开始；台阶高度度量的是"一次曝光记住了多少"。
   - 建议在 §2.3 主动交代。主证据改用"验证损失绝对上升，最终高于 no-gram"（both 5.27 vs no-gram 3.65，step 1000），这一点不受口径质疑。
4. **超参数看起来像是为放大效应挑的**
   - 128× table LR 是按 gap 峰值选的（agents.md §1.1）。
   - 附录要放稳健性图，正文用一句话引用。已有证据：
     - table LR 8×–1024× 下，2000 步 gap 都在 +5.3 到 +6.0 之间（`optv5f_rms_b099_s{8..1024}p0_2k`）。
     - 锁定绝对 table LR 时，backbone LR 从 6e-5 到 2e-3 的净 gap 都有数个 nats（4.0–14.9，终点 epoch 不同，§42 `blrabs_*`）；4e-3 时 gap 塌到 0.035，是因为 backbone 本身训不动（train 6.71）。
     - RMSProp ≈ AdamW（2.727 vs 2.731）；SGD 写不动表（0.047）（§35.5）。
     - 发现阶段的原始配置 table LR 是 0.6，比主线还高。
   - 现附录 C 只有 Draft note，这是最该补的附录。
5. **参数量**
   - both-table 模型共 1,675M 参数，backbone 只有 64.6M，表占 96%。审稿人会说"1.6B 的表过拟合 50M token 理所当然"。
   - 正文要主动给出参数量，并用 R=76k（表约等于 backbone）时 gap 仍是 no-gram 的 14–20 倍来回应。
   - 生产参照：DeepSeek-V4.1-Flash 的 memory:backbone = 196B:552B。
6. **§4.2 文字与新版 Figure 6b 不一致**
   - 图已经画到 5 个 epoch，正文仍在引 epoch 1→3 的数：
     - "7.5 to 9.0"：图上到 11.0。
     - "stays near 3.4"：图上是 3.5→4.2，在上升。
     - "4.3 to 2.8"、"6.9 to 5.6"：也是 epoch 3 的值。
   - "with a step at every epoch boundary" 在只有 epoch 末端点的图上看不出来，改成 "rises at every pass"。
7. **其他措辞**
   - §3.2 "no-gram control stays near zero"：20 epoch 时 no-gram gap 为 1.01（`s1v5_128_ep1xL4_20ep_nogram`），改为 "an order of magnitude smaller"。
   - "Across three decades of table rows" 与图上的 10^1–10^6 对不上。
   - §6 "which we consider reasonably complete" 删掉，自评式表述容易招审稿人反感。
8. **trigram-only 在第一遍就比 no-gram 差**
   - step 330 的 val：trigram-only 6.20，no-gram 5.04。原因可能是 19M 个 trigram context 挤在 1M 行里，碰撞严重。
   - 而 Figure 3c 语料轴和整个 Section 5 用的都是 trigram-only。审稿人可能会问："一个从来没帮上忙的模块，研究它的过拟合有什么意义？"
   - 两个选项：正文说明这些实验是机制探针，并给出 both-table 的对应结果；或者 Section 5 的关键 run 改用 both-table。
9. **DeepSeek-like 复现没有无记忆对照**：all-read mask 在 step 96 出现 Inf。补一个不带 Engram 的同配置 run，或在正文明确说明这只是 full vs masked 的对照。
10. **同期工作必须引用并定位**
    - Jha, Li, Leskovec, Liang, Zettlemoyer, arXiv 2609.11917（2026-09-10），"Mixtures-of-Experts Overfit More to Repeated Data"。
    - 结论与本文同构：稀疏容量 × 重复 → 更早过拟合；起决定作用的是总参数而不是激活参数；router 早期固化。
    - 按 ICLR 2027 reviewer guide 属于同期工作，不要求比较，但不引用会显得不了解领域。
    - 建议定位（沿用 agents.md §6 的口径）："n-gram 表是这一机制的极端情形：'路由'在第 0 步就由 hash 固定，每行的更新次数就是上下文频率；我们给出频率分辨的定量规律和因果干预。跨架构机制等价仍是待验证的预测。"
    - 这也是提升普适性、回应"标题太弱"最好的素材。
11. **台阶现象有先例**
    - fast.ai（Howard & Whitaker, 2023）报告过 LLM 微调中 train loss 在每个 epoch 边界骤降。
    - 回应：本文的 no-gram 对照没有台阶；n-gram 让 64M 的小模型在预训练级别的数据上就出现了单次曝光记忆。

---

## 7. Related Work 处理方案

**原则**（回应善斌"不确定哪些 work 必要、表述是否合理"）：
- related work 的作用是定位，不是罗列。每段写一类思路：1–2 句概括，再加一句"本文与它们的区别"。
- 不要逐篇复述每篇论文的 headline 数字（如 "1B outperforms 1.9B"）。只有支撑本文动机的数字才写，比如 DeepSeek-V4.1-Flash 的 196B memory。

**正文部分**
- 在引言末尾、findings 列表之后写一段 5–6 句，只引最关键的约 12 篇：
  - over-encoding 谱系：N-Grammer、Over-Tokenized、SCONE、Engram、DeepSeek-V4.1-Flash、speedrun / Parameter Golf / Recursive；
  - 重复数据：Muennighoff、Hernandez、Xue et al. 2023；
  - 同期 MoE 工作：Jha et al. 2026；
  - 经典平滑：Good 1953、Katz 1987、Kneser–Ney 1995、Chen & Goodman 1999。
- 最后一句用刘老师建议的写法："A detailed discussion of related work is deferred to Appendix A due to space."

**附录 A 分段**（约 1.5 页；总引用 40–50 篇即可，已超过刘老师说的 30 篇）

1. **Hashed n-gram embeddings / over-encoding**
   - feature hashing（Weinberger 2009）、FastText、Hash Embeddings、N-Grammer、Over-Tokenized、SCONE、Engram、Memory Grafting、Parameter Golf 的 BigramHash、DeepSeek-V4.1-Flash。
   - BLT：改写成"使用 hashed byte n-gram embedding"，这才是与本文相关的点。
   - modded-nanogpt record #62（2026-01）：逐层加到 residual stream。
   - Recursive：其最佳 NanoChat 方案就是把 hashed bigram/trigram 注入 attention value，正是本文的起点，目前没引。
2. **Memory layers 与检索**
   - PKM、Memory Layers at Scale（Berges et al., ICML 2025）、kNN-LM、RETRO、continuous cache、Memorizing Transformers、FFN as key–value memory。
   - 定位句：这些工作都在单遍或大语料下评估，重复数据下的行为没有被研究过。
3. **重复数据与多 epoch**
   - Muennighoff、Hernandez、Xue、Jha（同期）；去重：Lee、Kandpal 2022、Carlini。
   - 多 epoch SFT：LIMA（15 epochs）、fast.ai；以及 Hyperfitting。
4. **记忆与长尾**
   - Zhang 2017、Feldman 2020、Feldman & Zhang 2020、Tirumala、Biderman、Kandpal 2023（long-tail）。
   - 校准：Guo 2017，对应 §3.2 中 entropy < CE 的现象。
5. **缺失质量与平滑**（本文机制的理论根）
   - Good 1953、Gale & Sampson、Katz、Kneser–Ney、Chen & Goodman、Teh 2006（Pitman–Yor 幂律）、Gnedin et al. 2007。
   - 可以写一句定位：经典 n-gram LM 必须做平滑，正是因为 MLE 把全部概率给了见过的续写；把可学习的 n-gram 表接进 Transformer 并重复训练，重新制造了"未平滑 MLE"的失败，而低频屏蔽相当于 back-off。
6. **自动化研究**
   - MLAgentBench、AI Scientist v1/v2（v1 已有 Nature 2026 版）、autoresearch、Recursive、OPHIS（第三人称）。

**删除与改写**
- 删除 d'Autume 2019（Episodic memory）和 Jin & Ren 2024（What will my model forget）。它们讲的是持续学习里的 experience replay，和本文的"多 epoch 数据重放"是两回事，放在一起会让审稿人觉得概念混淆。若要保留，必须写明两种 replay 的区别。
- 空间紧时可以先删 Xu 2023（Why kNN-LM works）和 Geva 2021，它们与本文关系较弱。
- 以下描述错误必须改：
  - **Hernandez**：原文说重复会损害 copying / induction heads，并导致对重复子集的记忆，不是 "promote copying"。
  - **Over-Tokenized**：hashed n-gram embedding 是加到输入 embedding 上的，不是 "additional input channel"。
  - **Tirumala 与 Hyperfitting**：分开写。Hyperfitting 没有"保持测试性能"的结论，它报告的是 val loss 显著变差、但生成质量提升。
  - **modded-nanogpt**：bigram hash 来自 2026-01 的 record #62，约 290M 参数；"lightweight" 只在算力意义上成立。
- **Engram 的引用版本**：ACL 2026 正式版把模块改名为 "Deep Sparse Embedding (DSE)"，全文不再出现 Engram。如果正文要用 "Engram" 这个名字，就引 arXiv v2（21 位作者），并在附录注明 ACL 版的名称（bib 里两个版本都给了）。

**已核验的 BibTeX**
- `plan-7-assets/refs_existing_corrections.bib`：现有条目的修正版，key 不变，直接替换（21 条，bibtex 0 错误）。
- `plan-7-assets/refs_candidates_verified.bib`：39 条候选新文献，取自 DBLP / arXiv / ACL / Crossref，bibtex 0 错误。合并时注意：
  - 其中 `good1953population` 与现有 key 重复；
  - `DBLP:conf/nips/MuennighoffRBST23`、`DBLP:journals/corr/abs-2205-10487` 与现有的 Muennighoff、Hernandez 是同一篇，不要重复加；
  - DBLP 风格的长 key 建议改成短 key。

---

## 8. 补充实验清单

按性价比排序。每个 run 都按仓库规则登记 run_id，并用 md5 核对代码。

| ID | 实验 | 目的 | 规格 | 耗时（按现有日志） | 提交前？ |
|---|---|---|---|---|---|
| E1 | 单遍 vs 重复 2×2 | Figure 1 右侧两个 panel；"先帮后害" | {both-table, no-gram} × {337-batch 重复, ≥2000 个不重复 batch}，各 2000 步，两臂用同一 val 集，前 337 个 batch 相同。重复臂用 S1 主线数据（train shard 1，val 2–10+6542）；单遍臂的训练池是"shard 1 + 与 val 不重叠的新 shard"，需要改 launcher 的 train shard 列表。这属于数据 setting 变更，要按 `ngram-gap-settings` 登记。 | 4 run，每个约 11 min | 是 |
| E2 | no-gram 20× 预训练 + 4 个 continued-pretraining | Section 5 的对照 | 与 `s1v5_128_ep20_tri_20p0xL4_3ep_v2` 同设置，但不带表，20,220 步并保存 final_model；再用它作初始化，跑 ID、Magicoder、OpenR1、RLVR-MATH 各 1665 步 | 约 1.5 h，加 4 个各约 10 min | **是（最关键）** |
| E3 | 关键结论补 seed 43/44 | 统计可信度 | Figure 1 两臂、mask F≤{1,2,4,8,200}、freeze、4 个 SFT 的 trainable/frozen | 约 20–30 个 run，每个 6–11 min | 尽量做，否则放讨论期 |
| E4 | 冻结表 fixed-probe | 验证 §5 批注 7 的假设 | 重跑 `causalv5c_freeze_table_e1_128x`，记录 epoch 早段和末段的 fixed train probe | 1 run，约 6 min | 可选 |
| E5 | DeepSeek-like 无 Engram 对照 | 补附录 G 缺的基线 | 与 `dscont_v1_*` 同配置但去掉 Engram，2000 步 | 1 run | 讨论期 |
| E6 | 更大规模复现 | 回应"只是小模型" | ≥1B 参数、更接近生产的 memory:backbone 比例；或对开源 Engram 模型做小数据多遍 SFT | 大 | 讨论期（oral 的主要加分项） |

写这份文档时，ophis-gpu 上 GPU 0/1/2/3/7 空闲（4/5/6 在跑），E1–E4 今天就能全部排上。

---

## 9. 分工建议

**郭绍阳（需要判断力的部分）**
- 定标题、摘要、一句话 pitch 和 findings 列表的措辞。
- 定 Figure 1 的最终设计（审美和取舍）。
- 写 §2 侦探叙事的骨架（每小节第一句和最后一句），让善斌填充。
- 决定哪些实验在提交前跑；E2 出结果后决定 Section 4 的结论口径。
- 与刘老师确认 OPHIS 的引用方式（匿名性）。

**俞善斌（执行，按顺序打勾）**
1. [ ] 替换修正版 BibTeX，补 `huang2024mlagentbench`，删掉 bib 里的中文注释行，全文编译确认没有 "?"。
2. [ ] Related Work 整体移到附录 A，按 §7 分段、改错句、删 d'Autume 和 Jin；在正文引言末尾写 5–6 句摘要段。
3. [ ] 全文第三人称处理 OPHIS；删附录 F 的迁移说明，以及所有 Draft note、Figure pending、"Chinese draft"。
4. [ ] 加 AI Use Statement 和 Reproducibility Statement（放在正文之后、参考文献之前）。
5. [ ] 按刘老师批注修改：命名解释句、findings 列表、章节改名、Fig 9 台阶解释、Fig 5b 改画 val 并补 caption。
6. [ ] 修正 §4.2 与 Figure 6b 的数字；修改 §3.2 的 "near zero" 和 "three decades"；摘要 "training budgets" 改为 "training corpora"。
7. [ ] 图版面：删 Fig 4a，Fig 4b 并入 Fig 3；Fig 8 只留 gap 行，Fig 9 并入；Fig 2 移附录。
8. [ ] 用现有数据补附录 C 稳健性（table LR 8×–1024×、backbone LR、optimizer、行宽 d）。每张图都在 `docs/figs/` 和 `docs/plot_scripts/` 落地。
9. [ ] E1、E2 出结果后：画 Figure 1 右侧两个 panel，加 Section 4 的对照。
10. [ ] 提交前自检：
    - 正文 ≤ 9 页；
    - PDF 全文搜索 "we discovered / OPHIS / Feishu / Chinese / Draft / pending / TODO / ?"；
    - 正文每个数字都能追溯到 run_id / step / seed；
    - OpenReview 表单勾选 AI 使用；
    - 代码链接为匿名仓库。

**讨论期（到 11-18，可改到 10 页）**：E3 剩余的 seed、E5、E6，附录完善，回应审稿。

**关于 plan-6（附录结构）**：它是按中文稿设计的，有 8 个大节、50 多张图，提交前做不完，也没必要。toy / synthetic 线已退役（见 `experiment-lines.md`），不要进论文。附录保持精简，建议顺序：A related work → B discovery → C setting → D robustness → E interventions → F derivations → G DeepSeek-like → H post-training。审稿人不需要读附录，但会用它来查核正文。
