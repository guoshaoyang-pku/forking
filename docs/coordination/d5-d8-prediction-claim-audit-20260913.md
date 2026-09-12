# D5–D8 prediction and claim audit · 2026-09-13

Source: Feishu paper v2 rev 1082; local evidence is restricted to tracked docs and declared run artifacts.

## # 6 · 模型的进一步预测与验证 (line 197)

- L199: 基于第 5 节的缺失质量与局部锐化模型，本节检验改变训练条件后能够推出的进一步预测。各项均区分模型预期、验证设计和实验结果；具体推导、曲线与数值由对应实验补齐。

**Required evidence packet:** prediction equation or operational definition; setting delta; run_id; step/seed; plotted source; claim ceiling.

## ## 6.1 Replay 次数与相干更新 (line 201)

- L203: 固定唯一训练数据而增加 replay，数据的续接覆盖保持不变，模型预测 novel 条件伤害主要随已有证据的相干强化而增长。验证时同时比较各频率层的 novel / seen 贡献、logit margin 与每个 epoch 的 gap 增量，检验不同频率层能否由共同训练幅度对齐。已有记忆被冻结后仍可被 backbone 利用，是区分表写入和后续读出放大的关键对照。
- L205: ［待补：多 epoch 动力学的解析预测及实测对齐；区分有限窗口中的近线性累计、增量衰减和长期平台，给出模型失效的观测条件。］

**Required evidence packet:** prediction equation or operational definition; setting delta; run_id; step/seed; plotted source; claim ceiling.

## ## 6.2 学习率与 table / backbone 更新 (line 207)

- L211: ［待补：学习率—步数—相干更新强度的关系、表侧饱和区间与 backbone 响应曲线；区分已有经验结果和仍未由模型推出的部分。］

**Required evidence packet:** prediction equation or operational definition; setting delta; run_id; step/seed; plotted source; claim ceiling.

## ## 6.3 唯一数据量与 epoch 长度 (line 213)

- L215: 在固定 context 集合、各 $F(i)$ 同时变为 $kF(i)$，且聚合权重与训练幅度保持不变的理想化条件下，缺失质量项给出 $k^{-\alpha}$ 的缩放预测。实际增加独立数据还会改变 context 集合和续接覆盖；在固定总步数下也会改变 replay 次数。因此需要分别对齐总步数与 pass 数，检验数据覆盖改善和重复曝光的竞争是否解释 epoch-length 曲线。
- L217: ［待补：按验证 token 质量加权的积分表达、频率分布改变时的修正，以及固定步数／固定 pass 的配对验证。］

**Required evidence packet:** prediction equation or operational definition; setting delta; run_id; step/seed; plotted source; claim ceiling.

## ## 6.4 表容量、映射与干预收益 (line 219)

- L221: 改变表容量和 context→row 映射，检验局部化程度、碰撞及有效使用率如何影响模型预测。容量的作用应通过这些可观测量连接到 gap，而不能只由 gap–R 的斜率认定其与语料幂律存在确定的指数关系。低频屏蔽与映射刷新进一步检验机制针对性干预是否既缩小额外 gap，又改善绝对验证表现。
- L227: # 7 · 补充实验（预留）
- L229: 本节预留两项补充实验，用于检验机制的跨结构适用性与后训练风险。有限续接覆盖、局部预测环境和重复强化提供共同的检验线索，但不预设任意模型均会复现同样的现象。
- L231: ## 7.1 DeepSeek-like 模型的复现与干预（预留）
- L235: 可进一步比较低频屏蔽和熵正则等候选措施，评价其是否保留原有训练收益并改善绝对 val loss；这些是待验证的方案，不预设对具体模型已有净收益。［待补：基座与模块版本、与标准 setting 的对应、复现曲线、干预图表及适用边界。］
- L237: ## 7.2 小数据 SFT 压力测试（预留）
- L239: **后训练压力测试。**以 DeepSeek-like／Engram-style 基座为对象，构造少量共享实体或局部 context 的 SFT 样本，反复训练其中一种续接，同时保留训练前模型能够合理预测的其他续接作为测试。对比表冻结与可训练条件，并以无相应局部注入、不同重复次数或更完整续接覆盖的设置作为对照，检查局部结构是否放大替代续接的损失。这样的“攻击”用于检验机制，不预设现有商业模型已存在同样缺陷。
- L241: 评价同时记录目标续接、未训练替代续接、无关 context 和整体验证集的变化。若伤害主要集中于相关局部环境，并随 replay 增强、随机制针对性干预减弱，才支持将其归因于本文讨论的局部锐化；只有整体性能下降不足以排除一般 SFT 遗忘。预训练形成的局部结构是否会在表冻结后继续介导这一效应，是该实验需要回答的重点。以下风险矩阵与研究目标均属于待检验设计。 备用引用：[27] SFT 记忆早期检测；[29] 提取攻击方法学；[26] 重复收益成立条件。
- L263: ［待补：SFT 数据构造、冻结与可训练对照、目标／替代续接的概率变化及整体能力评估。上述画板保留为待验证的风险设计；本节结论及 §8 中对应的后训练讨论，以实际实验支持为准。］
- L271: Forking 及其类似的overfiting现象具有特征：在重复访问训练样本之后，训练与验证损失快速分离，并在 epoch 边界附近及后续遍历中呈现系统周期性阶梯分化。在本文的机制解释中，有限采样造成的续接覆盖不足、局部预测环境的形成和重复样本上的相干更新共同放大了分叉。对类似异常，从本文的研究来看，可以重点关注与过拟合现象同步变化的观测量，并仔细研究数据集的特征，从而做出具体的应对措施。具体来说，通常需要定位具有localize特征的模块，并根据数据集特征对这些作用路径进行处理：补充续接覆盖、限制低频局部读出的影响，或约束重复更新中的过度锐化。
- L275: N-gram 注入使不同 context 获得较为独立的局部特征，从而将原本共享的预测问题分割为更细的优化环境。这里的“子空间”由 n-gram context 与 backbone context 共同决定，在两者的共同作用下这种小子空间数量急剧增加，在这种小子空间中反复优化会使该环境内的预测分布向已见续接锐化，同时压低未被训练覆盖但仍合理的续接。这为严重的局部过拟合提供了一种产生方式，也提示可区分容量的增加需要与续接覆盖和概率校准共同评估。
- L277: 这一机制给出了针对带有 Engram 类记忆模块的 DeepSeek-like 模型的一种“攻击”思路：在受控训练中选择具有多个合理续接的 context，用少量重复样本强化其中一种续接，再测量其他合理续接的概率和损失变化。攻击目标是暴露局部预测被过度锐化的脆弱性，而不仅是制造总体任务分布偏移。对于采用更大查找容量或其他局部化模块的模型，关键挑战是防止局部训练收益掩盖严重的续接损伤；是否能在具体模型上实现这一攻击，以及影响范围有多大，需要匹配的实证检验。 备用引用：[18] 生产 over-encoding 实现；[17] PPL 与下游解耦；[29] 攻击方法学；[34] double descent 背景。
- L293: 要将这些技巧迁移到新情境，需要更高层面的、带有适用条件的规律。例如，续接覆盖、局部化程度和相干更新强度提供了比某个固定学习率或表大小更接近机制的描述。这样的规律应同时解释已有结果并给出可证伪的预测，使研究者能够判断哪些训练条件变化会保留收益，哪些变化可能触发失效。
- L297: 本工作的流程把自动研究中的异常转化为可分析的对象：从内部观测和 loss 轨迹提出问题，通过针对性干预定位组件，再将复杂训练程序约化为最小系统，最终用机制模型组织预测与验证。OPHIS 所强调的观测、假说和干预为这一过程提供了方法背景。[[12]](https://meta-circle.com/blog/ophis-a-new-paradigm-for-autoresearch) 观测量筛选减少了候选范围，而干预和最小重建使内部相关性能够接受因果检验。 备用引用：[42] AI Scientist；[43] MLE-bench；[44] speedrun 复现基准；[2] 本工作起源运行。
- L303: 当前证据仍主要来自小模型、有限数据池和较强 replay 的设置，128× 主线以单 seed 为主；seen / novel 的贡献分解主要在 bigram 分支得到支持。跨 seed、数据规模、tokenizer 与模型结构的稳定性，需要逐项验证。理论上，epoch length 的定量积分、学习率响应以及有限窗口内近线性增长与长期动力学的关系尚未闭合。大模型和 SFT 压力测试的结论取决于对应实验，不能由小模型结果直接代替；条件续接的幂律指数也不应预设为跨语料的普适常数。
- L305: # Appendix（结构预留）
- L313: 数据来自 Hugging Face 的 karpathy/climbmix-400b-shuffle，使用编号 1–10 和 6542 的文本 shard。在首次发现现象的配置中，shard 1–3 用于训练，shard 4–10 和 6542 用于验证。这是当时小数据实验的划分。训练加载器按固定顺序读取并打包文本，遍历训练 shard 后循环重启；三个训练 shard 不足以提供 2000 步互不重复的数据，因此训练会进入后续 epoch。初期训练损失和验证损失大体同步下降；当已见文本开始被 replay 时，两条曲线在接近 epoch 边界的位置迅速分离，训练损失继续下降，验证损失停滞或升高。
- L327: | 预算与记录 | 四组筛选实验均为 1000 步、随机种子 42。验证损失每 5 步评估 4 个 batch；优化器 observables 通常每 5 步记录，在 320–380 步窗口逐步记录，并包含参数统计和逐层统计。 |
- L338: | TR2：双 shard 顺序 replay | 使用 shard 1、2，遍历后循环；其余训练配置与 TR1 相同。 | 约 350 步时仍处于首次遍历，训练与验证曲线基本重合；第二个 epoch 从第 683 步开始，分叉相应推迟。它帮助区分固定训练时刻的变化和随数据重用移动的变化。 |
- L340: | 50-new / 50-replay | 从 shard 1 的基础数据流取连续 50 个新 batch，再按原顺序重放刚缓存的 50 个 batch，循环执行。 | 进入 replay 区间时，训练损失快速下降、验证损失升高；恢复新数据时，两条曲线重新接近。这一设置产生约 100 步的交替周期，用于筛选跟随 replay 状态变化的指标。 |
- L342: TR1、随机采样和 50/50 实验的验证 shard 为 2–10 和 6542，TR2 的验证 shard 为 3–10 和 6542。因训练 shard 的选择不同，这两套验证集合并不完全相同；筛选关注内部观测量的时间响应，损失曲线用于标记各自运行中的分叉。50/50 加载器在基础数据流的 epoch 尾部会缩短不足 50 个 batch 的块，并只重放实际缓存的 batch，因此跨 epoch 的局部周期可能短于 100 步。
- L346: ![图A.1为四组数据重用对照的实验结果图，对应文档中TR1、TR2及相关实验的验证内容。该图由四个子图构成，分别展示不同实验设置下交叉熵损失随优化器步数的变化情况：左上角TR1子图对比了训练损失与验证损失的曲线，可见训练损失在约700步后大幅下降，验证损失则呈上升趋势；右上角TR2子图中，训练与验证损失曲线整体变化幅度较小；左下角Replacement子图里训练损失逐步降低，验证损失保持平稳；右下角50-new / 50-replay子图中，两种损失呈现周期性波动，与文档中50/50加载器重放缓存batch的设置相呼应，用于验证数据重放相关的观测结果。](https://feishu.cn/file/UooubOXrookgaExJLRtcBYfvnHb)
- L348: 图 A.1｜四组数据重用对照。蓝线和红线分别为训练与验证交叉熵；灰色阴影为 350±25 步窗口，以及 TR2 的 700±25 步窗口；虚线为日志记录的实际 epoch 边界。50/50 面板展示第 100–600 步，橙色阴影标记 replay 区间。其余面板展示第 100–1000 步。曲线取自存档记录，未作平滑；验证损失在两次评估之间沿用最近一次记录。
- L376: #### A.2.3 人工审查与组件移除验证
- L380: ![这张图片是图A.2，展示了两类代表性n-gram observables的相关实验数据，上排为RMSProp更新向量的均方根幅度，下排为更新向量与参数向量的余弦相似度。图片四列对应四组对照实验，从左到右分别为TR1、TR2、Replacement、50-new / 50-replay，横轴为优化器步数，纵轴分别对应上述两个观测指标，数据呈现出不同实验阶段的数值变化趋势，用于直观展示两组n-gram相关观测指标的变化情况，辅助验证相关研究发现。](https://feishu.cn/file/Nx5TbRUbHoj5CJxctoZc4GCWnBh)
- L384: 随后，在相同的单 shard 顺序 replay 设置、1000 步预算与随机种子下，分别训练完整模型和关闭 bigram/trigram value 通路的模型，同时保留 unigram value embedding。完整模型在 replay 后出现急剧分叉；关闭两条高阶 n-gram 通路后，这一形态消失，验证 BPB 在后续 epoch 继续改善。两组运行的最终 validation BPB 分别为 1.6769 和 1.0768。该移除实验支持 n-gram 通路在此配置下引发 Forking 的判断；消失的是 replay 附近的急剧分叉，并不意味着所有训练—验证差异都归零。
- L388: ［待补：代码版本、CLI、数据划分、online / fixed-probe、raw / net gap、F / f 映射和完整统计账目。］
- L436: | [37] | 未见物种数最优预测 | Orlitsky, A., et al. (2016). PNAS | [PNAS](https://www.pnas.org/doi/10.1073/pnas.1607774113) |
- L445: # 本次修改记录与待核项（2026-09-08）
- L463: **同日补充修订（2026-09-08）。**按后续要求合并 §2 的 n-gram 与 Over-Encoding 介绍，补入数学工具相关文献 [13]–[15]；适度补清 §3–§7 的测量口径、因果解释、数学假设、可检验预测与 SFT 协议，并更正正文两处标准单目标交叉熵写法。§8.3 改为围绕实测频率幂律及条件续接长尾的可能解释。前一轮记录保留为历史记录，其中交叉熵等部分待核项已在本轮处理；原有图表、历史数值和其余提纲尚未全面复核，最终公式、图表与实验结论仍由绍阳确认。本轮没有开展或宣称完成新实验。Introduction、标题、§2 其余段落、§8 其余小节、附录及既有参考文献均未改动；v1、plan.md、英文 TeX 和 Overleaf 未同步修改。
- L465: **后半部分结构调整（2026-09-08）。**将原 §5.4 的预测与验证独立为新 §6，依次组织 replay、学习率、唯一数据／epoch 长度及容量／干预收益；原 §6 的跨模型与后训练内容归入新 §7，分为 DeepSeek-like 和小数据 SFT 两项补充实验预留。原 §7 的工作总结并入 §8 开头，局限并入 §8.6；原 §8 五个讨论方向保留。净验证收益段落及配图移入 §6.4，SFT 风险画板保留在 §7.2；只移动原生资源块，未重建或修改图像和画板内容。正文的相关章节引用随之调整，其他章节、v1、本地 plan.md 和 TeX 未同步修改。

**Required evidence packet:** prediction equation or operational definition; setting delta; run_id; step/seed; plotted source; claim ceiling.

## Current boundaries

- D5 replay: §48–§53 provide bounded replay dynamics and sharpening evidence; exact geometric-to-linear correspondence remains open.
- D6 learning rate: historical sweeps exist, but any new statement must distinguish table LR, backbone LR, optimizer and schedule.
- D7 data/epoch: existing S1 coverage is not a full table-size × dose × epoch crossing; fixed-step and fixed-pass comparisons remain distinct.
- D8 capacity/intervention: report absolute validation and net gap together; collision or reseed interpretations need the declared table mapping and run evidence.
- No section should be marked complete from prose alone.
