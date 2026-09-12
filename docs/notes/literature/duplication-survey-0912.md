# Duplication / Memorization / Statistical Estimation Survey

> Generated: 2026-09-12 | BibTeX: `citations/duplication-survey.bib`

## 1. Deduplication & Data Repetition

| bibkey | 一句话结论 | 与本课题关系 | 分类 |
|--------|-----------|-------------|------|
| lee-etal-2022-deduplicating | C4 等数据集中大量近重复样本导致 LM 逐字复制训练数据；去重后记忆率降 10×、收敛更快 | 直接支撑「replay ⇒ over-encoding」假设；我们的 gap 在 novel continuation 上放大正是去重文献预测的反面 | core |
| muennighoff2025scalingdataconstrainedlanguagemodels | 固定算力下 ≤4 epoch 重复数据 loss 几乎不变，更多重复则 compute 收益衰减至零；提出 data-constrained scaling law | 为我们多 epoch gap 近线性累积提供 scaling-law 对照基线；验证 >4 epoch 后退化加速 | core |
| chudnovsky2026internaldatarepetitiondestroys | 内部数据重复（同一语料多 epoch）比外部重复更严重损害 LM，提出 compute-equivalent loss 修正公式 | 与我们 §52 epoch-axis bug 修复后的观察高度吻合：prefix-slice shard 导致的隐式重复是 gap 累积主因 | core |
| hernandez2022scalinglawsinterpretabilitylearning | 重复数据的 learning curve 可分解为 memorization + generalization 两分量；重复次数增加时 memorization 分量幂律增长 | 为 train/val forking 的 mechanistic 解释提供理论框架；可直接对接我们的 freq-bin 分析 | core |
| slack-al-moubayed-2025-early | SFT/领域适配中 memorization 可在训练早期检测并用 n-gram-aware regularizer 降低 40% | 与我们 n-gram over-encoding 机制直接相关；其 regularizer 思路可作为干预实验候选 | core |
| kopiczko2026datarepetitionbeatsdata | Long-CoT SFT 中多 epoch 重复优于同预算下扩大数据量 | SFT/post-training 阶段 repetition 正面效应的反例；提示 gap 累积可能 task-dependent | credit |

## 2. Memorization Measurement & Extraction

| bibkey | 一句话结论 | 与本课题关系 | 分类 |
|--------|-----------|-------------|------|
| carlini2019secretsharerevaluatingtesting | 提出 canary insertion 方法定量测量序列模型对稀有训练样本的非预期记忆 | 奠定 memorization 量化方法论；我们的 gap 指标可视为其在 n-gram 注入场景下的变体 | core |
| carlini2021extractingtrainingdatalarge | GPT-2 级别 LM 可被提取出数百条完整训练样本；提取率随模型规模幂律增长 | 证明 memorization 是 scale-dependent 的系统性问题；支撑我们关注 table size × epoch 交互 | credit |
| carlini2023quantifyingmemorizationneurallanguage | 提出三条 log-linear 关系量化 LM 记忆程度；记忆率与 perplexity、模型大小、数据频率均呈幂律 | 为我们的 freq-bin gap 分析提供定量对标框架；novel vs. seen 的 gap 差异可直接用其模型拟合 | core |
| feldman2020neuralnetworksmemorizewhy | 通过 influence estimation 发现长尾样本是 memorization 主因；理论上证明学习长尾需要记忆 | 为「低频 n-gram over-encoding ⇒ gap」提供理论基础；long tail = 我们的 novel continuation | core |

## 3. Overfitting / Generalization Background

| bibkey | 一句话结论 | 与本课题关系 | 分类 |
|--------|-----------|-------------|------|
| zhang2017understandingdeeplearningrequires | 深度网络可完美拟合随机标签，传统泛化理论无法解释；generalization 需重新理解 | Intro 背景引用：说明 train loss 低 ≠ 泛化好，与我们 train/val forking 现象一致 | credit |
| kaplan2020scalinglawsneurallanguage | LM loss 随模型/数据/算力幂律下降；预训练通常不过拟合（compute-optimal 范围内） | 「预训练不过拟合」口径来源；我们的工作表明该结论在 replay/n-gram 注入条件下失效 | core |
| belkin2019reconcilingmodernmachinelearning | Double descent：过参数化区域 test error 再次下降，挑战经典 bias-variance 权衡 | 可选背景：我们的 gap 累积不是 double descent，而是 replay-induced overfitting，形成对照 | credit |

## 4. Good-Turing & Missing Mass / Distribution Estimation

| bibkey | 一句话结论 | 与本课题关系 | 分类 |
|--------|-----------|-------------|------|
| good1953populationfrequencies | Good-Turing 估计器原始论文：用频率的频率 n_r 估计未见物种概率 | 理论章核心：GT missing mass ≈ 我们对 novel n-gram 的 gap 贡献估计 | core |
| goodtoulmin1956numbernewspecies | GT 扩展：预测样本量增大后新物种数量及覆盖率增量 | 对应我们多 epoch 下「新增覆盖 vs. 重复覆盖」的分析框架 | core |
| orlitsky2016optimalpredictionunseen | 最优未见物种数预测：可达 t = O(log n) 倍外推精度 | 为 val 中 novel continuation 比例的理论预测提供工具 | core |
| painsky2022convergenceguaranteesgoodturing | GT 估计器的精确收敛速率与 MSE 界 | 理论章技术支撑：GT 用于 n-gram 频率估计时的误差控制 | credit |
| pananjady2024justwingitnearoptimal | Markov 序列下 missing mass 的近最优估计 | 语言是 Markov-like；为 n-gram 序列的 missing mass 提供现代理论 | credit |
| malagutti2024rolengramsmoothingage | n-gram smoothing（含 GT）可转化为神经 LM regularizer，效果可比 label smoothing | 直接桥接 GT 理论与神经 LM 实践；支撑我们将 GT 用于 n-gram over-encoding 分析 | core |
| chen-goodman-1996-empirical | 系统比较各类 smoothing 技术；GT/Kneser-Ney 在 n-gram LM 中的实证基准 | GT 应用的标准参考；为我们在 n-gram table 中使用 GT 类校正提供历史依据 | credit |

---

**统计**: 共 20 篇 | core: 14 | credit: 6
