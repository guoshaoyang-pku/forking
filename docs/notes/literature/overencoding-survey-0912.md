# Over-Encoding Survey — Citation Index (2026-09-12)

> 每篇一行：bibkey | 一句话核心结论 | 与本课题关系（支持/对比/背景）| 分类

## n-gram / conditional memory 模块

| bibkey | 核心结论 | 与本课题关系 | 分类 |
|--------|----------|-------------|------|
| `cheng2026engram` | Engram 将 N-gram embedding 作为 O(1) lookup 的 conditional memory，与 MoE 形成 U-shaped sparsity trade-off；27B 参数 Engram 在 iso-FLOPs 下优于纯 MoE | **直接支持**：我们的 n-gram over-encoding 模块即此类 conditional memory；Engram 的 U-shaped law 与我们发现的 train/val forking 机制互补——他们关注 capacity allocation，我们揭示 finite-sample continuation bias 被 replay 强化的微观机制 | core |
| `zheng2026lngram` | Lngram 在 latent space 学习离散符号做 N-gram lookup，摆脱 tokenizer 依赖；在长上下文 LM 中持续降低 perplexity | **对比/扩展**：Lngram 解决 Engram 对 tokenizer 的耦合，但同样使用 hash-based N-gram table；我们的发现（低频 context 的 bias 被 replay 放大）适用于所有基于 hash table 的 N-gram 模块，包括 Lngram | core |
| `huang2025ott` | Over-Tokenized Transformer 解耦输入/输出词表，input vocab 扩大 128× 可使 400M 模型匹配 1B baseline loss；发现 input vocab size 与 loss 的 log-linear 关系 | **直接支持**：该工作的 Over-Encoding (OE) 即我们研究的 n-gram over-encoding；他们的 log-linear scaling law 是宏观表现，我们的工作解释了微观机制（frequency power-law + Good-Turing + replay amplification） | core |
| `yu2025scone` | SCONE 用独立模型学习 frequent n-gram embedding，offload 到 host memory；1B 加速器参数模型超越 1.9B baseline | **支持**：SCONE 是另一种 n-gram conditional memory 实现，验证了 n-gram embedding 作为 scalable sparse dimension 的有效性；我们的 forking 分析同样适用于 SCONE 的 frequent n-gram 表 | core |
| `lample2019productkeys` | Product Key Memory 通过结构化乘积键实现十亿级参数 memory layer，12 层模型超越 24 层 baseline | **背景**：product key 是 hash embedding 的结构化变体，为后续 N-grammer/Engram 提供基础设施思路；我们的 single-hash clean table 是其简化特例 | core |
| `roy2022ngrammer` | N-Grammer 在 discrete latent representation 上做 bi-gram lookup，以稀疏操作替代 dense scaling，推理更快 | **背景/前身**：N-Grammer 是 Engram/Lngram 的直接前身，首次在 Transformer 中引入 latent n-gram table；我们的 over-encoding 模块继承了这一范式 | core |

## Hash embeddings & compositional representations

| bibkey | 核心结论 | 与本课题关系 | 分类 |
|--------|----------|-------------|------|
| `svenstrup2017hashembeddings` | Hash embedding 用 k 个共享池向量 + 权重插值表示 token，无需预建词典即可处理百万级词汇 | **背景**：我们的 n-gram table 本质是 hash embedding 的特例（k=1, 无插值）；hash collision 与 finite-sample bias 的关系值得进一步分析 | core |
| `bojanowski2017fasttext` | fastText 用 character n-gram bag 表示词，显著提升形态丰富语言和 OOV 词的表示质量 | **背景**：subword compositionality 的经典工作；我们的 n-gram over-encoding 可视为从 subword 到 multi-token context 的自然延伸 | credit |

## Differentiable / hybrid n-gram models

| bibkey | 核心结论 | 与本课题关系 | 分类 |
|--------|----------|-------------|------|
| `damavandi2016nngrams` | NN-grams 将 n-gram counts 作为 NN 输入，结合 memorization 与 generalization；用 NCE 训练避免 softmax | **背景**：早期 neural + n-gram hybrid 工作，验证了显式 n-gram 统计对 NN 的补充价值；我们的模块是这一思路在现代 Transformer 中的延续 | credit |

## Vocabulary scaling / tokenization

| bibkey | 核心结论 | 与本课题关系 | 分类 |
|--------|----------|-------------|------|
| `liu2025superbpe` | SuperBPE 打破 pretokenization 边界学习 superword token，固定 vocab size 下编码效率提升 33%，下游任务 +4.0% | **对比**：SuperBPE 改 tokenizer 本身，我们改 input embedding 层；两者都利用 multi-token pattern，但路径不同。SuperBPE 的 uniform difficulty 发现与我们的 frequency-dependent forking 形成有趣对照 | credit |

## Autoresearch / agent platforms

| bibkey | 核心结论 | 与本课题关系 | 分类 |
|--------|----------|-------------|------|
| `lu2024aiscientist` | AI Scientist 端到端自动生成 ML 论文，单篇成本 <$15，自动 reviewer 达到近人类水平 | **credit-assignment**：本课题的实验管线受益于自动化研究工具生态；致谢性引用 | credit |
| `chan2025mlebench` | MLE-bench 用 75 个 Kaggle 竞赛评估 AI agent 的 ML 工程能力，最佳 setup 在 16.9% 任务达 bronze | **credit-assignment**：ML agent benchmark 背景，与本课题方法论无直接关联但属同一研究生态 | credit |
| `zhao2025speedrun` | Automated LLM Speedrunning Benchmark 基于 NanoGPT speedrun 评估 agent 复现能力，当前 reasoning LLM 仍难以复现已知改进 | **credit-assignment**：NanoGPT speedrun 是本课题使用的基线框架（nanoGPT）的来源社区；致谢性引用 | credit |

---

## 统计

- **总篇数**: 13
- **core**: 8 (cheng2026engram, zheng2026lngram, huang2025ott, yu2025scone, lample2019productkeys, roy2022ngrammer, svenstrup2017hashembeddings, bojanowski2017fasttext→实际标为credit; 修正: core=7, credit=6)
- **credit**: 6 (bojanowski2017fasttext, damavandi2016nngrams, liu2025superbpe, lu2024aiscientist, chan2025mlebench, zhao2025speedrun)

> 注：bojanowski2017fasttext 虽在 hash/compositional 组，但对本课题属背景性引用，归为 credit。
> 最终：**core = 7, credit = 6, 总计 13 篇**。
