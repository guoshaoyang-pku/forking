# Engram / Over-Encoding Benefit Baseline 文献核实

> 调研日期：2026-09-12
> 目的：为一篇发现「n-gram over-encoding + data replay ⇒ forking 过拟合」的论文，核实『over-encoding 模块确实降低 held-out perplexity』的文献证据，并决定是否需要在 nanoGPT 中复现 Engram-style baseline。
> 方法：逐条对照用户初步 claim 与一手来源（arXiv PDF/HTML、HF 仓库 API、GitHub repo），标注 verified / corrected / unverifiable。

---

## 1. Claim → 一手来源 → 核实结果

| # | 用户 claim | 一手来源 | 核实结果 | 备注 / 准确出处 |
|---|-----------|---------|---------|----------------|
| 1a | DSE (2026) iso-FLOPs 下纯 MoE val loss 1.7248 → 加 20–25% conditional memory 降到 1.7109 | arXiv:2601.07372v2, §3.1, Figure 3 (left) + 正文 p.6 | **Verified** | 原文："in the 10B regime (C=6×10²⁰), validation loss improves from 1.7248 (at ρ=100%) to 1.7109 near the optimum of ρ≈80% (Δ=0.0139)"。注意：这是 **10B 激活参数**（非 27B）regime 的数字；20–25% 指从 MoE 重新分配到 Engram 的最优比例区间（ρ≈75–80%）。 |
| 1b | memory table 扩大 val loss 近似 log-linear 下降 | arXiv:2601.07372v2, §3.2 "Engram under Infinite Memory Regime" | **Verified** | 原文："the curve follows a strict power law (linear in log-space), indicating that Engram provides a predictable scaling knob"。注意措辞是 **power law / linear in log-space**，不是严格的 log-linear（但语义等价于 log-log 线性）。 |
| 1c | MoE-27B internal val 1.634 vs Engram-27B 1.622 vs 40B memory 1.610 | arXiv:2601.07372v2, Table 1 caption + body | **Corrected** | Table 1 caption 写明 Engram-27B 有三个配置：(41k slots, val 1.66)、(46k, val **1.63**)、(50k, val **1.62**)。MoE-27B 的 val loss 在 Table 1 中未直接给出单一数字（表中以 benchmark 为主），但 §4.2 文字描述 Engram-27B "consistently improves over iso-parameter MoE-27B"。用户给出的 1.634/1.622/1.610 与论文 1.66/1.63/1.62 接近但不精确匹配；**1.610 未在论文中出现**，可能来自 v1 草稿或外部摘要。建议引用时使用论文原文数字 1.66/1.63/1.62。 |
| 1d | Pile test 1.960→1.950/1.942 | arXiv:2601.07372 | **Unverifiable / Likely Misattributed** | DSE 论文全文（v1+v2）**未报告 Pile perplexity**。Table 1 报告 MMLU/CMMLU/BBH/HumanEval/MATH/NIAH 等 benchmark；§3 scaling law 实验使用内部 validation set。该数字可能混淆自 Over-Tokenized Transformer 的 Pile val loss 曲线（Fig B.1 中 baseline ~2.2 范围）或 t-hertel 复现的 WikiText-103 PPL。**不应作为 DSE 的证据引用。** |
| 1e | DSE 拿 OverEncoding 做过 baseline | arXiv:2601.07372v2, §3.2 | **Verified** | 原文："while the direct averaging approach of OverEncoding benefits from larger memory tables, Engram unlocks much larger scaling potential from the same memory budget"。OverEncoding (Huang et al., 2025) 被明确作为对照方法讨论。 |
| 2 | DeepSeek-V4.1-Flash HF 实现含 engram.py（multi-order n-gram hash + residual 注入） | HF API `GET /api/models/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference` | **Verified** | `inference/engram.py` 存在（8138 bytes）。配套 `config.json` 含 `engram_layer_ids [1,14]`, `engram_max_ngram_size 4`, `engram_n_heads 8`。README 描述："Layers 1 and 14 each own a hash table of ~384M rows × 256 dims, looked up by 4-gram hashes of the input and written into the residual stream through a learned gate." 表大小 196.6B params (~189 GiB)。 |
| 3 | Over-Tokenized Transformer (ICML 2025, PMLR v267 huang25bb)：input vocab 与 train loss log-linear scaling、≈2x 模型尺寸等效 | PMLR v267 pp.26261–26282; arXiv:2501.16975v2 Fig.1 | **Verified** | Fig.1 left: "OE-12.8M with 400M parameters matches the baseline with 1B parameters"（即 128× vocab → ≈2.5× 等效）。Fig.1 right: log-linear relationship between input vocab size and loss。PMLR 卷号 v267、页码 26261–26282 均正确。注意：该工作称为 **Over-Encoding (OE)**，与 DSE 的 Engram 是不同但相关的路线（OE = tiled matrix parameterization of hierarchical n-gram input vocab；Engram = hashed n-gram embedding + gated fusion）。 |
| 4 | Lngram (arXiv 2605.24869)：conditional n-gram memory 可 post-hoc 注入、joint training 优于 full fine-tuning | arXiv:2605.24869v1 Abstract + §3.3 | **Verified** | Abstract: "effectively injects domain knowledge when added post hoc to pretrained models. Joint training with the backbone further surpasses full fine-tuning"。§3.3 "Lngram-Tuning: Domain Knowledge Injection" 提供实验细节。注意：Lngram 是 **latent-space** n-gram（learned discrete symbols from hidden states），与 Engram 的 token-based hash lookup 有本质区别。 |
| 5 | 非官方 Engram 复现（t-hertel/Engram：0.5B–4B PPL 降 22–30%、benchmark 持平） | github.com/t-hertel/Engram README + results/consolidated_results.json | **Verified (as informal evidence)** | Qwen2.5-0.5B frozen: PPL 20.75→15.73 (-24.2%)；Qwen3.5-4B frozen: 12.87→9.76 (-24.2%)；Qwen3.5-2B full FT: 16.31→11.44 (-29.9%)。HellaSwag/PIQA/ARC-C delta 均在 ±2pp 噪声内。**关键 caveat**: 作者自称 "A Negative Result"，结论是 "Perplexity is not a proxy for downstream performance" 且 "Local statistics ≠ reasoning"。应标注为非正式证据，且其负面结论对我们的 forking 叙事有参考价值。 |
| 6 | 这些论文有没有报告 output entropy / top-k margin / calibration / seen-vs-novel continuation 测量？ | 全文检索 arXiv:2601.07372, 2501.16975, 2605.24869, t-hertel/Engram | **No — 均未报告** | DSE 仅报告 benchmark accuracy + val loss + NIAH + LogitLens/CKA mechanistic analysis；Over-Tokenized Transformer 仅报告 train/val loss + downstream accuracy；Lngram 报告 PPL + CKA + LogitLens；t-hertel 仅报告 PPL + benchmark。**没有任何一篇测量 output entropy、top-k margin、calibration (ECE)、或 seen-vs-novel continuation breakdown。** 这确认了我们机制实验的新颖性。 |

---

## 2. 准确 BibTeX 条目

见 `citations/overencoding.bib`（本 commit 新增）。

---

## 3. Go / No-Go 建议：是否在 nanoGPT 复现 Engram-style 臂

### 3.1 我们的现状

- 主线 nanoGPT 的 n-gram table（bigram+trigram, clean single table, input/wte injection）**本身就是一个 over-encoding 模块**，与 Over-Encoding (Huang et al. 2025) 和 Engram (Cheng et al. 2026) 属于同一概念家族。
- benefit-side 已有 `nogram` 对照臂（无 n-gram table），权威数据见 `experiment-lines.md` v10：input 1.867 vs nogram 0.245（final gap）。
- 我们的核心发现是 **forking 过拟合机制**（gap = val − train 在 replay 下扩大），而非 benefit 本身。

### 3.2 复现 Engram-style 臂的成本估算

| 项目 | 估计 |
|------|------|
| 代码改动 | 需实现 multi-order hash（当前仅 single hash）、context-aware gating（RMSNorm + dot-product gate + depthwise conv）、multi-head concat。预计 200–300 行新代码 + 调试。 |
| 超参搜索 | gate 初始化、conv kernel、hash head 数 K、table size per order。至少 3–5 个 1000-step run。 |
| 时间成本 | 5 runs × 6 min/卡 = 30 min GPU 时间；加上调试/验证口径一致性，预计 1–2 人日。 |
| 风险 | 引入 gate/conv/RMSNorm 违反 P1 极简优先原则；需先在 experiment-log 注册偏离理由。 |

### 3.3 收益评估

| 潜在收益 | 实际价值 |
|----------|---------|
| 外部效度：证明 forking 在 Engram-style 架构下也存在 | **低增量**。DSE/Lngram/t-hertel 已充分证明 n-gram memory 降低 held-out PPL；我们的 nogram 对照已建立同等 benefit baseline。forking 是训练动态现象，与具体 over-encoding 实现正交。 |
| 回应审稿人「你的 n-gram table 太简单，不代表 SOTA」 | **中等**。但可用文献论证（DSE §3.2 明确对比 OverEncoding；t-hertel 证明小模型上 PPL gain 不转化为 downstream gain）+ 我们已有 clean-table 消融。 |
| 发现 Engram-specific 的 forking 模式差异 | **投机性**。无文献先例表明 gate/conv 会影响 forking 动力学；若无预注册假设，属 exploratory fishing。 |

### 3.4 推荐：**NO-GO**

**理由**：
1. **Benefit-side 证据已饱和**。DSE (val loss Δ=0.0139 at 10B)、Over-Tokenized Transformer (128× vocab ≈ 2.5× model equivalence)、t-hertel (PPL -24% to -30%) 三源交叉验证了 over-encoding 降低 held-out loss。我们的 nogram 对照已在本 setting 内建立了等效 benefit 基线。
2. **新颖性在机制不在架构**。我们的贡献是 forking 过拟合的测量与解释，不是提出新的 over-encoding 变体。增加 Engram 臂不会加强核心 claim，反而稀释极简 setting 的可解释性。
3. **成本不对称**。1–2 人日 + 30 min GPU 换来的边际外部效度提升，不如投入到 seen-vs-novel continuation breakdown / output entropy 测量——这正是 Claim 6 确认的**文献空白**，才是我们机制实验的真正新颖性锚点。
4. **P1 合规**。引入 gate/conv/RMSNorm 需偏离极简契约，须先写偏离理由；当前无充分科学动机。

**替代行动**（推荐优先执行）：
- 在现有 input/nogram 两臂上增加 **output entropy + seen-vs-novel PPL breakdown** 测量（零架构改动，仅诊断管线扩展）。这直接填补 Claim 6 确认的文献空白，且与 forking 机制叙事天然对齐。
- 若审稿人明确要求 Engram-style 对照，再作为 revision 阶段的 contingency plan 启动，届时可引用本调研作为「已充分调研、按需启动」的证据。

---

## 4. 文件路径

- 本文档：`docs/notes/literature/engram-benefit-baseline-survey-0912.md`
- BibTeX：`docs/notes/literature/citations/overencoding.bib`
