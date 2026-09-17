# Related Work 段落 + 新颖性声明

更新时间：2026-08-08（证据状态随 8 月 4–7 日实验更新闭环）
对应：`../article_plan.md` §7（相关工作）+ §8（新颖性）；引用全部对应 `docs/literature/references.bib`（编号/作者/年份已复核）。
用法：英文段落可直接进论文 Related Work；中文对照用于内部讨论；证据状态表用于审稿自检。

## 0. 统一语言（全文保持一致）

- **frequency-conditioned gap**：按 n-gram 训练频率分桶的 train/val loss 差（Δ_bucket = E[ℓ_val|bucket] − E[ℓ_train|bucket]）。
- **row collision multiplicity**：映射到同一物理行的不同 n-gram 数量（row load）。
- **gate weight**：hidden-state 对 memory 注入的标量门控。
- **route-specific train/val loss**：对每个路由（exact hot / hashed tail / shared bucket / frozen table）分别计数的 train/val loss。
- 核心论断：**同平均 val loss 背后可能是完全不同的长尾记忆化（long-tail memorization）与碰撞干扰（collision interference）机制**；只有同时报告上述四个量的分析才能区分它们。

## 1. Related Work（英文，可直接用）

### 1.1 Hash-based n-gram memory

> Several recent methods augment Transformers with trainable n-gram lookup tables that address a fixed memory budget by hashing. Over-Tokenized Transformer (Huang et al., ICML 2025) maps 1- to N-gram keys into fixed modulo tables and sums the retrieved embeddings at the input. Engram (Cheng et al., 2026) uses per-order, multi-head hash tables injected at intermediate layers under a hidden-state gate, and shows that natural-language n-grams follow a Zipfian distribution, motivating a multi-level cache hierarchy in which the long tail resides in slower storage. LongCat N-gram Embedding (Liu et al., 2026) uses polynomial rolling hashing with multiple sub-tables and explicitly analyzes hash-collision rates as a function of vocabulary size. N-Grammer (Roy et al., ICLR 2023) hashes n-grams over product-quantized latent codes with per-head universal hashing. Byte Latent Transformer (Pagnoni et al., ACL 2025) adds hash n-gram embeddings over byte n-grams and reports that its earlier frequency-based tables were abandoned because infrequent byte-grams were not represented at all. These methods evaluate capacity in terms of collision rate, table size, or average (usually validation) loss. None of them reports a frequency-conditioned train/validation gap, and none separates n-gram frequency from row collision multiplicity and gate weight.

### 1.2 Collision-free and frequency-selected memory

### 1.2 Injection-position map (Figure 3.1)

Figure 3.1 is the architecture map used in §3.1. It distinguishes an input route (before QKV), a value-stream route (before attention), and an output-residual route (after attention); gated / ungated labels whether a hidden-state-dependent scalar gate is used. Over-Tokenized Transformer/Over-Encoding, N-Grammer, SCONE, and the input-side byte n-gram experiments in BLT operate before the first Transformer block without an Engram-style hidden-state gate. Engram, production DeepSeek-V4.1-Flash, Lngram, and Engram-compatible tensorized or hot-tier variants add a residual branch before self-attention and use a learned gate. Engram’s controlled experiments commonly use layer 2, with a two-layer reference at layers 2 and 6 (1-based); DeepSeek-V4.1-Flash records zero-based engram_layer_ids=[1,14], i.e. the second and fifteenth blocks under 1-based counting, and both sites are gated. Lngram reports layers 2 and 12 in its main comparison. LongCat N-gram Embedding and X-gram are input-side lookup/fusion designs whose primary descriptions do not establish a hidden-state gate. Infini-gram is external suffix-array retrieval and has no Transformer injection layer. Our model is Figure 3.1’s input route: before QKV in the first block, ungated (layer_id=0 in zero-based code).

> SCONE (Yu et al., NeurIPS 2025) selects a budget-limited set of frequent f-grams with exact keys (a minimum count of 5 during discovery), generates their vectors with a shared f-gram Transformer, and serves them at inference from a frozen offloaded key-value store with longest-match fallback. The authors acknowledge that after frequency-based ranking, longer f-grams are rarer and match downstream data less often. Tensorizing Engram / TN-gram (Zhou et al., 2026) avoids explicit hash collisions with a shared low-rank (CP) parameterization across n-gram orders, at the cost of a rank bottleneck. X-gram (Chen et al., 2026) targets the Zipfian long tail directly with frequency-aware hybrid hashing — high-frequency VIP tokens occupy private rows while the tail is routed through alias-sharing buckets — and introduces slot-collapse and row-utilization analyses to quantify under-trained rows. X-gram's frequency analysis, however, is conducted on the training side (activation magnitude, per-row update counts) and does not examine train/validation generalization; row utilization and generalization gap are not equivalent. All of these methods evaluate on average held-out perplexity or downstream accuracy, and none reports route-specific or frequency-bucketed train/validation loss.

### 1.3 Non-parametric retrieval memory

> A separate line augments language models with non-parametric memory: kNN-LM (Khandelwal et al., ICLR 2020) interpolates with a frozen nearest-neighbor datastore; Memorizing Transformers (Wu et al., ICLR 2022) reads external key-value memory at intermediate layers; Infini-gram (Liu et al., COLM 2024) indexes the corpus with a suffix array to answer unbounded-n queries with backoff. Retrieval memory avoids trainable-row collision and sparse-update issues entirely, but it still exhibits train/validation support mismatch, and recent analyses show its gains concentrate on high-frequency tokens (Nishida et al., NAACL Findings 2025) and degrade on reasoning tasks (Geng et al., NAACL 2025). We position Infini-gram as an upper-bound oracle rather than a direct baseline for trainable n-gram memory.

### 1.4 Frequency-conditioned evaluation and data selection

> The gap between frequency and utility is well documented in data selection: importance resampling (DSIR, Xie et al., NeurIPS 2023), perplexity filtering (CCNet, Wenzek et al., LREC 2020), influence-function selection (TRAK, Park et al., ICML 2023; LESS, Xia et al., ICML 2024; FINDR, Zhang & Wang, IJCNLP-AACL 2025, which scores hashed n-grams by influence rather than frequency), and surprisal-based criteria (Levy 2008; Entropy-UID, Shou 2025; RSI, Lv et al., 2026) all treat frequency as an engineering proxy rather than a value measure. On the generalization side, repeated-epoch training is known to induce memorization before overfitting (Tirumala et al., NeurIPS 2022) and multi-epoch degradation under token budgets (Xue et al., NeurIPS 2023), and reward-model overoptimization shows the same pattern under repeated PPO epochs (Gao et al., ICML 2023). We connect these two threads: trainable n-gram memory is a frequency-skewed, per-row parameter store that is updated repeatedly across epochs, so its frequency-conditioned train/validation gap is the natural observable that existing evaluation protocols omit.

### 1.5 Post-training and freezing memory components

> Parameter-efficient and frozen-memory practices provide the template for our recommendation: kNN-LM keeps the datastore read-only; BitFit (Ben-Zaken et al., ACL 2022) shows bias-only updates approach full fine-tuning; LoRA Learns Less and Forgets Less (Biderman et al., TMLR 2024) shows constrained updates forget less under instruction tuning. SCONE itself is applied during SFT of Qwen3 (Yu et al., NeurIPS 2025, Appendix E.3) without studying repeated-epoch dynamics. Our results add a mechanism-level reason to freeze or make read-only trainable n-gram memory during multi-epoch post-training: repeated replay writes accumulate train-specific historical row state that validation probes cannot access.

## 2. 新颖性声明

### 2.1 English (paper-ready)

> **Claim.** Existing n-gram memory methods evaluate capacity in terms of lookup collisions or average loss, without disentangling n-gram frequency, row collision multiplicity, gate weight, or route-specific train/validation loss. We introduce a frequency-conditioned gap analysis for trainable n-gram memory and show that the same average validation loss can hide qualitatively different regimes — long-tail memorization driven by repeated-epoch row writes versus collision interference from row sharing.

**Contributions.**
1. We establish frequency-conditioned gap as the core observable: high-frequency (≥5k repeats) n-grams contribute ≈0 to the train/validation gap, while novel and low-to-mid frequency buckets dominate (largest bucket contribution ≈1.5), and gate/table norms rise in alignment with gap onset.
2. We show that existing n-gram baselines cannot even detect this distinction: a protocol audit of 9 methods (Over-Encoding, Engram, LongCat, BLT, N-Grammer, SCONE, TN-gram, Infini-gram, X-gram) finds no frequency-bucketed train/validation gap analysis in any of them, despite several reporting separated train and validation losses.
3. We causally separate the two candidate mechanisms — long-tail memorization vs. persistent row addressing — by intervention, with three converging lines (all multi-seed): (i) removing the table (injection off, same setting) removes the gap by ~90%, and by ~99% in replay/shard variants; (ii) removing low-frequency fuel — masking low-frequency readout (0–200: −86%, 0–1000: −92%) or remapping rare tokens onto frequent ones as a BPE proxy (T=8000: −67%, still 0.28 at 6 epochs) — collapses the gap; (iii) breaking cross-epoch row identity via per-epoch hash reseeding removes it by ~79% (unified setting) to ~93% (both shells). Row rollback at the epoch-2 boundary and readout masking both remove ~89% of the gap, while freezing gate alone removes only ~19%: frequency structure lives in table row content, not in the gate. Frequency, row load, and gate must be varied independently.
4. We connect the result to practice: trainable n-gram memory should be frozen or read-only during multi-epoch post-training (SFT/RL), consistent with existing evidence that repeated epochs overfit (Xue et al. 2023; Tirumala et al. 2022) and that frozen/constrained memory forgets less (kNN-LM; BitFit; LoRA Learns Less).

### 2.2 中文对照（内部讨论用）

> 现有 n-gram memory 方法按 lookup collision 或平均 loss 评价容量，未区分 n-gram frequency、row collision multiplicity、gate weight 与 route-specific train/validation loss。我们提出频率条件化的 gap analysis，并证明：**同样的平均验证损失背后，可以隐藏完全不同的机制——重复 epoch 行写入驱动的长尾记忆化，与行共享导致的碰撞干扰**。

相对 vbird `article_plan.md` §7/§8 的差异：vbird 主打「trainable n-gram value memory 何时制造 train-only improvement」（writer-local kernel + historical row state）；本块把同一现象的**评价协议批判面**补全——用频率条件化语言把 vbird 的机制结论翻译成文献可比对的诊断语言，并新增两个 vbird 未覆盖的对照：① 评价协议盘点（9 方法无频率分桶 gap 分析）；② 后训练冻结启示的证据链（§1.5）。

## 3. 证据状态表（审稿自检）

| 论断 | 证据 | 状态 |
|---|---|---|
| 高频 bucket（5k+）gap 贡献≈0；Novel/低中频 bucket 主导 | plan-1 §2.2/§9 图（fig_hitcount_stats、Gap contribution 图） | ✅ 已验证 |
| gate/table norm 与 gap 形成时间对齐 | manual.md 7-22；norm 时间序列图 | ✅ 已观察（机制解读待定稿） |
| 无 n-gram → 无 gap | no-ngram 同 setting 3 seeds final gap 0.125±0.001（−90%）；replay6/shard2/shard3 no-ngram −99%（guide §15.6） | ✅ 已验证 |
| 两 epoch 混合 replay 仍有 gap | replay 50-50（article_plan §4.4）；replay4/6 gap 3.47→6.17 剂量（guide §15.6）；epoch 对齐后非单调（manual 08-07） | ✅ 已验证 |
| freeze ngram+gate → 无 gap | P1/P2 3 seeds（guide §12）：e2 回滚 −89%、readout mask −89%、freeze table −50%、freeze reader −59%、gate −19%；ref/rand 行回滚 null | ✅ 已验证（3 seeds） |
| 低频 readout 遮罩 → gap 降低 | P0 3 seeds（guide §15.5）：0–200 −86%、0–1000 −92%、1–5 −26.5%、≥1001 ≈0；svbird F 逐位一致 | ✅ 已验证 |
| 低频 gate=0 干预 → gap 降低 | guide §10 实验5：nanogpt_original −94%；current-shell −58%；exp7 含 novel 失败 → gate 不是有效位点（§12.4） | ✅ 已验证（位点教训） |
| 每 epoch hash 错位（reseed primes）→ gap 消除 | guide §10 实验4：双 shell −93%；统一口径 −79%；toy −72% | ✅ 已验证 |
| BPE/over-encoding 干预 → gap 消除 | guide §16.11 token 重映射代理 3 seeds：T=3000 −43%、T=8000 −67%、长时程 0.28 | ✅ 已验证（代理） |
| toy 2×2：表+低频键是充分组合 | same-order 公平考卷 3.94 vs 1.40/0.000/0.000；剂量 0→94% seen gap 0.005→7.08（guide §16.9/§16.10） | ✅ 已验证 |
| Exact-frequency baseline：gap ∝ 1/freq | guide §18（22.76M distinct context，P50=1，2000 步） | ✅ 已验证 |
| 9 方法均无频率分桶 gap 分析 | 2026-08-02 原文复核（§1.2 / plan-2 §2.3） | ✅ 文献核查完成 |
| 多 epoch 重复 → overfit；记忆化先于过拟合 | Xue et al. 2023；Tirumala et al. 2022 | 🔍 文献证据 |
| 冻结/只读记忆忘得更少 | kNN-LM 2020；BitFit 2022；LoRA Learns Less 2024 | 🔍 文献证据 |
| 检索记忆增益集中高频 token、推理退化 | Nishida et al. 2025；Geng et al. 2025 | 🔍 文献证据 |

## 4. 引用前注意（2026-08-02 已复核）

- Engram 第一作者为 **Xin Cheng**（arXiv:2601.07372），不是「Tian et al.」；正文/参考文献统一用 Cheng et al., 2026。
- SILO = arXiv:2308.04430（不是 2308.06184）；DSIR = arXiv:2302.03169（不是 2305.09777）；Nishida = arXiv:2503.22426；Geng = arXiv:2408.11815。
- 待核实项：Engram/LongCat/TN-gram/X-gram/Lngram/Engram-Nine/RSI/Entropy-UID 均为 arXiv preprint，venue 只写 arXiv + 年份；如投稿时已出正式 venue 需更新。
