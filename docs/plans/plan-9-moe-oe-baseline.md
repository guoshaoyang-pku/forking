# Plan 9 · OE × MoE：极简 MoE baseline 与 over-encoding 交叉实验设计

> 状态：设计中（2026-10-17）。实现与开跑前须按 P3 注册到 `experiment-log.md`。
> 本文回答两件事：① Qwen3.8 Flash 的 engram setting 查证结果（§1，全部公开来源）；
> ② 小而 solid 的 OE×MoE baseline 规格（§3–§5）。

---

## 1. Qwen3.8 Flash 的记忆 setting 查证：是 engram，且与我们的双支设计同构

**结论（公开证据，非内部情报）**：Qwen3.8-Flash-Next（Qwen4 前瞻架构，2026-08 发布）
明确使用 engram 形态的 n-gram conditional memory。证据链：

| 来源 | 关键事实 |
|---|---|
| 官方设计论文 [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)（Qwen Team, 2026-09-01） | 125B backbone / 6B activated / **+51B n-gram embedding tables**（off-accelerator，host-memory prefetch）；§2.3 专章消融 |
| HF README ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) | **N-gram Embedding: 20,000,000 rows（bigrams/trigrams），Layer 2 单层注入**；512 experts top-10 + 1 shared；MoE expert dim 640 |
| vLLM serving 文档 ([recipes](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)) | 配置旗标就叫 **`--engram-config`**；PLE = 注入 N-gram Embedding 的网络层；table 可 CPU offload |
| NGM 论文相关工作 ([arXiv:2605.16893](https://arxiv.org/abs/2605.16893)) | 「Engram = hashed N-gram lookup + context-aware gating + sparsity allocation framework，27B 规模，deep-layer injection」；LongCat-Flash-Lite 同路线 >30B params |

**与我们 setting 的同构性（逐项对照）**：

| 维度 | Qwen3.8 engram | 我们 v5 主线 |
|---|---|---|
| 分支 | bigram + trigram | bigram + trigram |
| 寻址 | 确定性 hash（multi-head hash） | 单一 hash（clean 表） |
| 注入 | Layer 2 单层，residual stream，contextual gating | input 注入（wte 处求和，无 gate） |
| 规模 | 20M rows × 2560 = 51.2B params | 2^20 rows × 768 × 2 支 |
| 消融（论文 §2.3） | placement Tab 7：w/o 1.585 loss → L2 单层 1.541（下游 L2 最佳）；vocab 20_V→200_V Tab 9：**loss 单调降、下游饱和**（Tab 8/9 不一致）；尝试过 token normalization / 非均匀 order 分配 / 频率分区 slot | freq-index / mask 剂量 / 表尺寸扫描（§76 superposition 塌缩） |

**对我们的三个直接信号**：① bigram+trigram 双支 + 确定性寻址是厂商验证过的主流形态，我们不用改；② 「loss 降、下游饱和」的不一致与我们的 forking 叙事（表容量放大 repetition 过拟合）可以在论文里对表；③ 厂商**没有做**频率/低频治理（mask_low / top-K 治理在论文里只作为 efficiency 尝试被提到，没有 forking 视角）——这正是我们 follow-up 的空间。

## 2. OE × MoE 的研究空间（为什么这个交叉值得做）

1. **MoE 重复数据退化已有现象级证据**（arXiv:2609.11917，agents.md §6）：固定 token 预算下 repetition↑ ⇒ dense 与 MoE val CE 都恶化，**MoE 更早更快**；总参数量/稀疏度比 active 参数更能解释退化；机制 = router 早稳定 + 重复数据反复更新固定 expert 子集。
2. **n-gram 表 = 无 router 版局部 MoE**：hash 确定性路由 + 行级 specialization + 重复曝光下 sharpening（C10 两状态模型、fig7 子空间证据）。两者是同一抽象（agents.md §6.1）的两个实例。
3. **交叉的核心问题**（当前无人回答）：
   - H1 复现：我们的 3-epoch replay 协议下，MoE 是否比 matched-active dense 更早/更快 fork？
   - H2 叠加：加 n-gram OE 后，forking 是否超线性（两个稀疏子空间争夺同一重复输入的 specialization）？
   - H3 交互：router 负载 vs 表命中频率的联合分布（重复 token 是否同时导致 router ossification + 表行 sharpening）？
   - H4 干预通用性：mask_low（表侧）与 aux-loss/expert-dropout（router 侧）是否跨架构互补/可替代？
4. 厂商侧存在性证明：Qwen3.8 = MoE + engram 同模型（125B/512 experts/20M rows），LongCat-Flash-Lite 同路线；但**都没有发 forking/频率治理证据**。

## 2.5 社区 OE 做法调研（2026-10-17）：parameter-golf 与 modded-nanogpt speedrun

### A. modded-nanogpt record 62 · Bigram Hash Embedding（训练内 OE 的小规模权威实现）

PR #201（KellerJordan/modded-nanogpt，val loss 3.2776，−5.6s/−165 steps ≈ 5%）：

- 表：`bigram_vocab_size = 5 × vocab_size`（50304×5 ≈ 251K rows）；hash 公式
  `(p1·prev) XOR (p2·cur) mod R`（rand_int 36313/27191）——**与我们 clean 表第一族公式
  同构，独立收敛**（外部效度）；hash 在 dataloader CPU 端算好（= 我们 plan-8 index-first 的同款思想）。
- 注入：embedding **zero-init**，每层残差加 `λ_l · bigram_ve`（λ 初始 0.1，可学）——
  每层注入而非单点；pos 0 走保留 index；跨文档边界未特殊处理（试过 masking，无收益且更慢）。
- **实现内负结果**：gating、trigram、multi-hash 都没跑赢「直接相加」（ClassicLarry 注明
  单实现 Caveat）。注意这与 Qwen engram（contextual gating + 双支）相反——规模与训练
  budget 不同，不能直接互推；但它支持我们无 gate 的 input 注入主线。
- 数据泄漏检查规范值得抄：随机 token 输入 → loss 平坦于 10.8 附近才算干净。

### B. OpenAI Model Craft「parameter-golf」（16MB 工件、8×H100 ×10min、FineWeb BPB 计分）

分析论文 [arXiv:2607.01517](https://arxiv.org/abs/2607.01517) + issue #140：冠军方案普遍用
**推理时 n-gram blending**（不是训练内 OE）：

- 形态：eval 时对近期 context 在线建 n-gram cache（Katz backoff，2–7 gram），
  `P_mix = λ·P_neural + (1−λ)·P_ngram`（字节级）。
- 三代进化：固定 5-gram α=0.20（−0.07 BPB）→ multi-order backoff 2-7（再 −0.02）→
  **entropy-adaptive λ**（按模型预测熵调整信任，再 −0.02）。neural-only 1.1271 → 0.9674。
- 为什么有效：局部重复是残差主导信号；零工件字节（cache 从 eval 数据现场建，test-time）。
- **关键区别**：固定表不会被梯度更新 ⇒ 不会 sharpen ⇒ 预测「train/val 不分化」。
  **已实测两轮（§79+§80，2026-10-17）**：
  - step-0 冻结**随机**表 gap **0.064** vs 可学习对照 **2.702**（§79）；
  - step-0 冻结 **count-init** 表（行 = 桶内经验续接分布的 init-wte 投影，真实信号
    但不可学习，golf 等价物）gap **0.022**，且 train 3.673 < 随机冻结 3.786、
    val 3.695 < 3.851（§80）——**真实信号改善了绝对损失却没有产生任何分化**，
    信号效用与 forking 解耦。
  用户曾挑战「冻结仍分割子空间、数据少了照样 fork」；两轮实测否证：分区存在、
  信号真实，都不 fork——**forking 需要表在重复数据上被梯度塑形**（训练动力学属性，
  非内容属性）。但既有 causal 臂支持其 backbone 侧直觉的另一半：e1/e2 冻结**已写满**
  的表 gap 3.452 > ctrl 2.81（§72 @128×）/ 5.386 = ctrl 94%（§38 m3）——**发起 fork
  需要表可学习，维持 fork 不需要**（backbone 接管甚至恶化）。判词：点火 = 表侧
  gradient sharpening；燃烧 = backbone 共适应；内容质量与冻结时机均非充分条件。

### C. OE 方法三分类（论文 related-work 的现成骨架；frozen 两种内容均已实测）

| 类 | 代表 | 表参数 | 重复数据下的预测 | 实测 |
|---|---|---|---|---|
| trained-in hash embedding（= engram） | 我们 / DeepSeek Engram / BHE / NGM | 学习 | **forking**（本项目主题） | gap 2.702（§79 ctrl） |
| test-time blending | parameter-golf / infini-gram | 无（或固定） | 不 fork（表不可学习） | 随机冻结 0.064（§79）；**count-init 冻结 0.022 且 train/val 同降（§80）** |
| frozen / reparam 表 | COUNT / MeKi / 我们的 freeze 线 | 固定 | 随机或 count 内容 init：不 fork；**学满后冻结：照样 fork 且可恶化** | 0.064 / 0.022 vs 3.452>2.81（§79+§80+§72）|

> golf 等价性注记（已测）：§80 的 count-init 冻结臂 = golf 的最贴近复刻（真实统计
> 信号 + 不可学习）→ gap 0.022，信号效用（train/val 绝对值都更好）与 forking（无）
> 解耦——「不可学习的表按构造安全」从预测升级为机制证据。

### D. 对 plan-9 的直接影响（§79/§80 后更新）

1. hash 公式与注入哲学（直接相加 vs gating）都有外部先例可引，审稿人问「为什么这样设计」有答案。
2. **廉价对照臂已落地 2/2**：step-0 随机冻结（§79:0.064）与 count-init 冻结
   （§80:0.022）都实测完成——「表可学习性是 forking 发起的必要条件」现在是
   两点证据 + §72/§38 的学满冻结对照构成的完整判词；论文机制章可直接引用。
3. BHE 的 5×V 表规模 vs 我们的 2^20（=2048×V）：两者都在「大于词表、远小于 distinct」
   的甜点区；S2 可加一臂 R=5V 的 BHE 式小表做交叉验证。
4. speedrun 的泄漏检查 + load-diff 评估规范纳入我们的验收清单（S0/S1 沿用）。
5. **协议基线警告（§79 副产物）**：本文 dense 权威值 1.867/0.245 出自 v10/2000 步/旧
   table-LR 协议；当前代码 1000 步协议的可学习表对照 = **2.702**（与 LR 扫描 2.73 @128×
   一致）。2×2 网格的 dense 两臂须在当前代码/协议上重跑（或先确认协议等价），不得与
   旧绝对值直比。
6. **论文新增可卖点（§80）**：「信号效用与 forking 解耦」——一个不可学习但信息丰富
   的 n-gram 表改善绝对损失而不引发 train/val 分化；这把 forking 定位为
   **训练动力学现象而非信息现象**，是 follow-up paper 的差异化论点之一。

## 3. 极简 MoE baseline 规格（小而 solid 的关键决策）

**原则**：P1 极简——在现有 nanoGPT 主线上只加一个变量（dense MLP → MoE MLP），n-gram 管线/口径/数据全部不动。不自建 router 学习花活（无 BPR、无 expert choice、无 soft routing），router = 线性 + top-k + 权威负载均衡损失（可关）。

### 3.0 权威锚点（2026-10-17 调研，替代自造配置）

| 来源 | 配置 | 我们借什么 |
|---|---|---|
| **OLMoE-1B-7B**（AllenAI，[arXiv:2409.02060](https://arxiv.org/abs/2409.02060)，全开放） | 64 experts / top-8，d=2048，expert dim 1024（= d/2），无 shared expert；balance loss **0.01** + router z-loss **0.001**；dropless per-token；norm_topk_prob=False | **K/E = 1/8**；expert dim = d/2；两个正则权重 |
| **Qwen3.8-Flash-Next**（arXiv:2608.30320） | 512 experts top-10+1shared，expert dim 640（= 0.25 d）；n-gram/表/router 留在 AdamW | 细粒度薄 expert 是旗舰方向；MoE+engram 共存 |
| **2609.11917**（MoE 退化论文） | 记号：granularity g = expert dim / (4d)；变 g、n、k；结论「退化随 **total params/稀疏度** 增长」 | **granularity 记法**；matched-active 设计的理论依据 |
| DeepSeek-MoE | 64 routed + 2 shared，top-8，balance 0.001 | 细粒度+shared 的起源（S3 可加 shared-expert 消融） |

### 3.1 结构（`code/train.py` 新增 `--moe_experts E --moe_topk K --moe_aux_weight w --moe_zloss_weight z`）

| 坐标 | 取值 | 权威依据 |
|---|---|---|
| expert 结构 | 标准 MLP（同 dense 块的实现，2 矩阵 GELU） | 与 dense 逐位可比；P1 优先于 SwiGLU |
| E / K | **E=32, top-8, d_e=384**（granularity g=1/2） | OLMoE 的 **K/E=1/8** 与 expert dim=d/2 原样缩放；active FFN = 8×768×384×2 = 4.7M ≈ dense FFN（**matched-active**）；total = 4× dense FFN → 直接检验 2609.11917「total params 主因」 |
| router | `nn.Linear(768, 32)`，fp32 softmax，token-level top-8，不归一化 topk 权重（norm_topk_prob=False） | OLMoE 同款 |
| aux loss | Switch 式负载均衡 **0.01** + router z-loss **0.001**（各可关） | OLMoE 权威值；关闭臂用于纯观察 |
| 诊断 | per-eval-block：expert load 分布、load 熵、dead-expert 数、aux/z-loss 值 | router ossification 直接测量（2609.11917 中间变量） |
| 其余 | backbone/n-gram/优化器/数据/口径 = §1 SSOT 原封不动 | P1 |

（弃选 E=8/top-2：组合数 C(8,2)=28 太少，router ossification 的动态范围不足；E=32/top-8 保留 matched-active 且对齐权威比例。）

### 3.2 run_id 与登记

- 新实验线 `moe_*`：`moe_moe2of8_ngon_s42`、`moe_moe2of8_ngoff_s42` 等（后缀沿用 §1.0 契约）。
- 开跑前在 `experiment-log.md` 注册：arm 定义、E/K/aux、与 2609.11917 的关系、验收标准。
- setting 表只允许加粗 MoE 三项（E/K/aux），其余与 §1.0 完全一致。

## 4. 实验阶梯（每级有明确验收，不跳级）

| 阶段 | 内容 | 验收 | 成本 |
|---|---|---|---|
| S0 sanity | moe2of8 + ng_on，1000 步 ×1 seed | router 稳定（step 200 起 dead experts = 0，load 熵 > 1.5 nat/8）；train loss 与 dense 同量级（matched-active 不应显著更差）；eval/diag 管线无 NaN | 1 run，约 1 天内 |
| S1 forking 复现 | moe2of8 × {ng_on, ng_off}，epoch-anchored E=3 协议，3 seeds | MoE 臂 3-epoch gap 曲线完整；与既有 dense 两臂（input 1.867 / nogram 0.245）组成 **2×2 网格** | 2 arms × 3 seeds = 6 runs |
| S2 干预交叉 | MoE + mask_low 剂量（le1/le8）；MoE + aux 变权重 | H4：表侧与 router 侧干预的效应量对比 | 4–6 runs |
| S3 增容 | E=32 top-2（total/active 比拉大） | H1 强度随稀疏度增长？ | 2 runs |

**S1 的 2×2 网格是核心交付**：dense+ng 与 dense+nogram 两臂已有权威数据，只需新增 moe 两臂——花 6 个 run 得到完整的「架构 × OE」forking 相互作用矩阵。

## 5. 已知风险与对策

1. **小规模 router 不稳定/塌缩** → aux loss 兜底 + dead-expert 诊断；若 E=8 仍塌缩，先升 aux_weight 单变量。
2. **matched-active 口径之争** → S0 报告 matched-active 与 matched-FLOPs 两种账（active params 相同 = matched-FLOPs 近似成立；total params 4× 是设计点不是混淆变量）。
3. **与 2609.11917 的协议差异**（他们固定 token 预算改 unique 数，我们固定 unique 改 epoch 数）→ 论文里明确写「epoch 维度复现」，不 claim 直接复刻。
4. **P5 红线**：MoE 属 backbone 大改，本文档即「先写明原因」的载体；开跑前还须用户点头。

## 6. 引用清单（正文可直接用）

- Qwen Team. On the Design of Qwen3.8-Next Architecture. arXiv:2608.30320（§2.3 placement/vocab 消融；Muon 分配：embedding/n-gram table/router/GR low-rank 留在 AdamW）
- Qwen3.8-Flash-Next HF README（20M rows bigram/trigram @ layer 2；512E top-10+1shared）
- MoE repetition：arXiv:2609.11917（agents.md §6 已有完整对照口径）
- Engram 形态溯源：Cheng et al. 2026（contextual gating）；NGM arXiv:2605.16893（相关工作综述：LongCat-Flash-Lite、MeKi）
- SCONE arXiv:2502.01637 / COUNT arXiv:2610.20456（plan-8 已收录，n-gram 注入方式对照）
