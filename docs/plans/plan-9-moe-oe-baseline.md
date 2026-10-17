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

## 3. 极简 MoE baseline 规格（小而 solid 的关键决策）

**原则**：P1 极简——在现有 nanoGPT 主线上只加一个变量（dense MLP → MoE MLP），n-gram 管线/口径/数据全部不动。不自建 router 学习花活（无 BPR、无 expert choice、无 soft routing），router = 线性 + top-k + Switch 式 aux load-balance loss（可关）。

### 3.1 结构（`code/train.py` 新增 `--moe_experts E --moe_topk K --moe_aux_weight w`）

| 坐标 | 取值 | 理由 |
|---|---|---|
| expert 结构 | 标准 MLP（768 → d_e → 768，ReLU/GELU 同 dense 块） | 与 dense 逐位可比 |
| E / K | **E=8, top-2, d_e=1536** | active FFN params = 2×768×1536×2 = 4.7M ≈ dense FFN（768×3072×2）→ **matched-active**；total = 4× dense FFN → 直接检验 2609.11917 的「total params 才是退化主因」 |
| router | `nn.Linear(768, 8)`，fp32 softmax，token-level top-2 | 极简；无噪声项（单变量） |
| aux loss | Switch 式负载均衡，`--moe_aux_weight 0.01`（0 = 关，单变量消融位） | 防 router 塌缩；关闭臂用于纯观察 |
| 诊断 | per-eval-block 记录：expert load 分布（均值路由份额）、load 熵、dead-expert 数、aux loss 值 | router ossification 的直接测量（对齐 2609.11917 的中间变量） |
| 其余 | backbone/n-gram/优化器/数据/口径 = §1 SSOT 原封不动 | P1 |

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
