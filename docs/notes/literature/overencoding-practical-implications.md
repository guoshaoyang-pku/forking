# Over-encoding 模块的实际效应：forking 机制 → scaling law → 预训练/后训练风险矩阵

> 调研日期：2026-09-06。续篇：`llm-overencoding-posttraining-survey.md`（2026-05，机制与文献面）、`posttraining_freeze_notes.md`（冻结证据面）。
> 本文回答一个问题：**我们发现的 forking 机制（表锚点 + backbone margin 锐化）对实际的 over-encoding 技术（Engram/SCONE/OverEncoding/X-gram/MoE）意味着什么？**
> 标注约定：**[实验证据]** = 我们 run 或公开论文的可复现数据；**[机制推断]** = 由我们的机制模型直接推出、可检验；**[推测]** = 类比外推。

---

## 0. TL;DR

1. **预训练侧**：Engram 论文的核心叙事「把 early-layer 静态回忆卸载进 memory、释放有效深度」与我们发现的「backbone 把低频记忆外包给表、转而做 margin 锐化」是同一件事的两面。卸载的收益是实的（iso-FLOPs 胜 MoE），代价藏在平均 val loss 里：**表把 train 分布的采样残差写成锚点，backbone 每 pass 把 margin 线性锐化 +1.08，经 1−p̂≈0.87 传递为 novel continuation 的 val 伤害 +0.94/pass**。9 个 over-encoding 方法（含 Engram）无一报告 frequency-conditioned gap，这个代价目前对文献不可见。
2. **scaling law 侧**：我们的 R 扫描给出 gap ∝ R^0.52（bigram）/ R^0.68（trigram）——表越大，forking 越重。这给 Engram 两条 scaling 曲线各加了一个修正项：① U-shape 最优 ρ≈75–80% 的右侧（表份额过大）应当伴随 novel-bucket val 伤害单调上升，可检验；②「infinite-memory regime 的 log-linear 改善」应附带「novel-mass val loss」辅助指标，否则 log-linear 里混入分布级记忆化。
3. **epoch/数据受限侧**：gap 按 step 线性（0.664/pass，R²=0.998），与「step 平权、epoch 不平权」一致。数据受限、被迫多 epoch 的预训练（Muennighoff 4-epoch regime）对带表模型更危险：**重复曝光对 dense backbone 是收益递减，对表是伤害递增**——最优 epoch 数比 dense 模型更小。
4. **后训练侧**：马嘉祺事件（MiniMax M2.5，官方已复盘并在 M2.7 修复）是**生成端**（lm_head）的稀疏 token 漂移：SFT 数据 <5 条样本的低频 token 被高频更新挤漂移，理解保留、生成丢失。显式表架构的对应失效模式是 **context 端**的三种失配（§4 矩阵）：frozen 表的 hash 碰撞被后训练兑现为新偏差、SFT 未覆盖键的「表还记得、backbone 不读了」、以及表若继续训练时的 value 漂移。MiniMax 的工程解法（全词表覆盖合成复读数据）可直接映射为**表键空间覆盖度监控 + 合成回放**。
5. **可操作清单**在 §5：评估纪律（val 按表命中/未命中分层）、预训练干预（每 pass reseed，实测 gap 清零至 0.069）、后训练策略（freeze 表 + 覆盖度下限）、scaling 报告规范（补 novel-mass val loss）。

---

## 1. 机制回顾（一页，全部 [实验证据]，128× 标准、2000 步、seed 42）

- **三条件**：①有限样本采样残差（Good–Turing 缺失质量 M_f=N₁/f，1x shard 整体 0.278）；②私有 key 稀疏记忆（hash n-gram 表）；③归一化优化器 + 重复曝光。三者齐 → forking；去任一（nogram / mask_low / reseed）→ gap 塌缩。
- **表是静态锚点**：pass 1 即写好（freeze_table@e2 仍保留 94% gap）；内容必要（hash_reseed → gap 清零）；之后持续重写（cos 0.5–0.76）但与 gap 无关。
- **backbone margin 锐化**：f=1 context 上 margin 线性 +1.08/pass；经 softmax 零和传递（系数 1−p̂≈0.87）压 novel token → novel val 伤害 +0.94/pass，pass 6 时占 gap 的 61%。
- **step 平权**：20 epoch 内 gap 对 step 线性 0.664/pass（R²=0.998），72% 增长发生在 pass 内部的普通 step 上，边界跳变只占 28% 且衰减。
- **频率形状**：d_e(f)/M_f 在 f∈[2,320) 平坦（CV 4–6%）——低频每个 context 等份受害，不是越稀有越糟。
- **R 依赖（§52）**：gap ∝ R^0.517（bigram）/ R^0.680（trigram）；epoch 长度 U 形；multi-layer 注入 gap 持平或更低。
- **干预账本**：reseed 每 epoch → gap 0.069（≈清零）；mask_low f≤8 → −72%；mask_high 阈值扫描 → gap 随保留的低频质量线性。

## 2. 预训练 regime：over-encoding 技术的收益叙事与隐藏代价

### 2.1 技术谱系与我们的审计位置

| 方法 | 记忆形式 | key | 训练 | 评估缺口（我们审计 [实验证据]） |
|---|---|---|---|---|
| Over-Encoding / Over-Tokenized (Huang, ICML 2025) | modulo hash 表，input 求和 | 1–N-gram | 与 backbone 联合 | 无频率分桶 gap |
| **Engram/DSE (DeepSeek, ACL 2026, arXiv:2601.07372)** | 多层 multi-head hash 表 + hidden gate；多级缓存 offload | n-gram | 联合，27B | 只有平均 val loss 与下游 benchmark |
| LongCat (Liu, 2026) | 多项式滚动 hash 多子表 | n-gram | 联合 | 只报 collision rate |
| X-gram (Chen, 2026) | 频率感知混合 hash（VIP 私行 + 共享尾桶） | n-gram | 联合 | 频率分析只在 train 侧（row 更新计数），无 val 分桶 |
| SCONE (Google, NeurIPS 2025) | 频率筛选 f-gram + 共享 f-gram Transformer 生成向量 | exact key | 预训练后 frozen，offload KV | longest-match fallback 的分布偏移未研究 |
| TN-gram (2026) | CP 低秩共享，无显式碰撞 | n-gram | 联合 | 同上 |
| kNN-LM / Infini-gram | 非参数检索 | exact | 不训练 | 增益集中于高频、长尾无改善（Nishida 2025） |

**审计结论**：9 个方法全部以「平均 held-out loss / 下游 accuracy / collision rate」为容量指标，无一报告 frequency-conditioned train/val gap。forking 这类代价在当前协议下**结构性不可见**。

### 2.2 Engram 的机制叙事 = 我们发现的两面硬币

Engram 论文的机制分析（arXiv:2601.07372 §5）：memory 模块「relieves the backbone's early layers from static reconstruction, effectively deepening the network」，收益最大的反而是推理（BBH +5.0）而非知识任务。 **[实验证据]**

我们的发现给出同一现象的完整账目 **[实验证据]**：
- 正面：表在 pass 1 就把 train 分布的 continuation 统计写成静态锚点，backbone 不再消耗容量重建它 → 「释放有效深度」。
- 背面：backbone 在可分环境（每个 train context 有私有 key）上把 margin 当信号持续锐化；锐化的能量来自采样残差 δ_c = p̂_c − p_c；1−p̂ 传递把 seen 侧的自信转化为 novel 侧的系统性压低。
- 净效应：在我们 2000 步、3 epoch、128× 的尺度上，input 注入臂 val 绝对值 6.95 vs nogram 3.34——**表净伤害 +3.6 nats**。Engram 的 27B、近 1-epoch、工业数据混合 regime 在 U-shape 最优点附近净收益为正，两者不矛盾：**净符号取决于「记忆卸载收益 − margin 锐化伤害」的差，而这个差随 epoch 数、表容量 R、数据量 N 系统性移动**（§3）。

### 2.3 RETRO 同构警告

RETRO/kNN-LM 的收益大部分来自数据库与测试集的 lexical overlap（arXiv:2302.12128；NAACL 2025） **[实验证据]**。**[机制推断]** Engram 类可训练表同样存在「评估泄露」的温和版本：held-out val 中与 train 共享的 context（我们实测 shared-context 占 val token 的大头）享受表的记忆收益，novel context 承担锐化伤害；平均 val loss 是两者的加权和，权重 = 共享质量。数据混合越接近重复、epoch 越多，平均数越好看、novel 侧越糟。**平均 val loss 在带表模型上是有偏估计，偏的方向恰好有利于表。**

## 3. 与 scaling law 的关系

### 3.1 给 Engram U-shape 一个微观候选解释

Engram 的 sparsity allocation：固定稀疏参数预算，MoE 份额 ρ 与 val loss 呈 U 形，最优 ρ≈75–80%（两个算力档位一致） **[实验证据]**。

我们的机制给出 U 形两侧的微观力 **[机制推断，可检验]**：
- 左侧（表份额过小）：key 空间不足 → collision 多重性上升 → 行写入互相干扰（我们 R 扫描的低端：R≤1e4 时 collision 主导，gap-R 幂律断裂、噪声剧增）+ 记忆卸载不足。
- 右侧（表份额过大）：① MoE 计算容量被挤压（Engram 自己的解释）；② **表容量越大，低频 context 拿到的私有行越多、训练越少欠拟合，backbone 锐化的燃料越多**——我们实测 gap ∝ R^0.52/0.68，在整个中段是单调幂律。
- 检验方案：在 Engram 的 ρ 扫描上补测 frequency-conditioned val loss，预测 U 右侧 novel-bucket 伤害单调上升、shared-bucket 收益饱和——两个分量的交点应比平均-loss 的 U 更靠左。

### 3.2 「infinite-memory log-linear」需要第二个指标

Engram 报告 embedding slots 数与 val loss log-linear（infinite-memory regime） **[实验证据]**。我们的 R^0.52/0.68 gap 幂律意味着：slots 每翻倍，forking 伤害按固定指数增长。 **[实验证据（我们的尺度）+ 机制推断（外推）]**
- 若 log-linear 的斜率里有恒定一块来自 shared-context 记忆收益，则「无限加表」的实际泛化收益会低于 log-linear 外推。
- 建议的配套指标：**novel-mass val loss**（按 train hit-count=0 的 context 聚合的 val 损失）与 shared-mass val loss 分开报告。两者对 log slots 的斜率差，就是表的真实边际泛化收益。

### 3.3 数据受限 / 多 epoch regime：带表模型的最优 epoch 数更小

- Muennighoff et al. 2023（data-constrained scaling）：重复数据在 ~4 epoch 内接近唯一数据价值 **[实验证据]**；Xue et al. 2023：token 受限下多 epoch 过拟合、模型越大越敏感 **[实验证据]**。
- 我们：gap 对 step 线性、与 pass 数弱相关（step 平权） **[实验证据]**。含义：固定 token 预算，第 k 次重复对 dense backbone 是「收益递减」（Muennighoff），对表是「伤害线性递增」（我们的 0.664/pass）。
- **[机制推断]** 带表模型的 data-constrained scaling 曲线应在 epoch 轴上更早拐点；最优 epoch 数随表容量 R 增大而减小。这给出一条可执行建议：**数据受限时，先缩表或 reseed，再考虑加 epoch。**

### 3.4 容量分配次序的显式化

Morris et al. 2025：模型容量 ~3.6 bit/param，且「记忆优先于泛化」 **[实验证据]**。表把这条隐式次序变成显式：表的 bit 几乎必然先被 train 特有残差填满（它是按 key 寻址的、归一化优化器驱动的、pass-1 饱和的快分量）。**[机制推断]** 表的加入不改变「记忆优先」的次序，只改变它的可观测性与可干预性——这是 forking 研究对 scaling-law 实践的主要价值：记忆化从「事后审计」变成「训练时可定位、可切除的组件」。

## 4. 后训练 regime：frozen table 的风险矩阵

### 4.1 马嘉祺案例精确复盘（全部 [实验证据]，MiniMax 官方复盘 2026-05-09）

- 事件主体是 **MiniMax-M2.5**（非 M3；M3 为 2026-06 起发布的 MSA 架构新模型）。M2.5 无法输出「马嘉祺」：知识在（能答履历、团体、出道时间），生成不出（输出「马嘉轩」「马丝祺」）。**M2.7 已修复。**
- 根因链：「嘉祺」是独立 token（id=190467），预训练充分（embedding 近邻全是明星人名）→ **SFT 数据含该 token 的样本不足 5 条** → lm_head 中该 token 向量方向漂移（低频无有效梯度，被高频 token 更新与衰减项持续挤压）→ 生成概率跌出 top-p。**input embedding 几乎不动 → 理解保留、生成丢失。**
- 系统性：全词表 4.9% token cos_sim<0.95（日语 29.7%，解释日俄混淆）；修复 = 全词表覆盖合成复读数据（每个 token 建立生成频率下限）→ cos_sim<0.95 归零，日→俄混淆 47%→1%。

### 4.2 映射到显式表架构：三种策略 × 三类失效模式

马嘉祺机制是**生成端**（lm_head 行）的稀疏漂移。显式表架构多了 **context 端**（表 key→row→value→backbone 读出）一整条链。设 SFT 数据对预训练 n-gram 键空间的覆盖不全（必然如此——SFT 语料远小于预训练）：

| 策略 | 失效模式 | 机制 | 严重度评估 |
|---|---|---|---|
| **A. 表 frozen**（常见默认） | A1. **碰撞兑现**：SFT 新 context（预训练未见的 n-gram）在 frozen 表上读到的是 hash 碰撞行的旧 value；backbone 在 SFT 中把这个伪先验当锚点锐化 → 预训练的碰撞在后训练被「兑现」为系统性偏差 | 我们的三条件在 SFT 分布上重新成立（新残差 + 私有（但错误的）key + 归一化优化器） | **[机制推断]** 随 SFT 新 context 比例与碰撞率上升；X-gram 的 VIP 私行设计可缓解 |
| | A2. **「表还记得，backbone 不读了」**：SFT 未覆盖的预训练高频键，表行 frozen 不变，但 backbone 的读出通路（gate、下游层、lm_head）在 SFT 分布上漂移 → 共适应失配 | 马嘉祺机制的 context 端镜像：frozen 行 ≈ 保留的 embedding，漂移的 backbone ≈ 漂移的 lm_head | **[机制推断]** 与 MiniMax 4.9% 漂移同源；对「知识在表里」的架构（Engram 叙事）尤其危险 |
| | A3. 僵尸锚点：表保留预训练分布先验，SFT 锐化叠加在旧锚点上 → 分布收窄时旧记忆干扰新任务 | step 平权：SFT 步数继续线性累积 margin | 中；可视为轻度的继续预训练 forking |
| **B. 表继续训练** | B1. value 向 SFT 分布漂移 → 预训练共适应关系被打破；预训练记忆退火（天然正则）或冲掉（灾难） | 我们的 reseed/reset 干预的慢性、非受控版 | **[推测]** 方向取决于 SFT 与预训练分布距离；retro-li 复现（arXiv:2410.00004）观察到检索模块与 backbone 过拟合耦合 **[实验证据]** |
| | B2. SFT 多 epoch（常见 2–3 epoch）+ 小数据 → 表的 forking 在 SFT 分布上重现，且 SFT 数据更小、M_f 更大 | 三条件在更小数据上更强 | **[机制推断]** 高；posttraining_freeze_notes.md 已据此建议 SFT 阶段表只读 |
| **C. 表转只读检索**（kNN-LM/SCONE 式） | C1. 检索库与 SFT 分布的 support mismatch；低频键依然无保护 | Nishida 2025：kNN 增益集中高频、长尾更差 **[实验证据]** | 低-中；但失去了可训练表的主收益 |

### 4.3 工程解法映射

MiniMax 的修复 = **为每个 token 建立生成频率下限**（全词表合成复读）。映射到表架构 **[机制推断]**：
1. **键空间覆盖度监控**：SFT 数据统计 n-gram 键命中分布，标记「预训练高频 × SFT 零出现」键（A2 风险）与「SFT 新键 × 高碰撞」（A1 风险）。
2. **合成回放**：对高风险键构造最低频率的复读样本（成本远低于全词表——只需覆盖键空间的长尾交集）。
3. **冻结 + 低 lr backbone**：冻结表消除 B 类风险；backbone 低 lr / LoRA 减缓 A2 漂移（LoRA Learns Less and Forgets Less, TMLR 2024 [实验证据]）。
4. **定期 reseed（预训练内）/ 每 pass 重置（多 epoch SFT 内）**：我们实测 reseed 每 epoch gap→0.069，是消除 A3/B2 的受控手段。

## 5. 可操作清单

**评估纪律（立即可做，零训练成本）**
- E1. val loss 一律按「表命中 / 未命中（train hit-count=0）」分层报告；scaling 曲线附 novel-mass val loss。
- E2. 带表模型的 benchmark 增加「重复采样多样性」与「长尾事实置信度校准」两项——margin 锐化压 novel continuation，预测直接表现为多样性下降与长尾过度自信 **[机制推断]**。

**预训练**
- P1. 多 epoch 必须配干预预算：每 pass reseed（gap→0.069）或 epoch 边界表重置；不允许「裸表」跑 >1 epoch。
- P2. 数据受限时先缩表/加 reseed 再加 epoch（§3.3）。
- P3. 表容量按数据量配比：M_f 随 N 缩减，R 的边际泛化收益递减快于 log-linear（§3.2）——R* 应随 N 亚线性增长 **[机制推断]**。

**后训练**
- S1. 默认 freeze 表（消除 B 类），配键覆盖度监控（§4.3-1/2）。
- S2. backbone 用低 lr 或 LoRA，保护「表-backbone 共适应通路」（防 A2）。
- S3. SFT 数据审计报告应包含 token 覆盖度（MiniMax 标准：全词表 cos_sim）**和** n-gram 键覆盖度两张表。

**对 scaling law 文献**
- L1. Engram 类 U-shape 扫描补 frequency-conditioned val loss，检验最优点是否左移（§3.1）。
- L2. infinite-memory regime 报告 novel/shared 双斜率（§3.2）。
- L3. data-constrained scaling 实验把「表容量 × epoch 数」作为正交轴（§3.3）。

## 6. 未解问题（按价值排序）

1. **Engram 27B 的 frequency-conditioned 审计**：我们的机制在工业尺度的直接验证。需要 DeepSeek 或第三方在开源 Engram 模型上按 train hit-count 分桶测 val。这是全文最重要的一条外部检验。
2. **A1「碰撞兑现」的受控复现**：我们的架构上做「预训练 → frozen 表 → 分布偏移 SFT」三阶段实验，测新 context 的 val 退化是否定位在碰撞行。仓库内可完成，成本 ~1 天。
3. **U-shape 右移/左移检验**：在 ρ 扫描上补分桶指标（§3.1），需要 MoE+表混合架构，成本较高。
4. **margin 锐化与 3.6 bit/param 容量次序的定量接口**：表的 bit 占用是否严格「优先」于 backbone 的泛化容量？可用 Morris 的容量测量协议在我们的表上重跑。
5. **生成端（lm_head）与 context 端（表）漂移的相对速度**：MiniMax 只测了 lm_head；我们架构可同时测两侧，预期表侧更稳（frozen）但读出通路先漂——指导 S2 的 lr 选择。

---

## 附：新增来源（本文首次引用）

| 主题 | 来源 |
|---|---|
| Engram / DSE（U-shape、27B、offload 叙事） | arXiv:2601.07372（ACL 2026）；github.com/deepseek-ai/Engram |
| TN-gram（CP 共享、Parameter-Golf 复现） | arXiv:2606.08347 |
| MiniMax 官方复盘（马嘉祺、lm_head 漂移、修复） | minimax.io/blog/sparse-token-forgetting（2026-05-09，M2.5 事件、M2.7 修复） |
| MiniMax-M3 / MSA（型号澄清） | arXiv:2606.13392 |
| data-constrained scaling | Muennighoff et al. 2023（NeurIPS）；Xue et al. 2023（arXiv:2305.13230） |
| 容量与记忆次序 | arXiv:2504.12527 |
| 其余（记忆化综述、RETRO/kNN 局限、MoE 偏记忆、冻结证据） | 见 `llm-overencoding-posttraining-survey.md` 与 `posttraining_freeze_notes.md` 来源清单 |

**机制数字出处**：`docs/experiment-log.md` §43–§52（净收益批、margin/传递系数、step 平权、R/epoch/multi-layer scaling）；`docs/report/index.html` 图 1–17。
