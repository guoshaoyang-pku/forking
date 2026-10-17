# Plan 9 · OE × MoE：matched-budget MoE × OE 研究家族（v2，按 GPT-pro 交接修订）

> 状态：实现中（2026-10-17 v2 重写）。设计单一事实来源 =
> `docs/plans/plan-9-handoff-gptpro-20261017.txt`（GPT pro 指正全文，已归档）；
> 本文是其在仓库内的落地版。v1 的 E32/K8/d_e384「OLMoE 原样缩放主基线」说法作废。
> 实现与开跑前须按 P3 注册到 `experiment-log.md`；主 6 run 开跑须用户确认（P5）。

---

## 0. 最终决策摘要（与 handoff §0 一致，含两处仓库现实修正）

1. **数据量**：U = 1.0B unique tokens × R = 3 完整顺序重放（T = 3.0B）。
   保存 epoch 1/2/3 精确边界 checkpoint，保留统一延长到 epoch 4 的能力（要延全臂一起延，
   禁止选择性延长）。S0 工程 sanity 只用 100M × 1 pass × 1 seed。
   **仓库现实修正 A（语料）**：handoff 假设 nanoGPT-OWT/GPT-2 BPE/12L-124M；本仓库实际
   为 **full-163 语料**（163 parquet 分片 ≈ 8B tokens、8192-vocab 固定 tokenizer、
   8L/768D backbone，N_active ≈ 63M）。保留本语料的理由：OE 全套基础设施
   （hash 素数/freq index/row precompute）与 dense 可比性绑定在此 tokenizer 上；
   handoff 自身规则「主模型仍优先保持 1B 以复用既有 dense 结果」优先。
   T/N_active = 3B/63M ≈ 48（20× 规则会给 U≈0.42B；本协议取 1B 以兼容 stage-8 的
   2B×3 复核梯度，重复压力弱于论文最优检测区间——若 3 遍效应太弱，全臂统一延第 4 遍）。
2. **主 MoE baseline**：d=768，dense_ffn=3072，**32 routed / top-4 / d_e=768**，
   S=0（无 shared），dropless token-choice；router logits/softmax fp32；
   Switch 式 aux **0.01** + router z-loss **0.001**；expert MLP 与 dense MLP 完全同构
   （2 矩阵 GELU、同 bias、同初始化）；OLMoE-faithful 默认 `norm_topk_prob=false`
   但**必须记录 selected_topk_probability_mass**。
   统一坐标：**G=4（granularity），X=8（FFN total expansion）**。
3. **shared 对照**：核心 2×2 不用 shared（最干净 learned-routing baseline）。
   严格 matched 的 shared 对照 = **31R/top-3 + 1 shared，d_e=768**（expert unit 总数、
   每 token active units、FFN 主体 FLOPs 三者都对齐）；禁止 32R/top-4+1S（active +25%）。
4. **旧配置去向**：E32/K8/d_e384 = **G8/X4 fine-granularity arm**（granularity 轴上的
   一点，非主基线；它也不继承 OLMoE 的 K/E=1/4≠1/8）。若要 G8 下同时保 X=8 与
   K/E=1/8，用 E64/K8/d_e384。
5. **开源定位**：不做 size-only family（FLAME-MoE 已发 7 个 active size）；核心资产 =
   `moeify()`/`oeify()`、正交结构轴（X/G/S）× OE 轴（off/trainable/frozen）、
   重复数据协议、router+OE 联合诊断、全 checkpoints。

## 1. Qwen3.8 Flash 的记忆 setting 查证（v1 保留，措辞按 handoff §2.5 修正）

**结论（公开证据，非内部情报）**：Qwen3.8-Flash-Next（2026-08 发布）明确使用 engram
形态的 n-gram conditional memory：

| 来源 | 关键事实 |
|---|---|
| 官方设计论文 [arXiv:2608.30320](https://arxiv.org/abs/2608.30320) | 125B backbone / 6B activated / **+51B n-gram embedding tables**（host-memory prefetch）；§2.3 专章消融 |
| HF README | **N-gram Embedding: 20,000,000 rows（bigrams/trigrams），Layer 2 单层注入**；512 experts top-10 + 1 shared；expert dim 640 |
| vLLM recipes | 配置旗标 **`--engram-config`**；table 可 CPU offload |
| NGM 论文 [arXiv:2605.16893](https://arxiv.org/abs/2605.16893) | 「Engram = hashed N-gram lookup + context-aware gating + sparsity allocation，27B，deep-layer injection」；LongCat-Flash-Lite 同路线 |

同构性：bigram+trigram 双支 / 确定性 hash / 单层注入（他们 L2，我们 wte）/ 规模
20M rows vs 我们 2×2^20 rows。论文 §2.3 消融：placement Tab 7（L2 最佳）、vocab
扫描 Tab 9（**loss 单调降、下游饱和**的不一致——与 forking 叙事直接对接）。
**正确表述（handoff §2.5 修正）**：厂商*做过* vocab 容量/placement/频率分区等尝试，
但**没有从 repetition-induced forking 的角度系统研究**——这是我们 follow-up 的空位；
不再写「厂商没有做频率治理」。

## 2. OE × MoE 研究空间与社区调研（v1 保留 + 算术修正）

核心四问（不变）：H1 MoE 是否更早 fork；H2 加 OE 是否超线性；H3 router 负载 × 表命中
联合分布；H4 表侧（mask_low）与 router 侧（aux）干预是否互补。

### 2.5 社区 OE 调研（parameter-golf / modded-nanogpt）

**A. modded-nanogpt PR #201「Bigram Hash Embedding」**（record 62，val 3.2776）：
5×V 表（≈251K rows）、hash `(p1·prev) XOR (p2·cur) mod R`（与我们 clean 表第一族
同构，独立收敛）、zero-init、每层残差 + 可学 λ、CPU 端 hash 预计算（= plan-8
index-first 同思想）、跨文档边界未 mask（试过，无收益）。**实现内负结果**：gating /
trigram / multi-hash 未跑赢直接相加（单实现 caveat，引用须 pin PR/commit，不引
leaderboard 主分支配方）。BHE 与 Qwen engram 的 gate 分歧不能跨规模互推。

**B. parameter-golf**（OpenAI Model Craft，[arXiv:2607.01517](https://arxiv.org/abs/2607.01517)）：
冠军方案 = **test-time n-gram blending**（eval 现场建 cache、Katz backoff 2–7、
entropy-adaptive λ；neural-only 1.1271→0.9674 BPB）。零工件字节。观察性证据
（排行榜分析），不是 trainable OE 的机制证据；BPB 数字必须绑定具体 submission。

**B'. 两轮冻结实测（§79/§80，2026-10-17）**：step-0 冻结随机表 gap **0.064**、
step-0 冻结 count-init 表 gap **0.022**，均 vs 可学习对照 **2.702**（49.7M×3 协议）；
count-init 冻结臂 train/val 绝对值还更好（信号效用与 forking 解耦）。学满后冻结
（e1/e2）照样 fork 且可恶化（3.452>2.81 / 5.386=94%）。判词：**点火 = 表侧 gradient
sharpening（需表可学习），燃烧 = backbone 共适应（不需要）**；表内容质量（随机/统计）
与冻结时机（step-0）都不是充分条件。注意 handoff §5.4 口径：frozen 行写「本协议内
实测不 fork」，不写「绝对不 fork」——更强重复压力/更长训练未测。

**C. OE 方法三分类**（related-work 骨架；frozen 两种内容已实测）：

| 类 | 代表 | 表参数 | 预测 | 实测 |
|---|---|---|---|---|
| trained-in hash embedding（=engram） | 我们 / DeepSeek Engram / BHE / NGM | 学习 | table+backbone 双侧 sharpening → **forking** | 2.702（§79 ctrl） |
| test-time blending | golf / infini-gram | 无（eval 局部） | 无训练期 table sharpening；但有 eval-local adaptation | —（不同方法类） |
| frozen / reparam 表 | COUNT / MeKi / freeze 线 | 固定 | 无 table sharpening；backbone 可利用固定信号过拟合（不能写绝不 fork） | 0.064 / 0.022；学满后冻 3.452 |

**D. 对本 plan 的直接影响**：① hash 公式与注入哲学有外部先例；② 廉价对照臂已落地
2/2（§79/§80）；③ ~~BHE 5×V vs 我们 2^20（=2048×V）~~ **算术修正**：2^20/8192
≈ **128×V**（v1 误用 50304 词表算出 20.8V/2048V 两处都错；本仓库 V=8192）。
外部有效性阶段可加 R=5V BHE 式小表臂；④ speedrun 泄漏检查规范纳入验收；
⑤ ~~dense 1.867/0.245 权威数据~~ **表述修正**：是本项目内部基线，非外部权威；
且出自 v10/2000 步/旧 table-LR 协议，与 p9 新数据协议不可比——p9 的 dense 两臂
**全部重跑**（见 §6 复用判定）。

## 3. MoE family 统一生成算法（moeify）

坐标（Fine-Grained MoE Scaling 记法，[arXiv:2402.07871](https://arxiv.org/abs/2402.07871)）：
**G = d_ff/d_e**（越大越细）、**X = total MoE FFN / dense FFN**、S = shared units。

```
d_e = d_ff / G;  K_routed = G;  E_routed = G·X − S;  E_shared = S
每 token active units = K_routed + S = G（严格 matched）
```

校验：G/X/S 合法整数；`d_ff % G == 0`；`0 ≤ S < G`；`E_routed ≥ K_routed`；
shared 与 routed 同 d_e、同 MLP 结构。`code/moe.py: moeify(d, d_ff, G, X, S, ...)`
生成后自动打印并写入 run metadata：d_e、E_routed、K_routed、E_shared、G、X、K/E、
dense/active/total FFN params、router params、OE params、whole-model active/total
params、理论 active matmul FLOPs/token、（运行期再记 measured tokens/s、step time、
peak memory）。

d=768、d_ff=3072 标准点（handoff §3）：

| 点 | G | X | S | 结构 | 用途 |
|---|---|---|---|---|---|
| A 主 baseline | 4 | 8 | 0 | **32R/top-4/d_e768** | 主 2×2 MoE 臂 |
| B shared matched | 4 | 8 | 1 | 31R/top-3 + 1S/d_e768 | shared 轴（阶段 5） |
| C 工程 pilot | 4 | 4 | 0 | 16R/top-4/d_e768 | **S0**（便宜一倍） |
| D capacity 轴（G4 固定） | 4 | 2/4/8 | 0 | 8R/16R/32R top-4 | 阶段 3：X→forking |
| E granularity 轴（X4 固定） | 2/4/8 | 4 | 0 | 8R/K2/1536 · 16R/K4/768 · **32R/K8/384** | 阶段 4（含 v1 旧配置） |
| F OLMoE-ratio 高粒点 | 8 | 8 | 0 | 64R/top-8/d_e384 | 可选 |

（v1 的「C(8,2)=28 太少淘汰 8选2」论证作废——E8/K2/d_e1536 是 G2/X4 有效对照点，
组合数不能直接推出 ossification 动态范围不足。）

**结构规则**：expert MLP = 与 dense MLP 完全同构（`c_fc` GELU `c_proj`，同 bias 同
init）；router = `nn.Linear(d, E_routed)` fp32；softmax 全 expert 后 top-k；
默认不 renormalize top-k 权重（OLMoE-faithful），`--moe_norm_topk_prob` 提供鲁棒性
对照；shared expert always-on、输出以权重 1.0 直加（DeepSeekMoE 式，定义固定进
单元测试）；aux = Switch 式 `E·Σ f_i·p_i`（f_i = top-k assignment fraction，Σf=1；
p_i = token 平均 router prob）；z-loss = `mean(logsumexp(logits)²)`（fp32）。

**参数量记法修正**：`total = X× dense` 只指 **FFN 参数**；whole-model 比例
≈ (4+8X)/12（每层 attn≈4d²、FFN≈8Xd²，忽略 embedding；X=8 → ~5.67×）。
所有实验启动时自动打印 7 类参数分解（§7 账本），禁止只报 X。

**matched-active ≠ matched wall-clock**：MoE 额外开销（router proj/softmax/top-k/
dispatch/imbalance/小 GEMM 利用率）。所有 run 同时记录理论 active matmul FLOPs/token、
measured tokens/s、step time、peak memory；论文不把 matched-active 写成
matched-compute。

## 4. 实现与验收（handoff §7–§8 全文有效，摘要）

**单元测试（`code/tools/test_moe_cpu.py`，正式训练前必须全过）**：
1. dense 等价：moeify(G=1,X=1,S=0) 复制 dense 权重、router 恒等 → forward/输入梯度/
   expert 权重梯度三方一致（fp32 atol 1e-6；bf16 路径容差单独注册）；
2. expert 置换等变：expert 权重 + router 列同置换 → 输出不变；
3. assignment 守恒：每 batch routed assignment 数 = N_tokens×K_routed，shared 调用数
   = N_tokens×S，dropless 不丢 token；
4. routing-weight contract：softmax 全 expert→top-k；默认不 renorm / norm 臂权重和=1；
   selected_topk_probability_mass 定义；padding/边界行为；aux/z-loss 公式逐项对拍。

**诊断（每 eval block，`moe_diag.jsonl` + panel npz）**：
- 负载侧（每层）：expert token fraction f_i、归一化 load 熵 H/log(E)、有效 expert 数
  exp(H)、load CV、max/mean、rolling dead-expert 数/率、aux、z-loss、topk mass。
  禁止跨 E 用未归一化熵阈值。
- ossification 侧（2609.11917 方法）：固定 **16,384-token held-out panel**（val 6542
  首 8 行，永不进训练）每个 checkpoint 记录 top-1/top-k expert IDs 与 logits →
  consecutive-checkpoint top-1 agreement、对 epoch-1 的 agreement、top-k Jaccard、
  rank-weighted overlap、K/K+1 margin、token-conditioned routing entropy。跨 E 比较
  报 chance baseline 1/E。**load entropy 不能替代 route stability**。
- loss contract：lm_loss 与 aux_raw/aux_weighted/zloss_raw/zloss_weighted/total 分开
  记录；报告 train CE 用 lm_loss。
- MoE×OE 联合（阶段 2 起）：I(expert; n-gram row)、I(expert; freq bin)、高频 row 的
  route stability、row update concentration × router stability 时间相关。

## 5. 数据合同（full-163 版；handoff §6 的结构全保留，粒度有仓库现实修正）

- **语料**：full-163 parquet 池（162 train 分片 + 6542 val；≈8B tokens、8192-vocab
  固定 tokenizer，`prepare_data.py` 管线已 byte-identical 验证）。tokenizer 以
  `/data2/.../cache/tokenizer` sha256 固定。
- **val（primary CE）**：分片 **6542**（group-canonical 未见过分片），fixed batches；
  16,384-token diagnostic panel = 6542 首 8 行；panel 永不训练；primary metric 不得
  中途替换。
- **抽样**：固定 random permutation（data_seed 独立固定）打乱**池内全部 packed rows**
  （2049-token、BOS 对齐），按该顺序拼接至 token budget；**仓库现实修正 B（粒度）**：
  .bin 不携带 document ID（上游 best-fit packing 已混合），doc-level permutation 需要
  重做整条 tokenize 管线并产生新口径——故以 **row-level permutation** 落地，同样满足
  handoff 的去局部性动机（跨 162 分片打散文档顺序/来源局部性），nested 子集性质
  保留（同一 permutation 的前缀）。manifest 记录该偏差。
- **manifest（immutable）**：permutation seed、全量 row 顺序（npz）、source shard 列表
  + 每 bin sha256、tokenizer hash、exact token count、materialized bins 的 hash。
  所有架构/OE/seed 共享同一 manifest；0.1B/1B/2B 梯度用同 permutation 前缀
  （`--train_max_tokens` token 精确截断）。
- **replay**：epoch 2/3 严格重放同一 row 顺序（packed order 不变，机制可解释性
  优先，不用 reshuffle）；data_seed 与 model_seed 分开；3 个 model seeds 用相同数据
  顺序（paired comparison）；LR schedule 用 `warmup_constant(100)`（SSOT 锚定，
  与全部既有 run 一致——constant 段与 anchor 无关，禁 warmdown）。

## 6. dense 两臂复用判定（handoff §1.9 九项比对；结论：全部重跑）

九项条件（commit / data manifest / tokenizer / token order / batch+seq / optimizer+
LR+wd / epoch 定义 / eval split+gap 定义 / OE 配置）：历史 dense 基线
（v10 两臂 1.867/0.245、frz0 ctrl 2.702 等）的 data manifest、token order、epoch 定义
三项均不满足 p9 合同（49.7M×3 ≠ 1B×3，无 manifest hash）→ **不构成同一 2×2 单元**，
标历史参考。主 2×2 需要 dense on/off × 3 seeds = 6 个 run 在 p9 manifest 上重跑
（成本与 MoE 臂同级，总 12 run，均在用户确认后启动）。

## 7. 统计与执行顺序（handoff §9–§10 全文有效，摘要）

- 主交互量：**I(e) = [Gap_MoE+OE − Gap_MoE] − [Gap_Dense+OE − Gap_Dense]**（每个
  epoch 的 I(e)、epoch 1→3 增量、interaction AUC、paired-seed mean±CI；必须同时报
  原始 train/val CE）。I>0 才支持「MoE 放大 OE forking」；单臂 gap 大不是证据。
- 阶段：**S0**（G4X4S0=16R/top-4，100M×1pass×1seed，OE on/off 两臂，验收见 §8）→
  **阶段 2 主 2×2**（G4X8S0=32R，1B×3，OE on/off 各 3 seeds + dense 重跑 6 run；
  第一核心交付）→ 3 capacity（G4 固定 X∈{2,4,8}）→ 4 granularity（X4 固定
  G∈{2,4,8}）→ 5 shared（31R/3+1S）→ 6 OE 外部有效性（trainable/frozen/BHE-5V/
  可选 test-time blend）→ 7 机制干预（mask_low → expert output masking → expert
  dropout → residual dropout → aux 0.003/0.01/0.03 → z-loss on/off；单变量 1 seed
  起步，禁止全笛卡尔积）→ 8 规模复核（headline 四臂 2B×3 或另一 backbone size）。
  每阶段预注册验收不过不进下一阶段。
- **run_id**（handoff §12）：`p9_{arch}_g{G}_x{X}_s{S}_e{Er}_k{Kr}_de{de}_{oe}_u{U}_r{R}_seed{seed}`，
  例 `p9_moe_g4_x8_s0_e32_k4_de768_oeon_u1b_r3_seed42`。v1 的 `moe_moe2of8_*` 作废。
  run metadata 至少含：git commit、dirty 状态、manifest hash、tokenizer hash、
  config JSON、optimizer/scheduler、全部 seeds、参数/FLOPs 报告、硬件、软件版本、
  起止时间、checkpoint 清单、eval 协议版本。
- 禁止事项清单以 handoff §16 为准（16 条，含：禁止用 load entropy 代替 route
  stability；禁止跨 E 未归一化熵阈值；禁止 matched-active 写成 matched wall-clock；
  禁止选择性延长 epoch；单 seed 趋势不得写稳定结论；golf test-time cache 效果不得
  解释成 trainable OE sharpening；Qwen recipe 不得称小模型 scaling law；speedrun
  数字必须 pin commit）。

## 8. S0 验收清单（当前阶段）

模型 G4/X4/S0 = 16R/top-4/d_e768；两臂 OE on/off；100M tokens × 1 pass × 1 seed
（≈ 679 步）。验收（数值报告，不用自然语言猜测）：
① 无 NaN/Inf；② assignment conservation（每 step 校验和）；③ 无永久 dead experts
（短期不均衡允许）；④ lm loss 与 dense 同量级；⑤ moe_diag.jsonl + panel npz 可复现
且字段齐全；⑥ tokens/s 与 peak memory 在账本中；⑦ 参数/FLOPs 账本自动打印并入库。
**S0 不用于论文结论；通过后主 2×2 等用户确认再启动。**

## 9. 引用清单（handoff §15 全文有效）

OLMoE arXiv:2409.02060（结构锚：64/8 与 32/4 granularity 对照、no-shared 决策、
31/3+1S matched 消融、LBL 0.01、z-loss 0.001）；Fine-Grained MoE Scaling
arXiv:2402.07871（G/X 记号）；FLAME-MoE arXiv:2505.20225（外部 size-family 基线）；
2609.11917（T≈20N、U/R 协议、ossification 方法锚）；Qwen3.8 arXiv:2608.30320；
golf arXiv:2607.01517；modded-nanogpt PR #201（pin commit）；nanoGPT OWT README
（数据源参考——本仓库语料为 full-163，见 §0 修正 A）。
