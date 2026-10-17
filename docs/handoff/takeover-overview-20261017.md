# v2 全景 · 2026-10-17 晚

> v2 目标：**缓解 forking，同时吃到尽可能大的 OE 收益**。机制线（为什么 fork）随论文发表已结题，
> 其结果在本文只作为缓解手段的依据。任务结构按飞书任务页
> （<https://mcna3q3ujim2.feishu.cn/wiki/EgPCwDniOiqsJDkBRA2cRdnxn6g>）。
> 聚合自三份 handoff：`v2-sweeps-e3-handoff-20261017.md`、`plan8-9-freeze0-countinit-handoff-20261017.md`、
> `../plans/plan-9-handoff-session-20261017.md`。数字已对 `output/harvest_{v21,e3}/` 抽查一致；
> run_id 全集见 `experiment-log.md` §73–§81。图：`../figs/v2_overview_20261017/tree.{html,png,svg}`。

## 1. 任务树

```mermaid
flowchart LR
  G["v2：缓解 forking<br/>+ 最大化 OE 收益"]
  G --> T1["① 生产级 PT 的 forking<br/>scaling 与外推"]
  G --> T2["② 端到端 OE 收益<br/>设计好的 OE"]
  G --> T3["③ 顺路：MoE × OE 模型家族"]
  T1 --> a1["n 扫描 2–8 + 2–5 联合<br/>✅ 已完成"]
  T1 --> a2["模型大小 × 数据<br/>🟡 只到 49.7M"]
  T1 --> a3["表大小 R 扫描<br/>✅ 58/60"]
  T1 --> a4["数据包络 6 种阶集合<br/>🟡 6/15 点，缺 1B"]
  T1 --> a5["gap-freq 幂律<br/>🟡 有碰撞混淆"]
  T1 --> a7["外推到 1B/20B/750B<br/>🟡 初版，两种形式分歧"]
  T1 --> a6["生产级 SFT<br/>❌ 未开始"]
  T2 --> b1["只留高频 + 无碰撞<br/>✅ gap 大降"]
  T2 --> b2["净收益 val 低于 nogram<br/>❌ 几乎都没做到"]
  T2 --> b3["更可训练的 OE<br/>❌ 未开始"]
  T2 --> b4["固定训练时长对比<br/>❌ 未开始"]
  T3 --> c1["MoE 基础设施 + S0<br/>✅ PASS"]
  T3 --> c2["FLAME 四臂家族<br/>❌ 未开始"]
```

✅ 完成　🟡 部分完成　❌ 未开始/未解决

## 2. ① 生产级 PT 复现 forking

### 2.1 已有结果

除非注明 3-seed，均为 seed 42。gap = fixed-val − online train。

| 子任务 | 结论 | 关键数字 | 出处 |
|---|---|---|---|
| n 扫描（2000 步） | gap 从 n2 升到 n5 后持平；n≥5 的 val loss 超过均匀分布（ln 8192） | n2 2.61 / n3 5.54 / n5 6.56 / n8 6.29；2–5 联合 8.23（3-seed） | §76，27 臂 |
| 表大小 R（2000 步） | R≤1k 不 fork；R≥100k 进入高原；高阶集合在 300k 出峰后回落 | n2 @1M 2.53；n2–7 峰 9.58 @300k | §74，58 臂 |
| 数据包络（E=3，10M→49.7M） | 低阶随池单调降；n≥5 联合几乎不降 | n2 1.67→0.98；n2–3 4.11→2.75；n2–7 6.76→6.53 | §78，36 臂 + 3-seed |
| 模型大小（E=3，17M→159M） | 只开 trigram 时大模型更 fork；n2–5 联合时大模型反而略少 | tri 2.02→2.78；n25 4.97→4.44（49.7M 池，3-seed ±0.01–0.05） | §78，60 臂 + 52 seed 臂 |
| epoch 剂量 E∈{2,3,6} | 严格单调，无平台；E=2 已明显 fork；重复次数比池大小更关键 | n2–5 @49.7M：2.89/5.77/8.26 | §78，18 臂 |
| 协议修正 | 固定 2000 步在小池上等于 30 epoch，把 gap 高估 4–6× | n2 @10M：10.53（2000 步）vs 1.67（E=3） | §78 |

### 2.2 scaling 与外推（详见 `../appendices/v2_scaling_20261017/README.md`）

生产级 PT 本身有 forking。第一条线的任务是从实验室规模外推到：
- 第一阶段：Dense 1B / 20B；MoE 取 FLAME-Large（1.3B 总量）。D = 20 × 总参数，重复 3–4 遍。
- 生产：Dense 100B 或 750B-A10B（DeepSeek-V4.1-Flash：552B + 196B Engram，8/16B 激活）。15–30T 语料，其中 100B 子集重复 3–4 遍。

| 结论（假设） | 依据 |
|---|---|
| trigram：`log gap = 2.01 + 0.108·log N − 0.170·log D`；沿 D/N 固定，gap ∝ N^−0.06 | 50 臂，E=3，log-RMSE 0.062 |
| 按可分幂律，目标规模 trigram gap 约 1.0–1.6（E=3–4），不会消失 | 外推 |
| 加 N×D 交互项同样拟合得好，但外推是「随规模恶化」 | 交互系数 90% 区间 0.033–0.081 |
| 生产的「100B 子集重复」对应 D_rep/N ≈ 0.13–1，正落在实验室测过的 D/N 区间 | 外推只发生在 N 轴 |
| 旧的「1B 时 forking 消失」：2000 步只训 0.3 遍，没有重复（E=1 的 gap 只有 E=3 的 2–6%） | §78 曲线 |

**在实验室内就能判别两种形式**：在 D/N=5 或 20 的线上，两者对 N 的趋势符号相反。

### 2.3 协议对齐（读历史数字用）

在 49.7M 池上：1000 步 ≈ 3 epoch，2000 步 ≈ 6 epoch。交叉核对：1000 步 bi+tri 对照
`frz0_ctrl_s42` 2.702 ≈ E=3 n2–3 2.754；2000 步 n2–5 hash 8.267 ≈ E=6 8.259。
**按 epoch 对齐后，不同协议的结果可以互读。**

### 2.4 未解决

- **D/N=5、20 的判别线**（L1/L2）与 clean n2-4 版本（L3）：数据用 ophis `data/manifest_full163/`（7.76B，与现有 run 同语料、同 tokenizer），搬到 360-1。E=4 曲线的前缀即 E=1/2/3，不另跑。
- **v24 的 n2-5 臂（75 个）漏传 `--bigram_clean_table`**，bigram 走了旧 2-hash 表，同条件 gap 比 clean 版低 0.7–1.0。「n2-5 联合时大模型 gap 反而降」需用 clean 表重跑后才能引用。
- **生产结构未测**：多 head hash 表、「新数据 + 重复子集」混合流、129k 词表、MoE 上的 OE forking。
- **无碰撞的 gap-freq 幂律**：§76 的 α 与「每行承载的上下文数」混在一起。188 臂（§78 166 + §73/§77 22）
  的 per-token dump 还在两台 360 上，没拉回本地分析。
- **R=3M 两个尾点**（n2–6、n2–7）：单卡 H200 放不下（138–165GB），需要表分片或多卡。
- **生产级 SFT**：v2 未开始。历史上有 continued-LM 对照（§57/58/68–72），不是 response-only SFT。

## 3. ② 端到端 OE 收益

### 3.1 continue-BPE + 无碰撞（飞书 2a-i，已完成）

dict = 只给 top-1M 高频上下文分配独占行；keephash = 同样只留 top-1M，但行仍用 hash；hash = 全部上下文 hash。
H200 同机，2000 步，seed 42：

| 阶 | hash gap | keephash gap | dict gap | 最好 OE 臂的 val | nogram val |
|---|---:|---:|---:|---:|---:|
| 2 | 2.57 | 1.82 | 2.14 | 4.41 | 3.32 |
| 3 | 5.65 | 1.59 | 2.13 | 4.25 | 3.32 |
| 5 | 6.59 | **0.92** | 1.17 | **3.76** | 3.32 |
| 2–5 | 8.27 | 3.87 | 4.43 | 5.57 | 3.32 |

- **只留高频（覆盖率）是降 gap 的主力**：hash→keephash 降 0.75–5.67。
- **在保留集合内再去碰撞，反而让 gap 略升**（+0.25～+0.64）。原因待 per-token 统计核实。
- **但 val 一个都没赢 nogram**：gap 降了，绝对 val 仍差 0.4–2.2。

### 3.2 净收益账本：val(OE) − val(nogram)，负数才算 OE 有用

| 手段 | 出处 | 约几个 epoch | Δval |
|---|---|---|---|
| 无约束 OE（input） | §43 | 6 | +3.58 |
| mask 低频 f≤8 | §43 | 6 | +1.33 |
| 只留 top-1M（keephash n5） | §77 | 6 | +0.43 |
| step-0 冻结 count-init 表 | §80 | 3 | ≈ +0.12（nogram 参照取自另一 run 的 3-epoch val 3.57，**非严格配对**） |
| 每 epoch hash reseed（input） | §43 | 6 | +0.81 |
| 每 epoch hash reseed（y / v 注入） | §43 | 6 | **−0.11 / −0.13**（唯一的净收益） |
| MoE S0，OE on vs off | §81 | 1 | +0.18（100M token，早期训练） |

**结论：OE 降 gap 已经有好几种办法，但「val 好过 nogram」只在 reseed + y/v 注入上出现过，幅度约 0.1。**
这是 v2 的核心未解决问题。另外 E=3 包络和 R 扫描**没有配 nogram 臂**，这两条线上的净收益现在算不了。

### 3.3 未解决

- **更可训练的 OE**（飞书 2a 后半句）：未开始，需要先定义候选。
- **保住表收益同时不 fork**：现有降 gap 手段都同时抬高 train loss，组合方案都还没跑，包括：
  - keephash + 每 epoch reseed；
  - mask 低频但保留 novel；
  - 第 1 个 epoch 后冻表或衰减 table LR；
  - count-init + 可学习。
- **固定训练时长对比**：没有按 wall-clock 对齐的比较；OE 的吞吐代价也没入账（MoE S0 里 OE 只慢约 1%）。
- **单 pass 下 OE 是否净赚**：现在没有可信的 1-epoch 配对数据（§66 的单遍对照因数据重复已作废）。

## 4. ③ 顺路：MoE × OE 模型家族（FLAME pivot）

依据 `../plans/plan-9-flame-pivot-addendum-20261017.txt`（与 Downloads 原件逐字节相同）。

| 项 | 状态 |
|---|---|
| 主线：每个尺度四臂 Dense / Dense+OE / FLAME-MoE / FLAME-MoE+OE | 定义完成 |
| 三点家族：Tiny 38M 激活/100M 总量，Small 115M/459M（主），Large 290M/1.3B | 定义完成 |
| 通用 MoE 代码、router 诊断、参数账本、CPU 测试 7/7 | ✅（旧 32R/top-4 降级为结构消融） |
| OE 审计 + 1B manifest（7.76B token 池，nested 前缀） | ✅ |
| S0（16R/top-4，OE on/off，100M×1 pass） | ✅ PASS |
| P1 `flame_profile`（64 routed/top-6 + 2×d_e 共享专家，第 1 层 dense，SwiGLU/RMSNorm/RoPE/无 bias/不共享 embedding） | ❌ |
| P2 dense twin（d_ff = 8×d_e：1408/2816/5632） | ❌ |
| P3 Small 四臂 S0（100M×1 pass） | ❌ |
| P4 Small 四臂 × 3 seed，1B × 3 pass（12 run） | 🔒 需用户放行 |

注意：pivot 用的是极简 setting 禁用的组件（RMSNorm/RoPE/SwiGLU），属于 P1 允许的「新架构变体」，
需在 experiment-log 写明偏离原因；与 v2 主线的数字不直接比较。

## 5. Todo

### 需要用户拍板

- [x] push：已推到 `MetaCircleAI/forking_v2` main（`53b77d4`，快进）。
- [x] 360-2 配额：用户与另一 agent 清理。
- [ ] **语料统一**：仓库所有现有 run 用的都是 ClimbMix（`karpathy/climbmix-400b-shuffle`，论文附录 B），不是 OpenWebText；full-163 也是同一语料池。ophis 与 360-1 上都没有 OWT。
  - 统一到 ClimbMix：现有 350+ 臂直接可比。
  - 统一到 OWT：v2 全部扫描需在新语料上重跑。
- [ ] L6 多 head hash 表属于新架构变体，开跑前需拍板。

### 实验，按优先级

1. **① 外推判别线 L1**：trigram，D/N=5，5 档 backbone，E=4，配 nogram（最长约 4h/臂，估）。之后 L2（D/N=20）。
2. **L3**：clean n2-4（对齐 DS 的 {2,3,4}）跑同样的线，替代有问题的 v24 n2-5 臂。
3. **② 净收益对照**：E=3 包络补 nogram 臂（约 10 臂），之后所有 OE 臂同时报 gap 和 Δval。
4. **② OE 收益组合臂**：
   - keephash + 每 epoch reseed；
   - keephash 改用 y/v 注入；
   - count-init + 可学习；
   - 第 1 个 epoch 后冻表。
5. **gap-freq 幂律**：拉回 188 臂的 per-token dump 分析。
6. **L4/L5**：R 随 D 放大；「新数据 + 重复子集」混合流（需新 loader 功能）。
7. **FLAME 家族 P1→P3**：语料与主线统一后开工；P4 等放行。

### 文档卫生

- [ ] `experiment-log.md`：两个 §76 重编号；§73/§75/§77/§79/§80/§81 标题状态过期；§73/§77 的 22 臂矩阵误放在 §79 节内。
- [ ] `claims-ledger.md` 停在 09-13：补 dict/keephash（C6 可从 UNRUN 改为本地代理已测）与净收益账本。
- [ ] panel：第四个进程挂起的两处更正（§76「仅 360-1」、幂律指数表述），等你确认设置图后改。
- [ ] 工作区里不属于任何 handoff 的改动（`agents.md`/`cluster-infra.md` 的 A100 行、`dispatch_multilayer_posttraining.py`、4 个 LoRA 脚本），需要确认归属。

## 6. 接手必读的工程坑

- ophis：python 用 `/data/home/guoshaoyang/ngram-gap-lab/.venv/bin/python`（系统 torch 2.6 不可用）；`data/tokenized` 软链只读；写盘与 `TMPDIR`/triton/inductor 缓存走 /data1；GPU0/5/6 常空。
- 360-1：gpu0/3/7 ECC 坏，用 1/2/4/5/6；/data 99% 满；隔离树 `/data/shared/guoshaoyang/forking-v2-v21` 可复用。
- 360-2：用户配额满，停摆。A100：用户已禁用，且 n2–5 四表在 40GB 上放不下。
- `train.py` 多会话共写，改前完整重读；跑前 md5 核对；远端清理用 `mv` 不用 `rm -rf`。
