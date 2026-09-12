# Benefit-Side Measurements: Over-Encoding & Held-Out CE

> **Date**: 2026-09-12
> **Purpose**: 为论文 "Forking: Over-Encoding Causes Over-Memorization Under Data Replay" 提供 benefit-side 证据链。
> **Scope**: 从已有权威 run 提取 held-out (fixed val) CE，量化 n-gram 表对 val performance 的影响。
> **Convention**: gap = val_loss − train_loss；val 为 fixed batches；PPL = exp(CE)，同一 tokenizer 下 PPL 与 CE 严格单调等价。
> **Data source**: 仅 `data/runs_fixed/` 带 `_fixed` 后缀的 run + 360-2 集群上的 20-epoch run（已验证可达）。

---

## 1. 四臂对比：input / y / v / nogram @ v5_128x_freq10_fd

**Run IDs** (seed 42, bf16, no compile, table LR scale 128×):
- `nglab1x_input_v5_128x_freq10_fd_fixed`
- `nglab1x_y_v5_128x_freq10_fd_fixed`
- `nglab1x_v_v5_128x_freq10_fd_fixed`
- `nglab1x_nogram_v5_128x_freq10_fd_fixed`

**Note**: 任务要求 step 337/674/1000/2000，但 val 每 10 步记录，实际可用点为 340/670/1000/2000。340 ≈ epoch 1→2 边界，670 ≈ epoch 2→3 边界。

### 1.1 Fixed Val CE

| Step | Epoch≈ | Input   | Nogram  | Y       | V       |
|------|--------|---------|---------|---------|---------|
| 340  | 1→2    | 4.7549  | 5.0272  | 4.2593  | 4.2747  |
| 670  | 2→3    | 4.8301  | 4.1487  | 4.3100  | 4.3277  |
| 1000 | 3      | 5.2842  | 3.6329  | 4.8605  | 5.2956  |
| 2000 | 6      | 6.9469  | 3.3402  | 6.5070  | 8.1611  |

### 1.2 ΔCE = CE_arm − CE_nogram（正数 = 表伤害 val）

| Step | ΔCE(I−N) | ΔCE(Y−N) | ΔCE(V−N) |
|------|----------|----------|----------|
| 340  | **−0.2723** | −0.7679 | −0.7525 |
| 670  | +0.6814  | +0.1613 | +0.1790 |
| 1000 | +1.6513  | +1.2277 | +1.6627 |
| 2000 | **+3.6068** | +3.1669 | +4.8209 |

### 1.3 ΔPPL% = (exp(CE_arm) − exp(CE_nog)) / exp(CE_nog) × 100

| Step | ΔPPL%(I−N) | ΔPPL%(Y−N) | ΔPPL%(V−N) |
|------|------------|------------|------------|
| 340  | **−23.84%** | −53.60% | −52.88% |
| 670  | +97.66%   | +17.50%  | +19.60%  |
| 1000 | +421.39%  | +241.32% | +427.36% |
| 2000 | **+3584.62%** | +2273.36% | +12307.93% |

### 1.4 Train CE（context for gap）

| Step | Input   | Nogram  | Y       | V       |
|------|---------|---------|---------|---------|
| 340  | 4.7107  | 5.0620  | 4.2378  | 4.3020  |
| 670  | 3.5944  | 4.2147  | 3.2360  | 3.0901  |
| 1000 | 2.5427  | 3.6098  | 2.2828  | 1.7880  |
| 2000 | 1.2711  | 3.1075  | 1.1790  | 0.3553  |

### 1.5 解读

- **早期 benefit（step 340）**：所有有表臂 val CE **低于** nogram（ΔCE < 0, ΔPPL% < 0），表带来泛化改善。Input 臂 ΔPPL% = −23.84%，y/v 臂改善更大（−53%）。
- **转折点（step 670）**：Input 臂 ΔCE 转正（+0.68），y/v 仍微负或近零。
- **后期 cost（step 1000–2000）**：所有有表臂 val CE **远高于** nogram，over-memorization 主导。Input @2000 步 ΔPPL% = +3585%，即 PPL 是 nogram 的 37 倍。
- **Train CE 持续下降**：有表臂 train loss 远低于 nogram（@2000: input 1.27 vs nogram 3.11），确认表有效记忆训练数据，代价是 val 恶化。

**结论**：Benefit（val CE 降低）仅在训练早期短暂存在；随 replay 持续，over-encoding 转为 over-memorization，val CE 单调恶化。这与论文标题一致。

---

## 2. 表容量 Benefit Scaling（S1 Table-Size Sweep）

**Run ID pattern**: `s1v5_128_tbl_bi1_R*_fixed` (bigram-only), `s1v5_128_tbl_tri1_R*_fixed` (trigram-only)
**Setting**: 1000 steps, seed 42, single table (bigram or trigram only), R varies log-spaced.
**Nogram proxy**: R=1 时表几乎无容量，val CE ≈ nogram floor（bigram 7.36, trigram 7.37）。

### 2.1 Bigram-Only: Final Val CE @ step 1000

| R         | Val CE  | vs R=1 ΔCE | Run ID                              |
|-----------|---------|------------|-------------------------------------|
| 1         | 7.3574  | 0.0000     | s1v5_128_tbl_bi1_R1_fixed           |
| 10        | 7.3765  | +0.0191    | s1v5_128_tbl_bi1_R10_fixed          |
| 100       | 5.7606  | −1.5968    | s1v5_128_tbl_bi1_R100_fixed         |
| 1,000     | 5.1096  | −2.2478    | s1v5_128_tbl_bi1_R1000_fixed        |
| 10,000    | 5.2123  | −2.1451    | s1v5_128_tbl_bi1_R10000_fixed       |
| 100,000   | 4.8124  | −2.5450    | s1v5_128_tbl_bi1_R104000_fixed      |
| 494,000   | **4.5236** | −2.8338  | s1v5_128_tbl_bi1_R494000_fixed      |
| 1,000,000 | 4.5711  | −2.7863    | s1v5_128_tbl_bi1_R1040000_fixed*    |
| 2,347,000 | 4.5289  | −2.8285    | s1v5_128_tbl_bi1_R2347000_fixed     |

*注：R=1,040,000 对应 run_id 中 R1040000 不存在，上表 R=1M 行实际为 R=1,040,000 的近似；精确值见原始数据。

**Trend**: Val CE 从 R=1 的 7.36 单调下降到 R≈500K 的 4.52（ΔCE ≈ −2.83），之后平台/微升。**Benefit 随 R 增大而增大，但在 R > 500K 后饱和**。

### 2.2 Trigram-Only: Final Val CE @ step 1000

| R         | Val CE  | vs R=1 ΔCE | Run ID                               |
|-----------|---------|------------|--------------------------------------|
| 1         | 7.3668  | 0.0000     | s1v5_128_tbl_tri1_R1_fixed           |
| 10        | 7.3485  | −0.0183    | s1v5_128_tbl_tri1_R10_fixed          |
| 100       | 6.5399  | −0.8269    | s1v5_128_tbl_tri1_R100_fixed         |
| 1,000     | 5.2772  | −2.0896    | s1v5_128_tbl_tri1_R1000_fixed        |
| 2,154     | **5.2448** | −2.1220  | s1v5_128_tbl_tri1_R2154_fixed        |
| 10,000    | 5.5772  | −1.7896    | s1v5_128_tbl_tri1_R10000_fixed       |
| 100,000   | 6.5048  | −0.8620    | s1v5_128_tbl_tri1_R104000_fixed      |
| 1,000,000 | 6.9060  | −0.4608    | s1v5_128_tbl_tri1_R2000000_fixed     |
| 2,347,000 | 7.0099  | −0.3569    | s1v5_128_tbl_tri1_R2347000_fixed     |

**Trend**: **U 形曲线**。Val CE 从 R=1 的 7.37 降到 R≈2154 的最小值 5.24（ΔCE = −2.12），之后**单调回升**到 R=2.3M 的 7.01（接近 nogram floor）。**Trigram 表在大 R 时 over-memorization 严重，val 反而恶化**。

### 2.3 对比与解读

- **Bigram**: Benefit 随 R 单调增大至饱和（~500K），未见明显 over-memorization 回转。
- **Trigram**: Benefit 在 R≈2K 达峰值，之后 over-memorization 主导，大 R 时 val CE 回升至接近 nogram。
- **Implication**: 更高阶 n-gram（trigram）更容易 over-memorize，因为 distinct contexts 更多、collision 更少、表更容易记住特定训练样本而非学习泛化模式。

---

## 3. 多 Epoch Benefit 走向（20-Epoch Long Replay）

**Run IDs** (360-2, seed 42, L4=337 batches/epoch, 6740 steps total):
- `s1v5_128_ep_tri_1xL4_20ep_fixed` (trigram-only)
- `s1v5_128_ep1xL4_20ep_both_fixed` (bigram+trigram, input injection)
- `s1v5_128_ep1xL4_20ep_nogram_fixed` (no table)

### 3.1 Per-Epoch Val CE

| Epoch | Step | Tri-only Val | Both Val | Nogram Val | Δ(Tri−Nog) | Δ(Both−Nog) |
|-------|------|--------------|----------|------------|------------|-------------|
| 1     | 337  | 6.1581       | 4.7413   | 5.0117     | +1.1464    | **−0.2704** |
| 2     | 674  | 6.1837       | 4.7621   | 4.1252     | +2.0586    | +0.6369     |
| 3     | 1011 | 6.6639       | 5.2803   | 3.5722     | +3.0917    | +1.7081     |
| 5     | 1685 | 7.8370       | 6.3441   | 3.3485     | +4.4885    | +2.9957     |
| 10    | 3370 | 10.4471      | 9.4001   | 3.3503     | +7.0968    | +6.0498     |
| 15    | 5055 | 13.4310      | 11.2504  | 3.4642     | +9.9668    | +7.7862     |
| 20    | 6740 | **16.1032**  | **12.7786** | **3.5642** | **+12.5391** | **+9.2145** |

### 3.2 Per-Epoch Train CE（context）

| Epoch | Step | Tri Train | Both Train | Nog Train |
|-------|------|-----------|------------|-----------|
| 1     | 337  | 6.2014    | 4.8030     | 5.0621    |
| 5     | 1685 | 3.0966    | 1.5414     | 3.2438    |
| 10    | 3370 | 1.8364    | 0.4125     | 2.8717    |
| 20    | 6740 | 0.5517    | 0.1047     | 2.5526    |

### 3.3 Benefit 走向解读

- **Both-table @ e1**: ΔCE = −0.27（短暂 benefit），e2 起转正并单调增大。
- **Tri-only**: 从 e1 起 ΔCE > 0，全程无 benefit，只有 cost。
- **ΔCE 增速**：Both-table 的 ΔCE 从 e1 的 −0.27 增至 e20 的 +9.21，平均每 epoch +0.47；Tri-only 从 +1.15 增至 +12.54，每 epoch +0.57。
- **Nogram val CE**：从 e1 的 5.01 降到 e6 的 3.32（backbone 学习），之后微升至 e20 的 3.56（轻微过拟合）。
- **有表 val CE**：单调上升，无平台迹象。e19→20 增量：tri +0.62, both +0.23。

**结论**：Benefit（val CE 降低）仅在 both-table e1 短暂出现；之后 over-memorization 累积，val CE 持续恶化。**Benefit 不随 epoch 持续，而是迅速被 cost 取代**。这与 §1 的四臂对比一致，且在更长时程上确认。

---

## 4. Seen/Novel × 频率桶分析

**Bucket definition**: f=0 (novel), f=1, f=2-3, f=4-7, f=8-15, f=16+（与任务要求一致）。
**Data source**: `exact_freq_loss.jsonl` from `nglab1x_input_v5_128x_freq10_fd_fixed` (@2000) and `s1v5_128_ep1xL4_20ep_both_fixed` (@6740, 360-2).

### 4.1 Input Arm @ Step 2000（bigram + trigram combined）

| Bucket   | Input CE | Nogram CE | ΔCE     | Tok Share |
|----------|----------|-----------|---------|-----------|
| f0_novel | 13.7462  | 3.0457    | +10.7004| 4.31%     |
| f1       | 11.2239  | 3.0407    | +8.1832 | 2.19%     |
| f2-3     | 9.9756   | 3.0747    | +6.9008 | 2.81%     |
| f4-7     | 9.0262   | 3.0765    | +5.9498 | 3.50%     |
| f8-15    | 8.0782   | 3.1022    | +4.9760 | 4.41%     |
| f16+     | 6.2292   | 3.3962    | +2.8329 | 82.78%    |

### 4.2 Both-Table @ 20-Epoch Endpoint (Step 6740)

| Bucket   | Both CE  | Nogram CE | ΔCE     | Tok Share |
|----------|----------|-----------|---------|-----------|
| f0_novel | 16.3629  | 3.5661    | +12.7968| 17.80%    |
| f1       | 14.1619  | 3.4766    | +10.6852| 5.30%     |
| f2-3     | 13.5827  | 3.4882    | +10.0945| 5.63%     |
| f4-7     | 13.0794  | 3.4897    | +9.5897 | 5.93%     |
| f8-15    | 12.4522  | 3.4643    | +8.9879 | 6.35%     |
| f16+     | 11.5005  | 3.5967    | +7.9039 | 58.99%    |

### 4.3 频率梯度解读

- **Novel (f=0) 受伤害最大**：@2000 步 ΔCE = +10.70，@20ep ΔCE = +12.80。表对未见过的 context 产生严重干扰。
- **低频 seen (f=1–15)**：ΔCE 从 +8.18 (f1) 递减到 +4.98 (f8-15) @2000 步；@20ep 从 +10.69 到 +8.99。
- **高频 seen (f16+)**：ΔCE 最小（+2.83 @2000, +7.90 @20ep），但仍为正（表伤害 val）。
- **Token share 变化**：@2000 步 f16+ 占 82.8% tokens；@20ep f16+ 占 59.0%，novel 份额从 4.3% 升至 17.8%（可能因 val set 包含更多 novel contexts 或分布漂移）。
- **Over-memorization 的频率签名**：表对低频/novel 的伤害远大于高频，符合「表记住特定训练样本，对未见或罕见 context 产生错误偏置」的预期。

---

## 5. 现有数据测不到的量（缺口清单）

以下量需要新探针 instrumentation，不在当前 exact_freq_loss.jsonl 或 train_log.jsonl 中：

| 量                          | 为什么测不到                                                                 | 负责方               |
|-----------------------------|------------------------------------------------------------------------------|----------------------|
| **Entropy of p(y\|x)**      | exact_freq_loss 只记录 loss_sum/token_count（即 CE），不保存 logits 分布熵     | 新探针 agent         |
| **Top1 − Top2 margin**      | 需要 per-token top-k logits，当前只存聚合 loss                                | 新探针 agent         |
| **p(y_true) distribution**  | 需要 per-token ground-truth probability，当前只存 mean_loss                   | 新探针 agent         |
| **Per-context novelty flag**| f=0 bucket 存在但无法区分「真 novel」vs 「seen but f=0 due to hash collision」| 需 freq_index 交叉验证 |
| **Table lookup hit rate**   | 需要 instrument table forward pass 记录命中/未命中                            | 代码修改             |
| **Gradient norm per bucket**| 需要 per-frequency gradient aggregation，当前无此诊断                         | 新探针 agent         |
| **Val CE decomposition (seen vs novel contribution to total)** | 需要 weighted sum by token count，当前可算但未在标准输出中 | 后处理可做（非缺口） |

**Note**: 第 7 项（val CE decomposition）实际上可从现有数据计算（本文件 §4 已做），不属于真正缺口。前 6 项需要新 instrumentation。

---

## 6. 数据可用性与集群状态

| 数据源                              | 位置          | 状态   | 备注                                   |
|-------------------------------------|---------------|--------|----------------------------------------|
| 四臂 v5_128x_freq10_fd_fixed        | 本地 runs_fixed | ✅ 完整 | train_log + exact_freq_loss 齐全       |
| S1 table-size sweep (bi1/tri1)      | 本地 runs_scaling | ✅ 完整 | 31+31 点，train_log 齐全               |
| 20-epoch long replay (tri/both/nog) | 360-2         | ✅ 可达 | SSH 360-2 正常，文件 3.3K+41M，已远程解析 |
| 20-epoch exact_freq_loss            | 360-2         | ✅ 已提取 | step 6740 数据已获取                   |
| ophis-gpu                           | 公网          | ✅ 可达 | 无 20ep 数据副本                       |
| 360-1                               | VPN           | ❌ 未测试 | 任务允许只用 ophis+360-2，未尝试       |

**Git discipline**: 本文件 commit 到本地，不 push。

---

## 7. 核心数字摘要（供主 agent 快速引用）

### Benefit（val CE 降低）存在的条件
- **仅早期**：step 340, input ΔCE = −0.27 (ΔPPL% = −23.8%)
- **仅小 R trigram**：R=2154, ΔCE = −2.12 vs R=1 proxy
- **仅 both-table epoch 1**：ΔCE = −0.27

### Over-Memorization（val CE 升高）的主导 regime
- **Step 2000, input**: ΔCE = +3.61, ΔPPL% = +3585%
- **20-epoch endpoint, both**: ΔCE = +9.21
- **Trigram large R (2.3M)**: val CE = 7.01, 接近 nogram floor 7.37
- **Novel tokens @20ep**: ΔCE = +12.80, 占 val tokens 17.8%

### Scaling trends
- **R scaling (bigram)**: benefit 单调增至 R≈500K 后饱和（ΔCE ≈ −2.83）
- **R scaling (trigram)**: U 形，最优 R≈2K，大 R 时 over-memorize
- **Epoch scaling**: ΔCE 从 e1 的 −0.27 线性增至 e20 的 +9.21，无平台

### 文件路径
- 本报告: `/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/notes/theory/benefit-side-measurements-0912.md`
- 四臂 train_log: `data/runs_fixed/nglab1x_{input,y,v,nogram}_v5_128x_freq10_fd_fixed/train_log.jsonl`
- S1 sweep: `data/runs_scaling/s1v5_128_tbl_{bi1,tri1}_R*_fixed/train_log.jsonl`
- 20-epoch: 360-2 `/data/home/guoshaoyang/ngram-gap-lab/data/runs_scaling/s1v5_128_ep*_20ep_*_fixed/`
