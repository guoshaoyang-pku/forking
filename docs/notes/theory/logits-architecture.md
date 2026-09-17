# Logits 架构与输出空间分类结构（§56 机制图的口径说明）

> 2026-09-17 整理。代码引用以 commit `4449758` 之后的 `code/train.py` 行号为准。
> 本文回答两个问题：① 本模型 logits 如何形成、n-gram 表在其中的结构地位；
> ② `fig_marm_three_effects` / `fig_marm_gap_by_ctxfreq` 用的「分类结构」
> 的准确定义、与架构的对应关系、已覆盖与未覆盖的轴。

---

## 1. Logits 形成通路（input 注入主线，逐行核实）

```
idx (B,T)
  │
  ├─ x = wte(idx) + wpe(pos)                      train.py:751-752
  ├─ x += Δ(idx)   ← n-gram 注入（唯一注入点）      train.py:765-767
  │     Δ(idx_t) = bigram_row[k_b(prev_t, idx_t)]
  │              + trigram_row[k_t(prev2_t, prev_t, idx_t)]
  │     k_b  = ((prev·p1) ^ (idx·p2))  mod 2^20    train.py:692
  │     k_t  = ((prev2·p3) ^ (prev·p4) ^ (idx·p5)) mod 2^20   train.py:700
  │     clean 单表 K=1、无 gate、行宽 = n_embd      train.py:705-745
  ├─ 8 × Block（attention 对 input 模式不再加 n-gram；
  │    v/y 分支仅在 injection_position = v/y 时激活）  train.py:291,301,811-843
  ├─ x = ln_f(x)                                   train.py:844
  └─ logits = lm_head(x)，lm_head.weight ≡ wte.weight（tied）
                                                    train.py:347-349,845
```

三个结构性推论（写作时可用）：

- **R1 表行 = 全词表 logit 偏置**。tied readout 下，
  logit(p) = ⟨W_e(p), ln_f(·)⟩；表行 Δ_c 与 W_e 同处 n_embd 空间，
  故 context c 的表行等价于一个 8192 维的 logit-bias 轮廓
  b_c(p) = ⟨W_e(p), Δ_c⟩ + backbone 改写项。n-gram 模块不是「特征」，
  是**逐 context 的加性 logits 先验**。
- **R2 挤出是架构内建的**。softmax 质量守恒：CE 把 seen continuation 的
  logit 抬高 Δlogit 时，同 context 下所有其他 token 的概率被等比压缩。
  效应 B（挤出）不需要额外机制，是 R1+softmax 的直接推论。
- **R3 更新按行稀疏、幅度 128×**。RMSProp(0,0.99)、table_lr_scale 128：
  每次更新只写当前 context 命中的那一行（backbone 全网共享更新）。
  一个 context 的行只在它出现时被拉向当前 continuation 的梯度方向——
  「见过的 continuation 记入自己 context 的行」在参数层面就是逐行的。

## 2. 输出空间的分类结构（机制图口径）

两个正交维度，均以 **train shard（shard_00001.bin，49,716,936 tok）计数**：

- **X 轴 · context 状态**：trigram (a,b,c) 在 train 的出现次数 f。
  novel (f=0) / f=1 / f=2-3 / f=4-31 / f=32-511 / f=512+。
- **Y 轴 · continuation 状态**：4-gram (a,b,c,y) 是否在 train 出现
  （seen / novel continuation）。

| | seen continuation | novel continuation |
|---|---|---|
| **seen ctx (f≥1)** | 记忆 cell（A/D：train 收益；val 上 f=1 时 0.62>native 0.50） | **挤出 cell（B）**：val rank-0 0.10→0.06，native 0.11→0.26 |
| **novel ctx (f=0)** | 不存在（trigram 层面） | **侵蚀 cell（C）**：val true_prob 0.10 vs native 0.24，31% token、占总 excess 57% |

对应关系：A=效应 A（记忆增益，train 侧）；B=效应 B（同 context 竞争挤出，
R2 的直接体现）；C=效应 C（novel context 泛化侵蚀，backbone 共适应代价，
**val 损失主源**）；D=A 在 val 侧的 seen 行（安全情形）。

### 已入数据的其他轴（`marm_seen_novel.json`）

- **bigram 级 context 分组**（`groups_bigramctx`：以 (b,c) 计 f2、
  以 (b,c,y) 判 seen）——注意 C cell（trigram novel ctx）内部混合了
  「bigram 已见」的位置，bigramctx 分组可将其细分；当前图未画。
- **candidate 侧 seen 掩码**：top-32 预测中每个 p 是否构成 train 见过的
  4-gram（`seen_top`），汇总为 `seen_mass_topk`（train 侧 0.43→0.67，
  图 `fig_marm_seen_novel` 右下）。true-token 侧与 candidate 侧是
  不同的划分，前者描述「真值在哪」，后者描述「概率质量去哪」。
- 每 context 的 unigram 频率 `y_unigram_mean`（已存均值，**未分箱**）。

### 未覆盖 / 后续可加的轴

1. C cell 按 bigram-ctx 是否 seen 细分（数据已存，一张小图即可）；
2. continuation 的 unigram 频率分箱（区分「常见词」与「罕见词被挤出」）；
3. rank>32 的质量去向（npz 只存 top-32，`p_at_rank` 有全谱但无逐位置）。

## 3. 一句话结构

> n-gram 模块在架构上是**逐 context 的加性 logits 先验**（tied readout
> 使表行直接投影到词表方向），训练按行稀疏、128× 幅度进行；由此，
> 记忆（A）、挤出（B）、侵蚀（C）三效应分别是「行内抬升」「softmax
> 质量守恒」「backbone 共适应」的结构性后果，而非三个独立现象。
