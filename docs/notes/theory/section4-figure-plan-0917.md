# Section 4 图片裁决（2026-09-17）

> 对象：飞书初稿 v2 `# 4. Forking 的最小机制模型与定量验证`（4.1 / 4.2 / 4.3）。
> 候选池：§4 现有图 4-1…4-4 + 本轮 module-arm 六图（`docs/figs/theory/fig_marm_*`）。
> 原则：一个论点一张图；每张图必须带 native 对照或理论曲线；同一张图内不混口径。

## 0. 理论链条与「图证」映射

| 链条环节 | 论点 | 图 |
|---|---|---|
| L1 结构 | 单点 input 注入 + tied readout ⇒ 表行 = 全词表 logit 偏置；挤出是 softmax 守恒内建 | §3.1 结构图 + `logits-architecture.md`（无需新图） |
| L2 现象级 | replay 使输出分布全体变尖（含 native）；n-gram 臂只在 train 侧多锐化，val 侧头部质量停滞 | **F4-1** logits_rank |
| L3 覆盖 | 损害沿 context 频率单调递减，集中在 novel/低频 | **F4-3** gap_by_ctxfreq |
| L3' 核 | 幅度 ∝ Good–Turing 缺失质量 M(f) | **F4-2** good_turing_kernel + **F4-4** val_damage_vs_missing_mass |
| L4 动力学 | 三效应随 epoch 分化：train 记忆 ↑、val 挤出/侵蚀相对 native 停滞 | **F4-5** three_effects |
| L4' 长程 | 6 pass 逐 f 账目 + margin/pass + table-direct vs backbone-carried | **F4-6** f_by_pass_accounting + **F4-7** table_cosine_and_margin |

## 1. 最终图单（7 张，3 新 4 留）

### 4.1 局部预测环境与建模假设
- **F4-1（新）** `fig_marm_logits_rank.png` — 4 臂 × train/val × epoch 1/2/3，top-10 rank 平均概率，线性 y。
  论点：replay 锐化是全体现象（native e1→e3 rank-1 0.16→0.36）；n-gram 臂 train 侧额外锐化（both 0.48），val 侧 rank-1 与 native 持平而 ppl 反而恶化 ⇒ 差异在「质量放在哪」不在「形状」。
  source: `module_arms_epoch_*_{e1,e2,e3}_ckpt`（seed 42, 337/674/1000 步, 128×），`eval_logits_rank.py`。
- 移入 **Appendix E.3**：`fig_marm_logits_rank_share.png`（per-position 归一头部形状，头部熵 1.3–1.9 各臂几乎一致）、`fig_marm_true_rank_cdf.png`。它们是 F4-1 的对照性 null result，正文一句话引用即可。

### 4.2 有限续接覆盖与频率幂律
- **F4-2（留）** `fig_v5_good_turing_kernel.png` — 理论核 g(f) 与 C·M(f)。
- **F4-3（新）** `fig_marm_gap_by_ctxfreq.png` — 配对逐位置 excess NLL（arm − native，同位置），按 trigram-context 频率 6 bin，3 臂 × 3 epoch，95% CI；右下 both@e3 贡献占比（novel bin 31% token 贡 57%）。
  论点：单调性 3 臂 × 3 epoch × 6 bin 全部成立；e1 时 bigram/both 在 seen context 上 excess 为负（早期正迁移），损害随 replay 才出现。
- **F4-4（留）** `fig_v5_val_damage_vs_missing_mass_bigram.png` — d_e(f) ≈ a_e·M_f 定量比例（主线 input 臂 2000 步 _fd）。

### 4.3 Replay 下的 logit 锐化与多 epoch 动力学
- **F4-5（新）** `fig_marm_three_effects.png` — 1×4：A 记忆（train seen-cont 按 f）、B 挤出（val seen-ctx/novel-cont, e1→e3）、C 侵蚀（val novel-ctx）、D 安全情形（val seen-cont），均带 native 对照。
  论点：「见过的 continuation 对其他 continuation 的效应」= B+C；native 同期 0.04→0.16 / 0.08→0.24 而 both 停在 0.05 / 0.10。
- **F4-6（留，从 4.2 移来）** `fig_v5_f_by_pass_accounting.png` — 6 pass × f 的 seen/novel 损害、margin/pass、table-direct vs backbone-carried。放 4.3 因为它本质是 pass 动力学；面板 (d) 直接接 §3.4 freeze 结果。
- **F4-7（留）** `fig_v5_table_cosine_and_margin.png` — margin +1.08/pass、传递系数 1−p̂≈0.87。

## 2. 弃用 / 降级

| 图 | 处置 | 理由 |
|---|---|---|
| `fig_marm_seen_novel.png` | 不入稿 | 2×3 总览，内容被 F4-3 + F4-5 完全覆盖且无 CI |
| `fig_marm_logits_rank_share.png` | Appendix E.3 | null result，支撑 F4-1 的解释但不承载论点 |
| `fig_marm_true_rank_cdf.png` | Appendix E.3 | F4-1 的 companion |

## 3. 入稿前必须处理的口径问题

1. **分类轴统一**。F4-4 / F4-6 用 bigram-context 频率、seen/novel 指 *continuation*；F4-3 / F4-5 用 trigram-context 频率、含 *novel context* bin。二者在 n-gram 层级上等价（trigram context (a,b,c) novel ⇔ bigram context (a,b) 的 continuation c novel），但正文须在 4.2 开头定义一次并在图注写明；或用 `context_freq_split.py` 已有的 `groups_bigramctx` 重算 F4-3 到 bigram 轴，与 F4-4/F4-6 完全对齐（推荐，改动小）。
2. **4.1 文字修正**。「带 n-gram 后 perplexity 降低很多、logits 变尖锐」只对 train 侧成立；val 侧 n-gram 臂 ppl 更高（both 191.6 vs native 35.8）且头部形状与 native 无异。应改写为：replay 锐化全体发生；n-gram 的作用是把锐化锁进按 context 分行的私有子空间，使 train 收益与 val 损害分离。
3. **horizon 标注**。F4-1/3/5 是 module-arm 1000 步（3 epoch）；F4-4/6/7 是主线 input 臂 2000 步（6 pass）。同一 128× 标准，但图注须写清 run_id 与步数，不得互相引用数值。
4. **e1 负 excess** 是新事实（F4-3），4.3「多 epoch」段应显式引用：模块早期给正迁移，gap 由 pass 数驱动，与 4.1「gap 只由相干更新次数控制」一致。

## 4. 关联记录
- 实验登记：`docs/experiment-log.md` §56
- 架构口径：`docs/notes/theory/logits-architecture.md`
- 图清单：`docs/notes/figure-inventory-0912.md`（待同步本裁决）
