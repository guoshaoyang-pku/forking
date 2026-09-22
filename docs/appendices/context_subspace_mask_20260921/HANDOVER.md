# HANDOVER · context loss masking 实验（experiment-log §62，论文 fig7 / fig7*）

> 交接对象：后续维护 fig7/fig7\* 口径与正文的人。**论文展示口径 = fixed（C_A），online 仅作对照。**
> 登记簿：`docs/experiment-log.md` §62（含 2026-09-22 晚口径修正段，必读）。

## 1. 实验问题

随机选 2% 的 trigram context 集合 A（B = 等量 disjoint 对照集），从某 epoch 边界起把 A 位置
的 per-token loss 置零（renormalize 到保留位置），观测 A 的 gap contribution 是否继续增长。
结论：**mask 后 A 的 gap 贡献跨 epoch 不变（甚至缓降）；正常训练的 A/B 持续上升** ⇒
gap 增长需要子空间内部对 seen continuation 的 sharpening 更新；纯 backbone 漂移使
train/val 两侧同步退化，不产生 gap。

## 2. run 矩阵（全部 seed 42，shard-1 fixed replay，均已完成）

| 批次 | run_id 模式 | 步数 | 说明 |
|---|---|---|---|
| type-mode 12 runs | `s1v5_128_marm_{nogram,bigram,trigram,both}_ctxmask_{ctrl,e1,e2}_fixed` | 1011 / 674 / 1011 | A/B 按 trigram **类型** 2% |
| type-mode 4-epoch 延展 | `s1v5_128_marm_both_ctxmask_{ctrl,e1,e2}_l4_fixed` | 1348 | **fig7 数据源** |
| type-mode 4-epoch nogram | `s1v5_128_marm_nogram_ctxmask_e1_l4_fixed` | 1348 | **fig7b 数据源**（2026-09-22 晚补跑） |
| row-mode 5 runs | `s1v5_128_marm_both_rowmask_{ctrl,e1,e2}_fixed`、`s1v5_128_marm_nogram_rowmask_{e1,e2}_fixed` | 同上 | A/B 按表**行** 2%（hash 碰撞判别） |

`e1` = mask 从 epoch 2 起点（step 338）生效；`e2` = 从 epoch 3 起点（step 675）生效；
`ctrl` = 不 mask（intervention=none）但同样记录 A/B。

集群位置：`ophis-gpu:/data1/guoshaoyang/ngram-gap-lab/data/runs_scaling/<run_id>_fixed/`
（每个含 `context_gap.jsonl`（每 10 step 一条）、`train_log.jsonl`、`epoch*_model.pt`、`final_model.pt`、`summary.json`）。
本地证据副本：`docs/appendices/context_subspace_mask_20260921/runs_scaling/`（只有 context_gap.jsonl）、
`context_kl/`（KL 评测 JSON）。

## 3. setting（极简契约，差异加粗）

vanilla nanoGPT 8L·6H·768D，vocab 8192，seq 2048，learned PE，LN，tied emb。
bigram+trigram input 注入（nogram/bigram/trigram 臂用 `--enable_bigram/trigram 0` 关闭），
clean 单表 R=2^20 两表，表 RMSProp `(0.0,0.99)`、LR 128×（0.0768），
backbone AdamW `(0.8,0.95)` wd 0.1 lr 6e-4，`warmup_constant(100)`。
train shard 1，val shards `2..10,6542`（不重叠），device batch 72，total batch 147,456 tok，
bf16 不 compile，seed 42。**差异项：`--epoch_batches 337 --fixed_train_probe 4 --context_eval_ab 1`（ctrl）
或 `--intervention random_context_loss_mask --intervention_epoch {1,2} --save_epoch_checkpoints --save_final_model`（mask runs）；
row-mode 另加 `--context_set_mode row`。**

A/B 选取（两次运行确定性一致）：
- type-mode：`freq_index.npz` 的 19,027,841 个 distinct trigram 类型，`np.random.default_rng(42)`
  permutation，前 2% = A（380,557 类型），次 2% = B。membership 用 sorted-keys searchsorted。
- row-mode：1,048,576 个 trigram 表行，同样 rng(42) 取 A/B 各 20,972 行；membership 直接用
  clean-table 哈希 `(prev2·p0 ^ prev·p1 ^ idx·p2) mod R`（`_BASE_TRIGRAM_PRIMES[0]`），
  与双表臂真实表哈希逐位一致（CPU smoke 验证），nogram 臂可选出相同行集。

干预语义：boundary step（`epoch_batches × intervention_epoch`）起，`F.cross_entropy(reduction="none")`
后把 A 位置置零并对保留位置取均值（`_context_loss_keep_mask`）；表参数与 optimizer state 不动。

## 4. 代码版本（权威源 = 本地 lab repo 已 commit 版本）

- 干预 + A/B + 评测：`code/train.py`，commits `b87bb20`、`6dede12`（type-mode），
  `db1ac75`（row-mode），`59372b67…`（online 累积器 gate 修：`_context_set_memb is not None`），
  `a4ed6ae`（diag setup 块归位）。运行用 md5 `507a60b95c52f9034f782cffb96d5265`（+gate fix 后版本）。
- 启动器：`code/cluster/run_ctxmask_ab.sh`（type-mode 12）、`run_ctxmask_ab_long.sh`（l4 延展）、
  `run_rowmask_ab.sh`（row-mode 5）。
- KL 工具：`code/tools/eval_context_kl.py`（全 vocab 精确 KL，fixed val batches，
  `--context_set_mode {type,row}`，`--pair name=run@ckpt,run@ckpt`）。
- 绘图：`docs/plot_scripts/plot_paper_fig7_ctxmask_real.py`（见 §6）。

## 5. 测量口径（三种，论文用第一种）

1. **fixed（C_A，canonical，论文口径）**：`C_S = Σ_{i ∈ S ∩ (probe∩val)} q_i [l_val,i − l_tr,i]`，
   q_i = context 在 fixed-val 的位置占比；tr 侧为 4 个 **fixed** train probe batches、val 侧为 4 个
   fixed val batches，当前模型逐 10 step 重评。无位置伪影；`n_join_ctx_A = 1072`
   （probe∩val 共同出现的 A context，偏向高频子集，但**跨 step、跨 run 完全同一子集**，比较有效）。
   ctrl 的阶梯（边界跳升 = 表行重拟合压低 tr_probe，epoch 内衰减 = backbone 漂移）是真实动力学。
2. **online（Cpos_A，对照）**：`val_share × (val_mean − tr_online_mean)`，tr_online 为当前 batch
   A 位置的 online loss。**有 shard 位置伪影**：epoch 初窗口落在早 shard 位置（无 backbone
   recency，tr 偏高 2.28→3.95），epoch 末反之 → 锯齿 + 边界骤降。只能「epoch 末对 epoch 末」读，
   或用 epoch 平均口径（剔除边界短窗 `tr_online_n_A < 20000`）。
3b. **KL 轨迹（fig7b，2026-09-22 晚重设计）**：ref = 各 run 的 epoch-1 末 ckpt，累计
   `KL(p_ep1 ‖ p_t)` 于 epoch 2/3/4 末，type-mode e1_l4 runs（mask 均从 epoch 2 起），
   `context_kl/context_kl_traj_e1_l4.json`。A(masked)/B(unmasked)/other：

| epoch 末 | both A | both B | nogram A | nogram B |
|---|---|---|---|---|
| 2 | 0.465 | 0.466 | 0.884 | 0.891 |
| 3 | 1.109 | 1.129 | 1.604 | 1.657 |
| 4 | 1.819 | 1.781 | 1.960 | 2.032 |

   **读法**：① 首个 masked epoch 双表臂 masked-KL 约为 nogram 的一半（0.47 vs 0.88）——
   表把 masked context 的分布漂移减半；② 优势随 epoch 数收窄（epoch 4 时 1.82 vs 1.96），
   累计共享参数漂移最终主导；③ 臂内 masked vs unmasked 差异很小（nogram B 略高 +0.01~0.07；
   both 基本持平，epoch 4 甚至反转）——无论有无表，masked context 的分布仍随 backbone
   大幅漂移，但该漂移同步作用于 train/val 两侧，不进 gap（与 fig7a 自洽）。
3c. **restricted-KL（子空间内口径，回答"other 本底"质疑，2026-09-22 23:55）**：用户指出
   full KL 可能被 "other"（非 seen/val continuation 的 ~8k 背景 token）主导。新口径：
   对每个 val 位置的 trigram context 取 **S(ctx) = train shard 中该 context 的已见
   continuation（4-gram）∪ 该 context 在 val 中出现的真值 token**，p/q 各自在 S 上重归一化后算
   `KL_S = Σ_{i∈S} p̃ log(p̃/q̃)`；同时给出**精确分解**
   `KL_full = Z_ref·KL_S + Z_ref·log(Z_ref/Z_new) + 外部项`。
   实现：`eval_context_kl.py --restrict_seen_novel`（support 用 train shard 4-gram CSR +
   fixed val batches 的 (ctx,y) CSR，预建一次 6 对共享）；数据
   `context_kl/kl_restricted_e1_l4.json`（同 fig7b 的 6 对，ref=epoch-1 末）。
   |S|：mean 10.7 / median 2 / p99 125。

| epoch 末 | both A rKL | both B rKL | nogram A rKL | nogram B rKL |
|---|---|---|---|---|
| 2 | 0.209 | 0.268 | 0.378 | 0.370 |
| 3 | 0.476 | 0.634 | 0.723 | 0.782 |
| 4 | 0.767 | 0.939 | 0.861 | 0.940 |

   Z(S)（输出质量落在 S 上的占比，mean）轨迹：

| epoch 末 | both A | both B | nogram A | nogram B |
|---|---|---|---|---|
| ref(ep1) | 0.447 | 0.445 | 0.232 | 0.231 |
| 2 | 0.430 | 0.564 | 0.348 | 0.354 |
| 3 | 0.425 | 0.655 | 0.415 | 0.433 |
| 4 | 0.423 | 0.700 | 0.447 | 0.468 |

   full KL 三项分解（in-S 加权 + S 质量项 + 外部项，both_e4 A 为例 0.329+0.171+1.319=1.819 ✓ 闭合）：

| pair | set | in-S 加权 | S 质量项 | 外部项 | 外部占比 |
|---|---|---|---|---|---|
| both_e2 | A | 0.090 | +0.051 | 0.324 | 70% |
| both_e2 | B | 0.083 | −0.073 | 0.457 | 98% |
| both_e4 | A | 0.329 | +0.171 | 1.319 | 73% |
| both_e4 | B | 0.287 | −0.122 | 1.617 | 91% |
| nogram_e2 | A | 0.099 | −0.058 | 0.843 | 95% |
| nogram_e4 | A | 0.206 | −0.083 | 1.837 | 94% |

   **结论（四层）**：① **用户猜测成立——full KL 的 70–95% 来自 S 外的背景本底**（共享 backbone
   的宏观漂移），这不是子空间内的 sharpening；② **子空间内（rKL）臂内对比首次干净显现**：
   both 臂 masked A 恒 < unmasked B（比值 ~0.78–0.82 跨三个 epoch 稳定），nogram 臂 masked≈unmasked
   （比值 0.92–1.02）——有表时 mask 停住子空间内重排，无表时停不住；③ **Z(S) 是最锋利的证据**：
   both 臂 unmasked B 质量从 0.445 灌到 0.700（+57%，sharpening 的直接定量签名），masked A
   冻结在 0.447→0.423（−5%）；nogram 臂 masked A 照样 0.232→0.447（+93%，与 unmasked +103%
   几乎无差）——没有表，"子空间"住在共享 backbone 里，mask 掉 context 的 loss 并不能阻止其
   分布被其他 context 的更新锐化；④ S 质量项的符号本身即机制：unmasked B 恒负（质量流入 S），
   both masked A 恒正（质量流出，遗忘），nogram 两臂皆负（都被锐化）。
3. **KL 三探针（旧 fig7\* 布局，已被 fig7b 取代）**：`KL(p_boundary ‖ p_end)` 逐 fixed-val 位置全 vocab 精确计算，按 A/B/other 分组。
   三个探针：`prev_normal`（epoch1→2 边界，正常 epoch 的漂移量）、`masked`（epoch2 边界→mask run 终点）、
   `normal_ctrl`（epoch2 边界→ctrl 终点）。type-mode A-KL（mean，`context_kl.json`）：both 0.505/0.469/0.608、
   nogram 0.919/0.584/0.728、bigram 0.436/0.334/0.458、trigram 0.445/0.707/0.640（trigram-only 异常：
   masked 高于 prev 与 ctrl，原因未查）；row-mode（`context_kl_row.json`）：both 0.540/0.502/0.692、
   nogram 0.904/0.657/0.744。**结论：mask 降低所有臂的 A 漂移（少 2% 更新），双表臂绝对漂移最小
   （0.47–0.50 vs nogram 0.58–0.66），但任何臂的 masked KL 都远非"几乎没有"（残余跨 context
   backbone 漂移）；该漂移同步作用于 train/val 两侧，故不进 gap（与 fig7 读法一致）。**

关键数字（fixed 口径，l4 runs）：ctrl C_A 峰值 0.002→0.006→0.0085 逐 epoch 抬升；
e1 mask 后 C_A ≈ 0.0003 持平三个 epoch；e2 mask 后 C_A ≈ 0.0015 持平两个 epoch。
（online epoch 平均口径：e1 0.0094→0.0076→0.0059、e2 0.0277→0.0241，ctrl_A 0.0080→0.0263→0.0401。）

## 6. 图与复现

```
python3 docs/plot_scripts/plot_paper_fig7_ctxmask_real.py \
    --runs_root docs/appendices/context_subspace_mask_20260921/runs_scaling --infix ctxmask   # 全部 fig7 变体
python3 docs/plot_scripts/plot_paper_fig7_ctxmask_real.py \
    --kl_json docs/appendices/context_subspace_mask_20260921/context_kl/context_kl_row.json \
    --arms nogram,both --infix rowmask                                                        # fig7*
```

产物（`docs/figs/paper/`）：
- **`fig07_context_mask_fixed_perstep.{pdf,png}` ← fig7a 论文用（fixed 口径逐 step，4 epochs）**
- **`fig07b_kl_traj_{lines,bars,nested}.{pdf,png}` ← fig7b 三版候选（KL 轨迹，epoch 2–4）**
- **`fig07b_rkl.{pdf,png}` ← fig7b（restricted-KL 单面板：KL_S 轨迹，ref=epoch-1 末=0）**（脚本 `--rkl_json`，数据 `kl_restricted_e1_l4.json`，口径 §5.3c；用户拍板只画 KL，Z(S) 仅作分解分量不进图）
- `fig07_context_mask_perstep.{pdf,png}`：fixed + online 双面板对照
- `fig07_context_mask_epochmean.{pdf,png}`：online 口径 epoch 平均（无伪影版）
- `fig07star_context_kl.{pdf,png}`：fig7\* type-mode 四臂（主版）
- `fig07star_context_kl_rowmask.{pdf,png}`：fig7\* row-mode 两臂对照
- `fig07_context_mask.{pdf,png}`：online 逐 step 原始锯齿版（备查）

## 7. 已知坑（血泪）

1. **DataLoader fork 与父进程 cmdline 相同**——看到"同 cmdline 双进程"先查 PPID / 有无 CUDA
   context，勿杀子进程（杀后主进程 futex 死锁，trigram_e2 曾因此报废重跑）。
2. online 累积器 gate 必须是 `_context_set_memb is not None`（row-mode 无 `_context_set_keys`）。
3. 边界额外 eval 造成 2–5 step 的短累积窗（n≈6k），epoch 平均时须剔除（n<20k）。
4. C_A 的 joined 子集偏高频 context（probe 只有 4 batches）；跨 run 比较有效，绝对量不代表
   A 全集的 gap 贡献。
5. 协作者原图的 epoch 编号是 0-indexed 习惯（"from epoch 1"= 第二遍）；本仓图统一 1-indexed
   （"masked from epoch 2" = step 338 起）。
