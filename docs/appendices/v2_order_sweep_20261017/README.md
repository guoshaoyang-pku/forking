# v2 order sweep (n = 2..8 separately, 2–5 jointly) · gap-vs-frequency

登记：`docs/experiment-log.md` §76。代码：commit `2b15d86`、`29de8b7`（分支 `v2_main`）。

## 文件

| 文件 | 内容 |
|---|---|
| `gapfreq_step2000.csv` | 每 (arm, seed, branch, F-bucket) 的 fixed-val 与 online-train 平均 loss、gap、token 数（step 2000） |
| `gapfit_step2000.csv` | 每 (arm, branch) 的对角幂律拟合（3 seed mean±SD）、val novel 占比、final gap |

生成脚本：`docs/plot_scripts/analyze_v2_order_gapfreq.py`、`fit_v2_order_powerlaw.py`；图：`plot_v2_order_sweep_gapfreq.py` → `docs/figs/v2_order_sweep_20261017/`。

## 设置

V5 标准：input 注入，每个 over-encoded order 一张 clean 单表 R = 2^20，backbone 6e-4 warmup_constant(100)，表 RMSProp(0,0.99)×128，bf16 无 compile，seed 42/43/44，train shard 1，val shards 2–10 + 6542，2000 步，freq/diag 每 10 步。
唯一变量：over-encoded order 集合。频率索引 `freq_index_orders2to8_s1.npz`（chunk 语义，n=2..8）。

## 拟合口径

- gap_bin = val mean loss − online train mean loss，同一 hit-count bucket、同一 logged step；novel（f=0）无 train 侧，不计。
- 只用 train 与 val 两侧均 ≥ 300 token 的 bucket；gap ≤ 0 的 bin 不取 log。
- 对角 = 第 n 个 order 的分支在 arm n 中；拟合 log gap = log A − α log F_mid，F_mid 为 bucket 几何中心，逐 seed 拟合后取 mean ± SD。

## 已知问题

1. **碰撞负载混淆（最重要）**：R 固定 2^20，各 order 的 distinct 训练上下文数 3.5M / 19.0M / 34.8M / 43.0M / 46.4M / 47.7M / 48.4M，即每行承载 3.4 / 18 / 33 / 41 / 44 / 46 / 46 个上下文。α 随负载单调下降，所以本批不能把 α 的变化归因于 n 本身。
2. **GPU 0 ECC**：360-1 gpu0 出现 32 个 uncorrected ECC。在 gpu0 上跑的 4 臂（n5 s42、n7 s43、n7 s44、nogram s42）已隔离到 `logs/suspect_gpu0_ecc/`（360-1），在健康卡重跑。同 seed 原值 vs 重跑 Δfinal-gap：0.012 / 0.032 / 0.134 / 0.004，与历史同 seed 跨卡 FP 噪声（§47）同量级，表中取重跑值。
3. **索引语义**：v2 索引按 chunk 计数（与模型看到的上下文一致）；标准 `freq_index.npz` 为拼接计数，bigram 上 20,027 个 key 的计数不同（跨 chunk 边界对），总位置数 49,692,672 vs 49,716,935。n=2 的 bin 与标准索引不能逐位比较。
4. **n ≥ 5 的 val novel 占比 85–98%**：对角 bin 只覆盖很小一部分位置，n = 6–8 的拟合只有 4–8 个有效 bin。

## 结果（final gap @2000 = mean ± SD over 3 seeds）

| arm | final gap | 对角 α | R² | val novel 占比 | distinct/R | val loss |
|---|---|---|---|---|---|---|
| n2 (bigram) | 2.607 ± 0.023 | 0.212 ± 0.003 | 0.84 | 0.04 | 3.4 | 5.1 |
| n3 (trigram) | 5.542 ± 0.023 | 0.353 ± 0.003 | 0.89 | 0.31 | 18.1 | 8.3 |
| n4 (4gram) | 6.324 ± 0.075 | 0.314 ± 0.006 | 0.93 | 0.66 | 33.2 | 9.7 |
| n5 (5gram) | 6.557 ± 0.143 | 0.247 ± 0.004 | 0.90 | 0.85 | 41.0 | 10.3 |
| n6 (6gram) | 6.583 ± 0.220 | 0.118 ± 0.006 | 0.76 | 0.94 | 44.2 | 10.5 |
| n7 (7gram) | 6.425 ± 0.128 | 0.065 ± 0.002 | 0.43 | 0.97 | 45.5 | 10.4 |
| n8 (8gram) | 6.287 ± 0.160 | −0.060 ± 0.012 | 0.29 | 0.98 | 46.2 | 10.4 |
| n2-3-4-5 joint | 8.228 ± 0.024 | bigram 0.027 · trigram 0.032 · 4gram 0.094 · 5gram 0.194 | | | | 8.5 |
| nogram | 0.235 ± 0.012 | — | | | | 3.3 |

F=1 处的 gap（对角）：n2 7.94、n3 7.06、n4 5.73、n5 5.30、n6 4.89、n7 4.24、n8 3.62。

## 观察（待 §76 判读）

- n ≤ 5：对角 gap 随 F 有干净的幂律下降（R² ≥ 0.84）；α 在 n=3 最大，之后随 n 下降。
- n ≥ 6：α → 0，R² 降到 0.76 → 0.29，与碰撞负载同步；不能与 n 效应分开。
- n ≥ 5 的 val loss 在 10.3–10.6，**高于均匀分布的 ln 8192 ≈ 9.01**：模型对未见上下文高度自信地出错，本批的 gap 主要是 novel 上的过度锐化，不是 seen 上的记忆。需要用 logit margin/entropy 直接检验（与 §53 同方向）。
- 联合臂 2–5 的 bigram 分支 α 仅 0.03（单独 bigram 臂 0.21）：联合训练改变了低阶的 gap-freq 形状。
