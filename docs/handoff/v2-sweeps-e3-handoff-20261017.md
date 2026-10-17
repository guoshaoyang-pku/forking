# v2 扫描 + E=3 包络 Handoff（2026-10-17 15:45 初版 · 19:40 H200-only 冲刺增补）

> **相关文档**：登记 `docs/experiment-log.md` §73–§78；进度 panel `docs/figs/sweeps_20261017/panel.html`；E=3 结果表 `docs/appendices/epoch_envelope_e3_20261017/e3_results.csv`。
> **数字口径**：本文所有数字可追溯到 run_id（`data/runs_fixed/`，本地 harvest 在 `output/harvest_e3/`、`output/harvest_v21/`、`output/harvest_v224/`）。单 seed（42），无显著性断言。

---

## 0.5 H200-only 冲刺增补（16:00–19:35，87 臂，全部 H200）

用户 16:00 指令「A100 不允许用，只允许 H200。有卡就排上队跑实验」。已完成：

1. **§73/§77 三方对照全矩阵同机化（22 臂，全 H200）**。dict / hash / keephash × {n2, n3, n4, n5, 2-3, 2-3-4, 2-3-4-5} + nogram：
   - **覆盖率因子（keephash−hash）主导**：−0.75（n2）～ −5.67（n5）；联合 2-5 −4.39。
   - **碰撞隔离因子（dict−keephash）轻微反向**：+0.25 ～ +0.64 —— kept 集合内 ~1 上下文/行的轻度负载区，零碰撞反而让记忆更锐。方向反转假设待 per-token dump 核实。
   - hash H200 值与 §76 v2ord 3-seed 差 ≤0.11；nogram 0.2396 ≈ §74 的 0.245。
2. **epoch 剂量网格（E2/E6 × {10M, 49.7M} × 6 阶集合，18 臂）**：所有 (order-set, 池) 组合 gap 随 E 严格单调；49.7M 端 E=2→6 涨幅更大（n2 0.39→2.62；2-5 2.89→8.26），无平台化。n≥5 联合在 E=2 已达 5.4–6.2 → forking 第 2 个 epoch 内启动，「3 epoch 重 forking」是中段不是起点。
3. **E=3 seed 复跑（52 臂）**：v23@10M 12 + v24@10M 20 + v24@49.7M 20。误差棒 ±0.01–0.05 下：tri 斜率 +0.61（@10M）/+0.76（@49.7M）**严格单调**；n25 斜率 −0.11/−0.53 **严格单调下降**。tri 中间档 ±0.4 非单调确认为 10M 端 seed 噪声（l10 SD 0.363）。§78 模型结论升级为带误差棒版本。
4. **infra 事故**：360-2 用户配额 16:29 耗尽（$HOME 521G），8 臂 ENOSPC 死亡，360-2 全停摆；nogram 臂另受 concurrent commit 的 `enable_bigram` bug 影响（HEAD c6ef4fb 已修）。v2.1 补跑整体移植 360-1 隔离树 `forking-v2-v21`（code 冻结 md5 3346bd35，data symlink 只读）。**未 push（P5）。**

---

## 0. 一分钟版本

1. **E=3 epoch 固定包络 96/96 全部完成并回填（§78 done）。** 360-1 66 臂 + 360-2 30 臂，每池点实际 epoch 精确 3.00。panel 已更新（234 臂、新 §78 章节、audit 0 issues），两张新图 `fig_e3_pool_envelope.png` / `fig_e3_modelscale.png`。
2. **步数混淆定量：固定 2000 步把 gap 高估 4–6 倍。** n=2 @10M 池：1.67（E=3）vs 10.53（2000 步=30 epoch）。形状不变：n≤4 随池下降，n≥5 联合不动（6.3–6.8，3 epoch 即重 forking，与 DS/Kimi epoch-3 报告同形）。
3. **模型斜率衰减但不变号。** tri 对 backbone 17M→159M 的 gap 斜率从 +2.8 缩到 +0.7；n=2..5 联合对模型大小不敏感（E=3 下随模型略降）。§75 固定步数版高估了模型效应。
4. **§73 缺的两个 2345 联合臂已在 360-2 H200 补跑完成（15:52）。** A100-40GB 物理放不下 4 表（二次 OOM 退役）。同机配对：dict 4.4349 vs hash 8.2669（360-1 三 seed 8.228±0.024，跨主机差 0.039）→ 零碰撞在联合设置压掉 3.83 nats；dict 联合绝对值仍高于单阶 dict 臂，覆盖率因子留给 keephash。summary 已入 `output/harvest_v21/`，§73 已回填，panel 已更新。
5. **§77 keephash 7 臂从未启动。** a100_t1 部署目录已被清理；全 8 卡被另一租户 RL 任务占用（~30GB/卡，08:59 起）。重排队方案需要拍板（见 §3）。
6. **本轮 commit 未 push（P5）。**

---

## 1. 各线状态

| 线 | 登记节 | 状态 | 位置 |
|---|---|---|---|
| v2.2 表容量 R scan（60 臂） | §74 | 58/60 done（2 点 R=3M infeasible） | 360-2，harvest `output/harvest_v224/` |
| v2.3 数据池包络（30 臂，2000 步） | §74 | 30/30 done；49.7M 点归入 E=3 | 360-2 |
| v2.4 模型 ladder（50 臂，2000 步） | §75 | 50/50 done | 360-2 |
| seed 复跑 ×43/44（24 臂） | §74/75 | 24/24 done | 360-2 |
| **E=3 epoch 固定包络（96 臂）** | §78 | **96/96 done，已回填+出图** | 360-1 + 360-2，harvest `output/harvest_e3/` |
| v2 阶数扫描 n=2..8（27 臂） | §76 | done（4 臂 ECC 重跑） | 360-1 |
| v2.1 dict/hash 词典线（15 臂） | §73 | **15/15 done**（13 A100 + 2 个 2345 联合臂 H200 360-2 补跑） | harvest `output/harvest_v21/` |
| v2.1 keephash 对照（7 臂） | §77 | **未启动，A100 被租户占满** | a100_t1（部署已不存在） |

---

## 2. E=3 主要数字（final gap，seed 42）

池（M tok）→      10M　14M　19M　27M　37M　49.7M

| orders | 10 | 14 | 19 | 27 | 37 | 49.7 |
|---|---|---|---|---|---|---|
| n=2 | 1.67 | 1.58 | 1.38 | 1.19 | 1.08 | 0.98 |
| n=2,3 | 4.11 | 3.89 | 3.65 | 3.21 | 2.95 | 2.75 |
| n=2,3,4 | 5.90 | 5.72 | 5.44 | 5.00 | 4.68 | 4.50 |
| n=2..5 | 6.61 | 6.48 | 6.36 | 6.09 | 5.90 | 5.77 |
| n=2..6 | 6.67 | 6.55 | 6.45 | 6.37 | 6.31 | 6.34 |
| n=2..7 | 6.76 | 6.65 | 6.54 | 6.52 | 6.56 | 6.53 |

- 固定 2000 步对照（§74）：n=2 @10M 10.53、n=2..5 @10M 10.68、n≥5 联合 9.0–9.7 → E=3 全线下移 4–6×。
- 模型 ladder（E=3）：tri 2.97→3.69（17M→159M @10M，斜率 +0.7，固定步数版为 +2.8）；n25 5.97→5.82（@10M）、4.98→4.40（@49.7M），随模型略降。
- tri 中间档（35M/104M）在 10M/19M 有 ±0.4 非单调，超出跨卡噪声（0.004–0.134），判读以两端档为准；如需可补 seed 43/44。

## 3. 需要拍板

1. **360-2 配额清理（新阻塞）**：$HOME 521G 配额耗尽，360-2 无法接任何新实验。候选清理：`ngram-gap-lab-plan7`（82G）、`ngram-gap-exp`（75G）、`workdir`（51G）；另有本轮事故的部分目录 `ngram-gap-lab/logs/failed_launch/`。删除需你确认（P5）。
2. **>52M 池 / 1B 点语料**：raw parquet 在 ophis 无读权限（§74 blocker 未解）。1B×E3 ≈ 20,343 步 ≈ 1.6h/臂，语料一到即可排（360-1 健康卡 5 张）。
3. **本轮 commit 是否 push**（P5，等确认）。

## 4. 基础设施快照（2026-10-17 19:40）

- **360-1**：健康卡 1/2/4/5/6 全部空闲（本轮队列 19:35 清空）；gpu0/3/7 ECC 不可用；/data 99% 满（102G 余量，够小 dump）。隔离树 `forking-v2-v21` 保留（code md5 3346bd35，可复用）。
- **360-2**：**配额耗尽全停摆**（写 1KB 都失败），8 卡 0%；恢复需清理（见 §3.1）。
- **A100（perm/t1）**：按用户指令不再使用。
- **ophis-gpu**：用户配额满，未用。
- 代码：本轮所有 run 冻结于 md5 `3346bd35`（train.py）；launcher `run_e3_seed.sh` md5 `7200553d`（两机）、`run_v21_dict_lines.sh` md5 `81e239be`。

## 5. 本轮产物坐标

- 图：`docs/figs/sweeps_20261017/fig_e3_{pool_envelope,modelscale}.png` + `panel.html`（§78 章节新增）
- 脚本：`docs/plot_scripts/gen_e3_epoch_envelope_figs.py`（读 `output/harvest_e3/` + `output/harvest_v224/`）
- 数据：`output/harvest_e3/`（96 summary，平铺 `<run_id>.json`）；CSV 见文首
- 文档：`docs/experiment-log.md` §73 launch update 2 / §74 状态 / §77 状态更正 / §78 完成回填
