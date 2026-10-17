# Handoff · plan-8 infra + plan-9 MoE/OE 调研 + §79/§80 freeze0 家族（2026-10-17）

> 本会话（Verdent agent）成果交接。并行会话同期完成了 plan-9 P0/S0（moeify + MoEFFN +
> S0 PASS，commits `8029eb3`/`5401971`/`0d7d5ae`）与 §73–§78（覆盖/collision 矩阵、
> epoch 剂量、seed 重复），**不在本 handoff 声明范围内**，仅作状态引用。

## 1. 一句话总览

打通了 plan-8 的 index-first 快路径 infra 并定版（bf16 锁定），完成 plan-9 的权威
MoE/OE 调研与规格修订，然后用两个判决性对照对（§79 随机冻结 / §80 count-init 冻结）
把「forking 机制」收紧为三态判词：**点火 = 表在重复数据上被梯度塑形；燃烧 = backbone
共适应；内容质量与冻结时机都不是充分条件**。

## 2. 判决性数据（全部 seed 42，v5 1000 步协议，ophis-gpu H200）

| run_id | 臂 | final gap | train | val | 判决 |
|---|---|---:|---:|---:|---|
| `frz0_ctrl_s42` | 表可学习（当前代码对照） | **+2.7024** | 2.5568 | 5.2591 | 复现 LR 扫描 2.73 @128×（\|Δ\|=0.03）✓ |
| `frz0_table_s42` | step-0 冻结随机表（§79） | **+0.0642** | 3.7863 | 3.8505 | 随机分区**不 fork**（ctrl 的 2.4%）|
| `frz0_count_s42` | count-init + 冻结（§80，golf 等价物） | **+0.0219** | 3.6730 | 3.6949 | 真实信号 + 不可学习**也不 fork**，且 train/val 绝对值都优于随机冻结 |

count-init 统计：trigram 19,027,841 keys → 1,048,576/1,048,576 行非零，mean|row| 0.215
/ max 0.588（≤ init wte 尺度）。冻结参数 1,610,612,736 = 恰好两张 1M×768 表
（input 路线无 gate）。smoke 产物存档 `frz0_smoke_*` / `frz0_count_smoke_s42`。

拼合既有 causal 臂后的**三态机制判词**（写进 plan-9 §2.5B/C，论文机制章可直接引）：

1. step-0 冻结（随机 0.064 / count 0.022）→ 不 fork：**点火需要表可学习**；
2. e1/e2 冻结已写满的表 → gap 3.452 > ctrl 2.81（§72 @128×）、5.386 = ctrl 94%（§38 m3）：
   **维持不需要表侧更新**，backbone 接管甚至恶化；
3. 全程可学习 → 完整 forking（2.70）。
4. §80 新卖点：**信号效用与 forking 解耦**——不可学习但信息丰富的表改善绝对损失而
   不产生 train/val 分化；forking 是训练动力学现象，不是信息现象。

## 3. recipe 判读（用户 22:00 问询的结论，未落文档）

- **部署侧配方基本成型**（预防 > 治疗，证据链完整）：
  ① 数据设计最强：同 3 pass，1 shard 2.70 vs 20 shards **0.220**（`sftbase_ep20_tri_ckpt_fixed`），
  而 1 shard×20ep 达 12.67/15.55（§41）——唯一数据量 × 重复次数是主控变量；
  ② 续训阶段冻表白拿收益：§57/58 三线 frozen 全部 val 优于 trainable
  （ID 3.372 vs 4.376；Magicoder 1.554 vs 1.752；OpenR1 1.324 vs 1.480）；
  ③ 重放不可避免时 mask_low 有 val 绝对收益（le8：val 4.119 vs ctrl 5.259，gap 0.765，
  剂量单调；mask_high 互补臂证明 gap 载体是高频 context）；
  ④ 治疗：reseed 可重复击落（e1e2 双边界 gap 0.069）但非治愈。
- **缺的是「保住表的训练收益同时消灭 gap」**：所有降 gap 手段都在同步牺牲效用
  （freeze0 把 train 抬到 3.67–3.79；mask_low 每升档 train 单调上升）。
  最短补齐队列（均未跑）：mask f∈[1,t] + 保留 novel（le0 证明 novel 是净正贡献，
  现有 mask_low 语义恒屏蔽 novel，是明显空格）；table LR e1 后衰减 / table weight
  decay / 小 R；count-init + 可学习反向臂。
- 全部结论限单 seed、1000 步、8k 词表 toy 协议；规模化验证接用户的 2B/20B/1B 计划。

## 4. plan-9 调研成果（§1/§2.5/§3.0 已入文档）

- **Qwen3.8-Flash-Next 查证**：engram 形态 n-gram 记忆确认（arXiv:2608.30320，
  125B/6B act/+51B tables，20M rows，L2 单层注入 + contextual gating；vLLM 旗标
  `--engram-config`）；MoE 侧 512 experts top-10+1shared；**无 forking/频率治理证据**
  ——follow-up 空位。~180B 全参训练不可行（三岛集群单岛最大 1.13TB）。
- **MoE 权威锚点（弃自造配置）**：OLMoE-1B-7B（arXiv:2409.02060）原样缩放 →
  **E=32 / top-8 / d_e=384（granularity g=1/2，K/E=1/8）**，balance 0.01 + z-loss 0.001，
  norm_topk_prob=False，matched-active（active FFN 4.7M ≈ dense），total = 4× dense
  直接检验 2609.11917「total params 主因」。2609.11917 的 granularity 记法采用。
- **社区 OE 调研（§2.5）**：modded-nanogpt record 62 BHE——hash 公式与我们 clean 表
  第一族**独立同构**（外部效度）、zero-init、每层 λ 注入、CPU 端预计算（= plan-8
  index-first 的独立收敛）；gating/trigram/multi-hash 在其实现内是负结果（单实现
  caveat）。parameter-golf（arXiv:2607.01517）= 测试时 Katz backoff 2–7 blending，
  与我们 trained-in OE 不同类；由此 OE 三分类表（trained-in / test-time / frozen）
  与两个廉价对照臂（frozen 已 2/2 实测落地，见 §2）。
- **协议基线警告（重要）**：plan-9 旧引用的 dense 权威值 1.867/0.245 出自 v10/2000 步/
  旧 table-LR 协议；当前 1000 步协议对照 = **2.702**。2×2 网格的 dense 两臂须在当前
  协议重跑，不得与旧绝对值直比（已写入 plan-9 §D.5）。

## 5. plan-8 infra 定版（前半程，详录 `fastpath-precomputed-rows.md`）

- G1–G3/G5 落地：离线预计算行索引（`data/rows/shard_00001.{b,t}rows.u32.bin` 各 199MB，
  u32 与 token 流 1:1）+ 训练纯 gather + 载入逐位验证；129k 词表 uint32 loader +
  chunked CE；静态低频 mask 预防臂（keep_novel 默认 1）。
- **bench 定版**（`fastpath_{eager_base2,eager_pc2,compile_pc,compile_pc_fp8}_s42`，
  300 步 QC）：A 3.71 / B 1.00× / C **1.17×**（§18「compile 反慢」修复确认）/ D fp8 1.00×。
- **bf16 锁定**（用户拍板，`cd39f16`）：H200 0.1B 实测 fp8 无收益，`--fp8_torchao`
  保留为工程开关，实验线禁用，G4 🔒 不复评。

## 6. 本会话 commits（12 个，均未 push，P5 红线）

`49a543d` infra G1–G6 → `45306e4` bench 结果+环境 → `cd39f16` bf16 锁定 →
`dcfeeef` plan-9 初版 → `b801f00` 权威锚点+OE 调研 → `4dc45db` §79 注册 →
`c6ef4fb` dump_orders 修复 → `24b2623` §79 回填 → `11223e6` §80 注册+实现 →
`bac3244` primes 结构修复 → `f1b7659` device 修复 → `714dd3a` §80 回填。

新代码：`code/cluster/run_v5_freeze0_pair.sh`、`code/cluster/run_v5_freeze0_count.sh`、
`--table_init_count`（count_init_clean_tables）、CPU 测试 `code/tools/check_count_init_tables.py`
（PASS：加权均值/零未见行/确定性/hash 对齐/负计数防御）。

## 7. 工程坑（后续实验必读）

- **quota 陷阱**：`data/tokenized` 是指向 quota-dead home 的软链；写盘一律走
  `data/rows`（/data1 实目录），`TMPDIR`/`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR`
  全部重定向 /data1，否则表现为 `Disk quota exceeded` / InductorError。
- **Python**：ophis 系统 python3 是 torch 2.6，须用项目 venv
  `/data/home/guoshaoyang/ngram-gap-lab/.venv/bin/python`（torch 2.9.1）。
- **同步纪律**：每轮 rsync 后 md5 核对（本会话 6 次全对齐：…`54563dda`…）。
- **远端清理**：`rm -rf` 会被本地安全 guard 拦截；用 `mv` 归档 smoke 产物代替。
- **并发编辑**：train.py 处于多会话共写状态（v2 dict-mode、epoch-anchored、MoE 集成），
  改前必须完整重读；本会话修过 v2 合并遗留的 `cfg.enable_bigram`→`enable_bigram_ve`。

## 8. 下一步（按优先级）

1. plan-9 S1：MoE forking 复现 2 臂 × 3 seeds（S0 已由并行会话 PASS，launcher
   `run_p9_s0.sh`，PYTHON 需 venv override）——直接接 OLMoE 规格。
2. recipe 缺口臂（§3 清单，建议先跑 mask f∈[1,t]+keep novel 单变量对）。
3. 2×2 网格 dense 两臂在当前 1000 步协议重跑（对照 2.702 校准）。
4. 12 个 commit 待用户批准后 push（P5）。
