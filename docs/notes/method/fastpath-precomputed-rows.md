# Fast-path 基础设施：预计算行索引 / 静态低频 mask / chunked CE / FP8

> plan-8（`docs/plans/plan-8-fastpath-index-router.md`）的实现契约。
> 本文件写「怎么用、语义边界、验收」；方向与目标看 plan-8；结论与 run_id 回登记簿。

## 1. 预计算行索引（`_pc`）

**工具**：`code/precompute_row_index.py`。对 train shard 离线算好每个 token 位置的
bigram/trigram 表行号（clean 单表 K=1 第一 hash 族），与 token 流 1:1 对齐：

| 文件 | 内容 |
|---|---|
| `shard_<id>.brows.u32.bin` | bigram 行号（uint32，每 token 一项） |
| `shard_<id>.trows.u32.bin` | trigram 行号 |
| `shard_<id>.bmask.u8.bin` / `.tmask.u8.bin` | 静态 mask 有效性位（仅 `--freq_index --min_count` 时生成） |
| `rows_meta.json` | 契约（primes、R、vocab、dtype、阈值、shards） |

**运行侧**：`train.py --row_index`（可选 `--row_index_dir`，默认 `data_dir`）。
训练 forward 变纯 embedding gather；hash 算术全部消失。

**边界（自动拒绝，防静默 fork）**：仅支持 input 注入 + clean 单表双支；
`hash_reseed`、`--bigram_perfect_map`、多层注入、legacy K=2 一律启动报错。

**逐位验证（强制）**：启动时用独立数据集实例取第一个 batch，用模型自己的 hash
现算行号与离线行号 `torch.equal`，任何不一致拒绝启动（含 static mask 位对
freq index 的逐位校验）。行号流长度 ≠ token 流长度同样拒绝。
验证不消耗训练流（独立实例），训练流首 batch 不移位。

**口径**：`_pc` 与基线**逐位一致**（同 step 同 loss），不构成新口径类；
建议 run_id 仍加 `_pc` 后缀便于追溯。

**存储**：8 bytes/token（双支行号）+ 可选 2 bytes/token（mask 位）。
shard 1（49.7M tokens）≈ 398MB(+100MB)。只为 train shards 预计算；
val/eval/diag 走动态 hash 路径，无需行号文件。

## 2. 静态低频 mask 预防臂（`--static_low_mask t`）

**语义**：`enabled = (f >= t) | (novel & keep_novel)`，从 step 0 生效。
这是「词典外 = 禁用 → 该位置不施加 OE」的预防形态，对应用户拍板的
index-first 默认：推理时 hash 查表，禁用 context 直接零贡献。

**与 `mask_low_freq` 干预的区别（重要）**：干预臂恒禁 novel；预防臂默认
`--static_low_mask_keep_novel 1` **保留 novel**——依据 `causalv5m_mask_low_le0_e1`：
仅禁 novel 反而抬 gap（2.945 > 对照 2.724），novel 表读出对 val 是净正贡献。
两臂不得组合（启动校验拒绝）。

## 3. chunked CE 与大词表（uint32）

- `--token_dtype u32`：>65535 词表必须（`prepare_data.py --token_dtype u32` 产出）。
- `--ce_chunk_positions`：`0`=auto（全量 bf16 logits > 4GiB 才开，chunk ≈2GiB）、
  `-1`=关、`>0`=强制。129,280 词表 × 147,456 positions 全量 logits ≈ 38GB bf16，
  必须分块；per-token CE 行独立，分块不改数值（T3 验收 atol 1e-6）。
  实现为 `torch.utils.checkpoint`（非重入）逐 chunk 重算，峰值 ≈ h + 单 chunk logits。
- diag/probe 内存安全：`ngram_freq.compute_per_token_loss` 路由到
  `model.per_token_losses`；logit-stats 路由到 `model.projection_stats`
  （熵/margin/ptrue/top20 逐 chunk 流式，从不物化全量 logits）。
- 8k 词表主线 auto 不触发，路径逐位不变。

## 4. FP8（`--fp8_torchao`，未验收前不进主线）

torchao float8，仅 backbone Linear；**lm_head 与全部 n-gram 表/gate 排除**
（table 路径保持 bf16/fp32；诊断 forward 保持 bf16 `_fd` 管线）。
torchao 不在位时启动即报错。**采纳协议**：300–500 步 bf16-vs-fp8 配对臂，
gap 轨迹形状一致才并入主线；fp8 run 加 `_fp8` 后缀，单独口径类。
预期收益 1.2–1.5×（仅 backbone GEMM 受益；先落 compile+pc 再上 fp8）。

## 5. torch.compile

`--compile` + `--row_index` 才是推荐组合。§18 的 3.5× 反慢根因是动态 hash 的
graph break；行号预计算 + input 注入死代码消除后 forward 为纯 gather+GEMM。
compiled run 数值同分布但非逐位（融合改变求和序），加 `_c` 后缀。
CPU smoke（fp32）实测 compiled 与 eager final gap 完全一致；
H200 bf16 吞吐以 `run_v5_fastpath_bench.sh` 实测为准。

## 6. Bench 协议（`code/cluster/run_v5_fastpath_bench.sh`）

单卡四臂（eager / eager+pc / compile+pc / compile+pc+fp8），v5 标准 setting，
300 步（工程 QC，非科学 run）。

### 6.1 ophis-gpu 实测（2026-10-17，GPU 7 / H200，300 steps，seed 42，v5 setting）

run_id（`data/runs_fixed/` @ ophis-gpu /data1 工作区）：
`fastpath_eager_base2_s42` · `fastpath_eager_pc2_s42` ·
`fastpath_compile_pc_s42_v2` · `fastpath_compile_pc_fp8_s42_v2`
（首轮 `fastpath_eager_{base,pc}_s42` 等为 rows 传输优化前的数据，保留作 provenance。）

| arm | steps/s | speedup | final gap |
|---|---:|---:|---:|
| A eager base | 3.71 | 1.00× | −0.0445 |
| B eager + pc | 3.70 | 1.00× | −0.0426 |
| C compile + pc | 4.35 | **1.17×** | −0.0433 |
| D compile + pc + fp8 | 3.72 | 1.00× | −0.0436（32 个 backbone Linear 转 fp8） |

**验收结论**：

1. **§18 修复确认**：compile 从 3.5× 反慢变为 1.17× 加速。1.5× 预期未达到——
   8L/768D @ 147k tokens 是 GEMM 主导，动态 hash 本就不是主要开销；pc 的价值
   在于让 compile 可用（graph break 消除）而非 eager 提速。
2. **B 轨迹干净**：同臂重跑噪声 |A−A2| = 3.6e-3 > 快路径信号 |B2−A2| = 1.9e-3
   （flash-SDPA 原子归约的 run-to-run 非确定性）。逐位性由 CPU T2 位级测试 +
   启动行号逐位验证承担；原 <1e-6 判据对非确定性 GPU 训练不适用，改为噪声界。
3. **FP8 在 0.1B/8k-vocab 规模不提速**（cast 开销吃掉 GEMM 收益），运行稳定、
   gap 同分布。按 §4 协议不进主线；到 1B / d=1536 目标规模再复评。

### 6.2 ophis-gpu 环境备注（后续实验前必读）

- 工作区 `/data1/guoshaoyang/ngram-gap-lab`；`data/tokenized` 与
  `data/freq_index.npz` 是指向 /data3 home 的**软链**。home（/data3）与
  root fs（含 /tmp）的用户配额 grace 已过期 → **写入会被拒**。
- 对策：行索引/编译缓存/临时文件一律放 /data1——`--out_dir data/rows`、
  `TMPDIR=/data1/guoshaoyang/tmp`、`TRITON_CACHE_DIR=/data1/guoshaoyang/triton_cache`、
  `TORCHINDUCTOR_CACHE_DIR=/data1/guoshaoyang/inductor_cache`。否则 triton
  launcher 的 gcc 以 `Disk quota exceeded` 失败（表现为 InductorError，
  首轮 bench arm C/D 即此因）。
- Python：`/data/home/guoshaoyang/ngram-gap-lab/.venv/bin/python`（torch
  2.9.1+cu128，只读可用）；torchao 0.13.0 在 `/data1/guoshaoyang/pylibs`
  （PYTHONPATH；0.18 需 torch≥2.11 不兼容）。
- rows 文件：`data/rows/shard_00001.{b,t}rows.u32.bin`（各 199MB，/data1 本地）。

## 7. 本次实现附带的两个鲁棒性修复

1. **空 epoch 自旋守卫**：shard 切不出一个完整 device batch 时（chunk_size 与
   shard 长度不匹配，如漏传 `--sequence_len`），fixed replay 迭代器过去会
   `while True` 零产出自旋；现于 `TokenizedShardDataset.__init__` 直接报错。
2. **rows_requested 修正**：`--row_index` 且 `--row_index_dir` 为空（=默认
   data_dir）时，`_load_rows` 曾误判「不需要行号」而早退返回 None；
   现由显式 `rows_requested`（= `--row_index`）驱动。

## 8. 测试

`python3 code/tools/test_fastpath_cpu.py`（CPU，无需 GPU）：
T1 离线索引 == 动态 hash（逐位，含 chunk 边界）· T2 快路径 forward 逐位一致 ·
T3 chunked CE 等价 · T4 静态 mask 位语义（keep_novel 0/1）+ 禁用位零贡献 ·
T5 u32 + >65535 词表行号数学 · T6 拒绝路径 · T7 流长度/缺失报错。
