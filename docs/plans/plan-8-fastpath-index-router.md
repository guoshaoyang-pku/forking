# Plan 8 · Index-first 基础设施与 n-gram 路由路线图

> 状态：执行中（2026-10-16 夜起）。
> 本文是 infra 路线 SSOT：用户拍板的方向逐条记录在 §1，目标 G1–G8 在 §2，验收标准在 §3。
> 实验事实与 run_id 一律回到 `experiment-log.md` / `claims-ledger.md`；本文不产生新结论。

---

## 1. 用户拍板的方向（2026-10-16，原话要点）

1. **Index-first 成为默认 setting**：后面默认先离线扫描 n-gram 并做 index。
2. **低频 mask 走 continue-BPE 形态**：mask 低频 context 大概率以 continue BPE
   （仅保留高频）的形式做，因此**提前 index** 是前提。
3. **推理路径**：先做 index；推理时 hash 然后查表；**假如 hash 里该 context 被禁用，
   直接返回零、不施加 OE（over-encoding）**。
4. **5gram 路由是后续默认方向**：默认传给一个 5gram 路由；路由看 5gram 窗口里
   包含了几个高频 n-gram，甚至可能是一个小模型做组合。
5. **多后缀问题（核心设计点）**：5gram ABCDE 中，DE 和 BCDE 可能都是常用后缀。
   要决定只留 BCDE（最长匹配，SCONE 语义）还是两路都加进去；都加就要有权重。
6. **权重可学习且低延迟**：每一路的权重可以根据传入的 x 或上个 token 最终的 x
   来训练，但**延迟必须仍然非常小**。
7. **非后缀 n-gram 也可能值得加**：比如 BCD 很重要，即使上个 token 已经考虑过
   BCD，当前 token 也值得再考虑一遍。但它不是后缀，需要特殊处理，可能做变换
   （例如 RoPE 类的位置变换）。这里有设计空间。

## 2. 目标与执行顺序

| # | 目标 | 内容 | 状态 |
|---|---|---|---|
| G1 | 预计算行索引快路径 | `precompute_row_index.py` 离线算好 bigram/trigram 行索引（uint32，与 token 流 1:1 对齐）；`train.py` 训练步纯 embedding gather；input 注入路径消除 per-block 死代码（VE gather + mask searchsorted 在 `input` 位置从不被消费）；载入时首批逐位验证 | ✅ 2026-10-16 完成（T1/T2/T6/T7 全过） |
| G2 | 129k 词表支撑 | uint32 token loader；chunked checkpoint CE（129,280 词表全量 logits ~38GB 不可行）；diag/probe/eval 走分块安全路径 | ✅ 2026-10-16 完成（T3/T5 过；compile CPU smoke 与 eager 同值） |
| G3 | 静态低频 mask 预防臂 | 离线按 freq 阈值预算每位置 enabled 位（u8）；快路径 gather 后乘有效性位；novel（f=0）可选保留——§35.8 `mask_low_le0` 证据：仅禁 novel 反而抬 gap（2.945 > 2.724），novel 读出对 val 是净正贡献 | ✅ 2026-10-16 完成（T4 过；e2e smoke 验证位逐位 OK） |
| G4 | FP8 harness | torchao float8，仅 backbone Linear（排除 lm_head / 全部 n-gram table / gate）；table 与诊断 forward 保持 bf16/fp32；配对验收协议见 §3 | ✅ 代码就绪 2026-10-16（**需 GPU smoke**，未验收不进主线） |
| G5 | CPU 数值一致性测试 | 离线索引 vs 动态 hash 逐位一致；pre_rows 快路径 loss 逐位一致；chunked CE 等价；u32 loader 往返 | ✅ 2026-10-16 完成（7/7 PASS） |
| G6 | H200 bench + 文档 + commit | 四臂：eager / eager+pc / compile+pc / compile+pc+fp8；§18（compile 反慢 3.5x）修复的验收 | ✅ 2026-10-16 launcher 就绪（`run_v5_fastpath_bench.sh`），**待 GPU 实测** |
| G7 | 词典表（top-K dictionary table） | 真·continue-BPE 形态：表只含高频 context（离线扫描选出），无 hash 碰撞；禁用 context 不占行。依赖 G1 的扫描/index 管线。**下一步实现** | ⬜ 后续 |
| G8 | 5gram 多后缀路由 | 设计稿 §4；等 G1–G7 bench 数据后立项 | ⬜ 设计中 |

run_id 后缀约定（P4 口径纪律）：

| 后缀 | 语义 | 数值口径 |
|---|---|---|
| `_pc` | 预计算行索引（+ 静态 mask 位） | 与基线**逐位一致**（载入验证强制），同一口径类 |
| `_c` | `--compile` | 数值同分布但非逐位（融合改变浮点求和序），单独口径类，配对验收后并入 |
| `_fp8` | FP8 backbone | 单独口径类，必须过配对验收才可进主线 |

## 3. 验收标准

- **G1**：`test_fastpath_cpu.py` 全过：离线索引 == 模型动态 hash（逐位，含 chunk 边界位）；
  `forward(pre_rows)` loss 与动态路径逐位一致；载入首批验证在人工破坏行文件时拒绝启动。
  hash_reseed / perfect-map / 多层注入与 `_pc` 互斥（启动即报错）。
- **G2**：`--token_dtype u32` 往返正确；`ce_chunk_positions` 开启时 per-token loss 与
  全量路径逐位一致（row-local 运算）；8k 词表主线默认路径不变（auto 规则不触发）。
- **G3**：有效性位与 `GlobalFrequencyIndex` 阈值语义一致（f≥t 或 f==0&keep_novel）；
  首批逐位验证覆盖。
- **G4**：GPU smoke 前不进主线；协议 = 300–500 步 bf16 配对臂，gap 轨迹形状一致才采纳。
- **G6**：bench 接受标准：compile+pc ≥ 1.5× eager 基线吞吐（预期；实测为准）；
  eager+pc final gap 与 eager 基线差 < 1e-6。

## 4. G8 设计稿 · 5gram 多后缀路由（记录，待立项）

**输入**：位置 t 的 5gram 窗口 W = (A,B,C,D,E_t)。候选 n-gram = W 的所有后缀
（E, DE, CDE, BCDE, ABCDE）∩ 词典（G7 的 top-K 表），外加可选非后缀项（如 BCD）。

**待决策问题**（按用户要点）：

1. **单路 vs 多路**：最长匹配（只留 BCDE，SCONE 语义，对齐问题已解决但 CD 行的
   更新计数被抢）vs 多路相加（每路有权重，计算次数不对齐但覆盖「DE 也常用」）。
   判别实验：8k 词表上最长匹配臂 vs sum-read 臂（现有管线的对齐消融，先做）。
2. **路由权重**：`w_r = g_r([x_t ; x_{t-1}])`，g 为 32 维 gate 线性（现有
   `ve_gate_channels=32` 先例，O(routes) FLOPs，延迟可忽略）；残差
   `Σ_r w_r · V_r(row_r)`。初始 w=0 起步等价无 OE，训练信号走同一 loss。
3. **非后缀项（BCD）**：上个 token 已消费过 BCD 的读出；再消费需要位置区分——
   候选方案：(a) RoPE 类旋转把「两步前」的 context 读出变换到当前位置；(b) 独立的
   offset embedding；(c) 仅在 gate 判定有用时启用。纯设计空间，先记录。
4. **路由器形态**：v1 = 静态权重表（按 routes 组合数查表）；v2 = 32 维线性 gate；
   v3 = 小模型组合。每级都要过延迟预算（与 embedding gather 同量级）。

**与现有证据的接口**：mask 剂量曲线（`causalv5m_mask_low_le{0,1,2,4,8}`）=
「词典外=禁用」的干预版；G7 词典表 = 预防版；路由 = 多词典分支的组合版。

---

## 5. 本次落地清单（G1–G6）

- `code/precompute_row_index.py`（新）：离线行索引 + 静态 mask 有效性位 + `rows_meta.json` 契约。
- `code/train.py`：`--row_index` / `--token_dtype u32` / `--ce_chunk_positions` /
  `--static_low_mask` / `--fp8_torchao`；快路径 + 死代码消除 + 载入验证 + chunked CE。
- `code/ngram_freq.py`：uint32 支持；`compute_per_token_loss` 走内存安全路径。
- `code/diag_worker.py`：pair/logit payloads 走模型分块方法（129k 安全）。
- `code/tools/test_fastpath_cpu.py`（新）：G1/G2/G3 的 CPU 验收。
- `code/cluster/run_v5_fastpath_bench.sh`（新）：G6 四臂 bench。
- `docs/notes/method/fastpath-precomputed-rows.md`（新）：契约与 run_id 后缀口径。
