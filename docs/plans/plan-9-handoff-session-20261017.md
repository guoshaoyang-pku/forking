# Plan-9 会话 handoff（2026-10-17 晚，FLAME pivot 后）

> 交接对象：接手 plan-9 的下一个 agent。范围：本会话全部成果 + 当前状态 + todo。
> 权威顺序：`plan9_flame_pivot_addendum.txt`（已归档
> `docs/plans/plan-9-flame-pivot-addendum-20261017.txt`，冲突时以它为准）
> > `plan9_moe_oe_agent_handoff.txt`（已归档
> `docs/plans/plan-9-handoff-gptpro-20261017.txt`）> 本文档（只记录事实与状态）。
> 登记簿见 `experiment-log.md` §79/§80/§81；设计文档 `docs/plans/plan-9-moe-oe-baseline.md`。

---

## 1. 成果清单（全部已 commit，未 push——P5）

### 1.1 机制实验（§79/§80，freeze 对照系）

| 实验 | run（ophis `data/runs_fixed/`） | 结果 |
|---|---|---|
| §79 freeze0 对照对 | `frz0_ctrl_s42` / `frz0_table_s42` | 可学习表 gap **2.7024** vs step-0 冻结随机表 **0.0642**（2.4%） |
| §80 count-init 冻结 | `frz0_table_countinit_*` | gap **0.022**（golf 式复刻：统计信号但不可学习 → 不 fork） |

三态结论：**点火需要表可学习（梯度 sharpening）；燃烧是 backbone 侧共适应
（e1/e2 冻结已写满表照样 fork 且可恶化）；随机/统计初始化的冻结表都不 fork**。
这直接改写了 plan-9 §2.5 的 OE 分类表（frozen 行从预测升级为实测）。
副产品修复：`train.py` dump_orders `cfg.enable_bigram`→`enable_bigram_ve`
（v2 dict-mode 合并遗留 bug，commit `c6ef4fb`）。

### 1.2 MoE 基础设施（P0，commit `8029eb3`）

- **`code/moe.py`**：`moeify(G,X,S)` 合法性校验与推导（d_e=d_ff/G、E_routed=G·X−S、
  K_routed=G−S，matched-active 恒成立）；`MoEFFN` dropless token-choice（fp32 router
  全 softmax 后 top-k、默认不 renorm、Switch aux + z-loss、负载统计 drain、
  ossification panel 捕获、eval 期诊断暂停）；`params_ledger` 精确参数/FLOPs 账本
  （X 按 matrix-only 公式）。
- **`train.py` 集成**：`--moe_g/--moe_x/--moe_shared_units/--moe_aux_weight/
  --moe_zloss_weight/--moe_norm_topk_prob/--moe_panel_interval` 七旗标；dense 路径
  零改动（回归 diff 6e-6，GPU 噪声内）；loss contract（train CE/gap = 纯 LM loss，
  aux/z 单独入 `moe_diag.jsonl`，val 是纯 CE 路径）；`moe_panel/step*.npz`
  （16,384-token 固定 panel × 8 层 topk_idx/probs/margin）；`summary.json` +
  `moe_ledger.json` 自动记账（吞吐/峰值显存/终态 aux/z）。
- **`code/tools/test_moe_cpu.py` 7/7**（Mac + ophis 双环境）：moeify 合法性、
  G1/X1 dense 三方等价（fp32 1e-6 / bf16 2e-2）、expert 置换等变、assignment 守恒
  （routed + shared hook 双计数）、routing/aux/z 公式逐项对拍（uniform router
  aux≡1.0）、autocast 下 router fp32、账本恒等式（含 31R/3+1S matched 对照）。
- `prepare_data.py`：`--lib_data_dir` 覆盖 + 单分片 dummy-test 修复（lib.py 拒绝
  train/test 重叠）。

### 1.3 审计与数据（P1，commit `5401971`）

- **OE 审计**（`code/tools/oe_param_audit.py`，真实实现核算）：表参数
  **1,610,612,736**（bigram_ves+trigram_ves 各 805,306,368，与 §79 冻结计数逐位
  一致）；表侧显存 **18.0 GB**（fp32 参数 6.0 + RMSProp state 6.0 + 瞬时梯度 6.0）；
  touched rows（shard-1 流）：bigram 唯一行 1,010,522/2^20（**96.4%**）、trigram
  **100% 饱和**（1B 协议下双表全占用，碰撞必然——频率 bin/mask_low 解释的前提）；
  回退 hash 公式与 precompute row 文件逐位一致。
- **dense 两臂复用判定**：九项条件中 data manifest / token order / epoch 定义三项
  不满足 → 历史 dense 基线（1.867/0.245/2.702 等）**全部重跑**，不与 p9 新协议直比。
- **1B manifest**（`code/tools/build_p9_manifest.py` → ophis `data/manifest_full163/`）：
  池 = 162 train 分片全部 packed rows = **7,760,093,184 tokens**；固定 row permutation
  （seed 20260917，与 model seed 独立）物化为 158 个 `shard_%05d.bin`（train.py 零改动
  即可消费，epoch 重放精确）；nested 前缀：s0_pilot_100m = 48,852 rows、main_1b =
  488,281 rows、scale_2b = 976,562 rows（`--train_max_tokens` token 精确截断）；
  val = 6542 整分片 + panel = 前 8 行；sha256 全档（源×162 + permutation + 物化×158 +
  tokenizer×3）。偏差登记：.bin 无 doc id，粒度 = 2049-token packed row（跨分片打散
  文档顺序/来源局部性，非 doc-level）。
- **全池 tokenize 完成**：163/163 bins（6 worker 并行）；copied 旧分片 byte-identical；
  独立重 tokenize 的分片（shard 66 双写竞态）也 byte-identical——tokenizer 确定性
  额外验证。

### 1.4 S0 工程 pilot（§81，commits `6894230`+`0d7d5ae`）

G4/X4 = 16R/top-4/d_e768，OE on/off 两臂，100M × 1 pass × seed 42（678 步），
manifest nested 前缀。**验收全过**：

| 臂 | train | val | gap | tok/s | peakGB | dead | loadH |
|---|---:|---:|---:|---:|---:|---|---|
| oeon | 4.3043 | 4.2704 | −0.034 | 188,515 | 74.6 | 全 0 | 0.997–1.000 |
| oeoff | 4.1511 | 4.0944 | −0.057 | 190,310 | 62.6 | 全 0 | 0.991–0.999 |

无 NaN；assignment 守恒（68 block × 8 层 assign_frac 和恒 1.000000）；moe_diag
68 行 + panel 7 npz 字段齐全。观察（非结论）：per-layer aux ≈ 1.0（打印的 8.06
是 8 层和）、1-pass 下 gap ≈ 0（forking 检验在 epoch 2/3）。
launcher `code/cluster/run_p9_s0.sh`（`PYTHON` 覆盖：ophis 系统 python3 是
torch 2.6，必须用项目 venv）。

### 1.5 会话 commits（v2_main，均未 push）

`4dc45db` §79 注册+launcher → `c6ef4fb` dump_orders 修复 → `24b2623` §79 回填+
plan-9 更新 → `8029eb3` P0 全套 → `5401971` P1 审计+manifest+S0 launcher+§81 注册 →
`6894230` PYTHON override → `0d7d5ae` §81 回填（S0 PASS）。
（§80 三连 `bac3244`/`f1b7659`/`714dd3a` 与 §78 addendum `5c8d84e` 来自并行会话，
同分支共存。）

---

## 2. FLAME pivot 后的状态判定（对照 addendum §6 P0）

1. **1B×3 正式长跑未启动** ✓（S0 即止，等确认）。
2. **旧 32R/top-4 代码保留为 structural ablation** ✓——`moe.py`/测试/账本全部
   保留，`G/X/S` 网格（capacity/granularity/shared）降级为论文消融。
3. **可复用资产**（addendum §0 明示保留）：manifest/token 顺序/epoch checkpoint
   契约、OE 审计 + trainable/frozen 对照、router load/route stability/dead experts
   诊断、单元测试、参数/FLOPs/吞吐/显存账本、forking 统计口径。S0 的
   `moe_diag.jsonl`/panel 格式直接沿用到 FLAME profile。
4. **需要新建**（P1–P3，见 §3 todo）。

## 3. Todo（按 addendum §6 顺序；P4 前有用户闸门）

- [ ] **D0（用户决策）语料**：addendum 通篇写 OpenWebText，但仓库现实 =
      full-163 语料（8192-vocab 固定 tokenizer，全部 OE 基础设施绑定；GPT-pro
      handoff 的「仓库现实修正 A」已论证保留）。manifest 已 immutable。若坚持
      真 OWT 需重建整条管线（新 tokenizer/词表/row 索引/freq index）——**默认
      保留 full-163，名字上的 OWT 按语料现实读**；此决策须用户拍板。
- [ ] **P1 FLAME profile**（`--flame_profile`）：
  - backbone 旗标：SwiGLU、RMSNorm(eps 1e-6)、RoPE、Linear 无 bias、
    input embedding 与 lm_head **不共享**、16 heads、dropout 0、init std 0.02；
    第 1 层 dense FFN、其余层 MoE（逐层 pattern）。
  - MoE 语义：64 routed / top-6 + always-on shared（宽度 **2×d_e** 单个宽 FFN，
    非两个 d_e 单元）；router fp32 pre-softmax + softmax score；aux 0.01、
    z 0.001。
  - `moeify()` 扩展：FLAME 形状不满足 `d_ff % G == 0`（Small 2736/352=7.77）——
    需把参考宽度从 d_ff 解耦为 active width = 8×d_e（或允许非整 granularity +
    显式 twin 宽度参数）。现有 (G,X,S) 代码路径保持为 ablation 配置不动。
  - 账本自动打印 active/total（命名规范 `flame_115mA_459mT`，禁只写 total）。
- [ ] **P1 测试扩展**：flame_profile 下 dense 等价 / twin 匹配（前向+参数量+
  FLOPs）单项测试，沿用 `test_moe_cpu.py` 模式（fp32 1e-6 / bf16 2e-2 双容差）。
- [ ] **P2 exact-active dense twin generator**：`d_ff_twin = 8×d_e`
  （Tiny 1408 / Small 2816 / Large 5632），除 FFN 类型外与 MoE 版逐层同构；
  official-shape（1368/2736/5472）作辅助对照，不混报。
- [ ] **P3 Small 尺度 S0**：四臂（Dense / Dense+OE / MoE / MoE+OE）× 100M ×
  1 pass × 1 seed；验收同 §8（NaN/账本/router 无塌缩/loss 同量级/诊断完整/
  吞吐显存含 dispatch 开销）。
- [ ] **P4（用户闸门，未经确认不得启动）**：Small 四臂 × 3 seeds = 12 run，
  1B unique × 3 passes，paired data order，epoch 1/2/3 checkpoints，
  预注册 difference-in-differences。
- [ ] **P5 家族扩展**：Small 核心现象成立后，Tiny/Large 四臂各 1 seed →
  补 3 seeds；语料 nested 前缀 0.33B/1B/2.5B（同一 manifest 前缀，2.5B < 7.76B
  池 ✓）。
- [ ] **OE 接入注意**：family 主线用 capacity-normalized（固定 P_OE/
  P_active_backbone）；Small 保留当前绝对表（2×2^20×768）作 reproduction arm；
  每 run 打印 OE params/rows/rows-per-vocab/collision/touched ratio
  （`oe_param_audit.py` 已有大部分，需接进 ledger）。
- [ ] **后续消融（降级队列）**：旧 G/X/S 网格（32R/top-4 = G4X8 主基线变消融点）、
  frozen OE、total-matched dense control（只做 Small 或 Large）。
- [ ] 仓库卫生：agents.md §3 的 plan 目录树仍写 plan-1..5（已过时，可补
  plan-6..9）；所有会话 commits 未 push（P5 红线）。

## 4. 环境与操作备忘（接手必读）

- **ophis 工作区** `/data1/guoshaoyang/ngram-gap-lab`；**python 必须用**
  `/data/home/guoshaoyang/ngram-gap-lab/.venv/bin/python`（torch 2.9.1+cu128 +
  pyarrow 23.0.1；系统 python3 = torch 2.6 不可用）。launcher 用 `PYTHON=` 覆盖。
- `data/tokenized` 是指向配额已死 home 的软链（**只读可用，不可写**）；新数据一律
  落 /data1 实目录；`TMPDIR=/data1/guoshaoyang/tmp`（否则 triton/gcc 报 quota）。
- 语料池 `/data2/shared/ncpl-pathA/harry_autoresearch_full_data/data`（163 parquet，
  1.8T free）；tokenizer `/data2/ncpl-pathA/work/vbird_autoresearch/cache/tokenizer`
  （sha256 已入 manifest）；上游默认 `/data` 是别人配额死链，用
  `FIXED_TOKENIZER_DIR` + `--lib_data_dir` 绕开。
- 数据落位：`data/tokenized_full163/`（163 bins）、`data/manifest_full163/`
  （manifest + 158 物化分片 + permutation.npy + 6542 val）、`data/rows/`
  （预计算行索引，shard_00001 起）。
- GPU：GPU0/5/6 常空闲，GPU7 有他人残留；跑前 `nvidia-smi` + `CUDA_VISIBLE_DEVICES`
  占卡，一 agent 一卡；md5 核对 `code/train.py`/`moe.py`/`ngram_freq.py` 后再开。
- **并发警告**：`train.py` 由并行会话共同维护（v2 dict-mode、LoRA dispatch 线都在
  改它）；任何编辑前先完整重读目标区域，commit 前重查 `git status`。

## 5. 关键口径（不随 pivot 变）

- gap = fixed-val − 同 step online train loss；`VAL_LOSS_INTERVAL_STEPS=10`；
  MoE 臂 train CE = 纯 LM loss（aux/z 分记）；`data/runs_fixed/` 是唯一权威数据。
- run_id 前缀 `p9_`；MoE 帐须带 `flame_{active}A_{total}T` 命名；S0/pilot 不进论文。
- matched 边界叫 **active-FFN-matched**（router/dispatch 开销另记），
  禁止写 wall-clock matched；负载熵跨 E 用 log(E) 归一化；ossification 用固定
  panel 的 route stability，不用 load 熵替代。
