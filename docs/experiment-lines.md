# 实验线索引（Experiment Lines）

> 本文件回答一个问题：**「某个实验在哪、归哪条线、图在哪、结论可不可信」**。
> 数值细节看 `experiment-log.md`；断言状态看 `claims-ledger.md`；setting 定义看 `../agents.md` §1。

---

## ⚠️ 权威数据源

**`data/runs_fixed/` 里带 `_fixed` 后缀的 run 是唯一权威数据。**

历史的未修正版 run 与 `runs_fixed/` 里不带后缀的陈旧副本
**已于 2026-08-23 彻底删除**——按仓库原则，确认是 bug 污染的内容不归档、直接清除。
同时删除了由这些数据生成的全部图（`docs/_archive/figs_history/`，6 MB）。

### 两个 bug（代码已修复）

| bug | 症状 | 修复 |
|---|---|---|
| freq-bin 诊断复用训练迭代器 | 每次 freq eval 白吃 5 个 train batch，这些 batch 永不参与优化，且 epoch 计数虚高。**所有** launcher 都传了 `--freq_index`，故**全部历史 run 受影响** | `code/train.py` 已改用独立的 `freq_train_ds` 迭代器 |
| `table_betas[1]` 被静默覆盖 | `_table_rmsprop_step` 里 `b2` 错取 `self.ngram_beta2`，导致**所有显式传 `--table_betas` 的 rmsprop run 实际都跑 b2=0.999**，`--table_betas` 完全失效 | `code/train.py:537` 已改为 `b2 = self.table_betas[1]` |

详见 `notes/method/freq-bin-train-iter-bug.md`。
修复后的验证证据保留在 `data/runs_fixed/smoke_fixed_verify/`（400 步 smoke）。

### 修正幅度不是噪声

| run | pre-fix | `_fixed` | 变化 |
|---|---:|---:|---|
| `nglab1x_v10_input` | 1.931 | **1.867** | −3% |
| `nglab1x_v10_y` | 5.049 | 5.804 | +15% |
| `nglab1x_v10_v` | 5.041 | 5.450 | +8% |
| `nglab1x_v10_nogram` | 0.231 | 0.245 | +6% |
| `nglab0_25x_input_fv` | 12.99 | 9.69 | −25% |
| `nglab1x_opt_rmsprop_2x` | 2.376 | **4.674** | **+97%** |
| `nglab2x_opt_rmsprop_2x_b2_099` | 0.643 | **1.995** | **+210%** |
| `nglab1x_opt_sgd_09` | −0.002 | +0.073 | 符号翻转 |

⚠️ **历史 section 仍可能保留 pre-fix 数值**，与本表的 `_fixed` 对照表不能混读。
T1 目前仅完成远端 summary 回收与冲突审计；33 个重复 ID 均存在 host 间 endpoint 差异，未执行盲目覆盖。

## 主线（nanoGPT）

| # | 线名 | 科学问题 | run_id 前缀 | launcher | 作图脚本 | 图目录 | log § | 状态 |
|---|---|---|---|---|---|---|---|---|
| **M1** | 注入点消融 v50 | v/y/input 哪个产生 gap | `nglab_{v,y,input,nogram}` | `run_injpos.sh` | `gen_injpos_plot.py` | `figs/main/injpos_ablation.html` | §1–3 | 🗄️ 过时（val 每 50 步），被 M2 取代 |
| **M2** | **注入点消融 v10（博客主线）** | 同上，2000 步 + val 每 10 步 + 无表对照 | `nglab1x_v10_*_fixed` | `run_injpos_parallel.sh` | **`gen_all_figures.py`**（canonical）<br>`build_injpos_data_json.py`<br>`build_blog_clone_v10.py` | `figs/main/` | §8 | ✅ 完成 |
| **M3** | Table 优化器 · 1x epoch | RMSProp/AdamW/SGD 谁写得快；LR 剂量；β₂ | `nglab1x_opt_*_fixed` | `run_table_opt.sh` | `analyze_table_opt.py` | `figs/table_opt/` | §9 §9a §9b §9d | 🗄️ 历史结果受 β₂ bug 影响；当前证据由 V5-refresh/M3b 取代 |
| **M4** | Table 优化器 · 2x epoch | 同 M3，epoch 拉长看 β₂/LR 是否改变 | `nglab2x_opt_*_fixed` | `run_table_opt_2x.sh` | `analyze_table_opt_2x.py`<br>`analyze_table_opt_1x_vs_2x.py` | `figs/table_opt/` | §9c | 🗄️ 历史结果受 β₂ bug 影响；当前证据由 V5-refresh/M3b 取代 |
| **M3b** | 表学习率 × β₂ 消融深挖（附录） | β₂ 与表学习率的消融及交互；含 scale=8/16 高 table-LR β₂ gate | `optv5d_*`、`optv5e_*`、`optv5f_*` | `run_v5_clean.sh` + 专用高 scale 登记 | `docs/plot_scripts/plot_v5_optv5f_*.py` | `figs/main/` | §24c、§32 | ✅ 完成；scale=8/16 gate 与 scale=8–1024 收敛批均已回填 |
| **M5** | shard 大小扫描 | 「epoch shard 越大 gap 越小」是否成立 | `nglab{0_25x…8x}_input_fv*_fixed` | `run_shard_sweep{,_v2,_360}.sh` | `gen_shard_sweep_figs.py` | `figs/epoch_scale/` | §4 §6 §7 §10 | ✅ 完成（12 点齐） |
| **M6** | epoch 对齐批（历史 e5/e6） | 对齐 epoch 数后 M5 的单调关系是否消失 | `nglab*_e6_fixed` | `launch_360_*.sh` | `gen_epoch_aligned_figs.py`<br>`gen_nogram_vs_epochaligned_figs.py` | `figs/epoch_scale/` | §12 | 🗄️ 历史且不完整；由当前 v5 S1 三 epoch 阵列取代 |
| **M7** | 短 epoch × β₂ | β₂ 是否改变 per-epoch 台阶清晰度 | `nglab{025x,05x}_b2_099` | `run_epoch_short_b2.sh` | `gen_short_epoch_b2_figs.py` | `figs/short_epoch_b2/` | §11(B) | ⚠️ 完成但图未按 `_fixed` 重生成 |
| **V5-refresh** | 完整曲线证据刷新 | 把 M2、optimizer、causal、dose × frequency 统一到 current-batch / freq=10 口径 | `nglab1x_{input,y,v,nogram}_v5_freq10`；`optv5c_*`；`causalv5c_*`；`nglab*_input_v5_freq10` | `run_v5_optimizer_sweep.sh`；`V5_GROUP={inj_freq10,causal_refresh,dose_freq10} run_v5_main_manifest.sh` | `plot_v5_registry_figures.py` | `figs/main/` | §24b | ✅ 35/35 已验收：M2 4/4 + optimizer 11/11 + causal 6/6（另有 mask_high 阈值扫描 14/14）+ dose 11/11；只使用 `_fixed` 小型证据和 current-batch train loss |
| **X1** | optimizer × seed 复现 | v5 table optimizer 的三 seed final gap 与完整 step-10 曲线 | `optv5c_{rms,adamw,sgd}_*_s{43,44}`；seed 42 复用 | `run_v5_clean.sh` | `gen_theory_x1x2_verdict_figs.py` | `figs/theory/` | §27 | ✅ 完成；三臂三 seed 均值/SD 已回填 |
| **X2** | clean 表行宽扫描 | 同步改变 bigram/trigram row width `d`，观察 gap 是否向 no-gram 塌缩 | `ctbl_dim{192,48,12}_input_v5`；d=768 复用基线 | `run_v5_clean.sh` | `gen_theory_x1x2_verdict_figs.py` | `figs/theory/` | §28 | ✅ 完成；step-1000 final gap=0.803/0.364/0.157 |
| **M8** | replay 读出动力学 | gap 跨 pass 增长主要依赖继续写表，还是 backbone 逐轮学会读表 | `causalv5m3_*` | `V5_GROUP=causal_dynamics run_v5_main_manifest.sh` | `plot_v5_epoch_kernel_dynamics.py`（配套 10-epoch replay） | `figs/theory/` | §38 | ✅ 7/7 done @2022；freeze-backbone e1/e2/e3 gap=1.585/2.505/3.389，对照 5.733；freeze-table e2 保留 93.95% |
| **M9** | backbone LR × 固定绝对 table LR | 去掉旧扫描中 table LR 随 `--lr` 共变的混淆；检验动力学时间尺度与平衡位置 | `blrabs_{input,nogram}_*` | `run_v5_backbone_lr_abslock.sh`（跨 ophis/360-2 直接并行） | `plot_v5_backbone_lr_epoch_dynamics.py` | `figs/theory/fig_v5_backbone_lr_{epoch_dynamics,recurrence_diagnostics}.{png,svg}` | §42 | ✅ 14/14 done + QC；table LR 恒为 0.0768；同 dose 不折叠，pass 数是独立状态变量；一阶平台参数仅为窗口内外推 |
| **M10** | freeze 四因子长程 + pass 混匀 | 两状态模型硬预言：e2 全冻结后 gap 应停长；3 pass 全局混匀 vs 分块 replay 终点是否不变 | `ffqv5_freeze_{table,backbone,both}_e2_10ep`；`mixv5_tri_{replay,mixed}_3pass` | `GROUP={ffq,mix} run_v5_128x_rerun.sh`（360-2） | §45.1–§45.3 已回填；summary 位于远端 runs_fixed | `figs/theory/` | §45 | ✅ done；5/5 `_fixed` summaries；freeze_both final gap 3.0759（冻结后不变），mix 终点 2.4738 vs 2.2880 |

**历史 M6 的缺口已不再是当前待办**：§12 的「对齐 epoch 后单调关系消失」仍只由
8/12 个旧点支撑；缺失的大 shard 端（4x/5x/6x/8x）不补入当前证据，结论仅作
历史限定。当前 epoch 结论改由下方 v5 S1 三 epoch 阵列与 L4 长训承担。

## Toy / synthetic 线（已退役）

旧的 ngram5_freq_gap/ 受控数据 trainer 与 L1–L6 toy / synthetic 任务不再属于论文证据链，已于 2026-09-13 从当前工作树移除。历史实验日志、图和提交只用于 provenance，不再作为活动入口或新实验依赖。当前模型解释完全由 M2/V5、M8–M10、S1 以及 §48–§53 的自然语料证据承担。

## S1 三轴 scaling（T-scaling，极简 setting）

> 完整计划：`docs/plans/plan-5-s1-three-axis-handoff.md`；专题报告：
> `docs/appendices/s1_scaling_three_axis/report.md`；任务代码：
> `tasks/s1_scaling_three_axis/`。结果目录 `data/runs_scaling/`（新 namespace）。

| 轴 | 科学问题 | run_id 前缀 | launcher | 状态 |
|---|---|---|---|---|
| epoch · fixed-step | epoch 长度 L 是否影响 gap（相同算力 1000 步） | `ep_{L}_{arm}_fs[_s{43,44}]`（历史 L1-L4 × bigram/trigram/both/nogram） | `run_scaling_epoch_full.sh` | 🗄️ 历史 compile 波次；由当前 v5 S1 三 epoch 阵列取代 |
| epoch · fixed-epoch | 相同重播次数（6 epoch）下 gap 是否随 L 变化 | `ep_{L}_{arm}_fe[_s{43,44}]`（历史 L1-L4） | `run_scaling_epoch_full.sh` | 🗄️ 历史 compile 波次；由当前 v5 S1 三 epoch 阵列取代 |
| table size · historical | 1M 逻辑地址只向下，gap 由参数量还是 collision 决定 | `tbl_{TM}_{arm}[_s{43,44}]`（历史 dense/sparse） | `run_scaling_table_full.sh`（dense + sparse） | 🗄️ 历史 compile 波次；仅作 table-size 局部 slope 审计 |
| table size · clean | 物理行数 R、collision 与 gap 的 clean 单表关系 | `s1v5_128_tbl_bi1_R*`；`s1v5_128_tbl_tri1_R*` | `run_v5_s1_three_axis_queue.sh`；`plot_v5_registry_figures.py` | ✅ 当前 v5 62/62 done；两条单表轴各 31 点；净 gap 分窗口 slope：bigram 0.58、trigram 0.66；见注册表 `#registry-s1-table-clean` |
| frequency 轴 | `G(E,f)` 是否服从两因素模型（observational） | `s1v5_128_frequency_main` | `run_v5_s1_three_axis_queue.sh`；`plot_v5_registry_figures.py` | ✅ 当前 v5 main done；exact-f 与 Σf 图已登记 |
| backbone safety | 长训 no-ngram backbone 是否产生 gap | `bb_safety_L1_nogram_5000` | 手工（旧 cadence 50 步 + fp32） | ✅ done（旧口径；final +16.66 @5000，仅量级参考） |

### S1 v5 · 128× table-LR complete array（2026-08-27）

> 该批是快速现象筛查，不替代后续完整三轴证据。固定 v5 clean
> setting：`input`、bigram+trigram clean table、RMSProp `(0.0,0.99)`、
> backbone LR `6e-4`、`warmup_constant(100)`、bf16、无 compile、seed 42；
> table LR scale 固定为 **128×**（实际 `0.0768`）；train shard `1`，
> validation shards `2,3,4,5,6,7,8,9,10,6542`，结果写入
> `data/runs_scaling/`。

| 轴 | 科学问题 | run_id | 预算与统计 | 状态 |
|---|---|---|---|---|
| table size · bigram-only | 只开启 bigram，改变其 clean table 的 physical rows `R`；trigram 关闭 | `s1v5_128_tbl_bi1_R{10000..2347000}` 与小 R 扩展 | 31 个近似 log-spaced R；1000 steps；step 337/674/1000 已保留 | ✅ done；31/31；小 R 塌缩到 no-gram floor；净 gap 分窗口 R=2e3–2e5 的 log-log slope=0.576（R²=.997，n=12），raw-gap 敏感性 0.501 |
| table size · trigram-only | 只开启 trigram，改变其 clean table 的 physical rows `R`；bigram 关闭 | `s1v5_128_tbl_tri1_R{10000..2347000}` 与小 R 扩展 | 31 个近似 log-spaced R；1000 steps；step 337/674/1000 已保留 | ✅ done；31/31；小 R 塌缩到 no-gram floor；净 gap 分窗口 R=1e5–9.3e5 的 log-log slope=0.665（R²=.9997，n=8），raw-gap 敏感性 0.653 |
| frequency · main | 主实验完整双支路的 frequency-bin gap | `s1v5_128_frequency_main` | 1 个主实验；bigram+trigram；1000 steps；step 337/674/1000 的 frequency-bin/exact-frequency/table-RMS 已保留 | ✅ done；final gap=2.736 |
| epoch length · 3 epochs | 固定 3 个 epoch 时，epoch 长度是否改变 gap | `s1v5_128_ep_tri_{0p125..2p0}xL4_3ep` | 12 个 L4 倍数点；`epoch_batches=42…674`；仅 trigram；每点 3 epoch；step = `3×epoch_batches`，保留 e1/e2/e3 | ✅ done；12/12；≤1×L4 真长度段单调下降或持平（3.552→2.469）；>1×L4 旧点是 wrap-around replay/pass 数，旧 U 形读法作废 |
| epoch length · long | L4 下 gap 随 epoch 的长期变化 | `s1v5_128_ep_tri_1xL4_10ep[_nogram]` | trigram-only + no-gram；3370 steps；10 个 epoch boundary 记录 | ✅ done；trigram=8.675、nogram=0.480 @3370；epoch 增量逐渐变缓，尚未见明确平台 |

epoch 轴刻意不固定 1000 steps：每个长度都运行 3 个完整 epoch，以保持
replay 次数一致；L4 另跑 10 epoch 长训。table-size/frequency 轴为 1000-step
快速实验，并额外保留 337、674、1000 三个标准检查点，分别作为约 1、2、3
个 L4 epoch 的备用对齐点。所有 train loss 仍是
当前 batch 的 online loss，gap 为同一步 fixed validation − online train。

**口径（用户 2026-08-24 拍板）**：L4 = 337 batches/epoch（完整 shard 1，
24,264 chunks / 72）；L1/L2/L3 = 42/84/168 嵌套前缀。普通网格不跑
exact-frequency（不传 `--freq_index`），只算在线 train/val + fixed probe；
频率轴单独一小批 run（带 exact-freq）。原有 261 个 `_fixed` run 属于
`bf16 + torch.compile` 历史波次；最新标准是 **bf16 不 compile**，因此这些
run 只保留为历史数学审计，不能标记为当前标准完成。当前 no-compile S1
重跑已完成：table-size 单表 62/62、epoch trigram-only 14/14、frequency main 1/1。
历史 261 个 run 均通过当时的 contract / NaN / probe-hash QC；
其中 table 原始 21 个 run 为 dense
每 10 步监测，48 个 seed-42 + 72 个 seed-43/44 加密 run 为 sparse 只监测
最终 step。历史三 seed 的 H1–H4 探索性检验见附录报告 §7：ΔG 三 seed
全同号（方向 seed-stable）；trigram 在有限窗口内上升但未解析饱和，不能写成
无条件幂律；两因素模型只有有限区间的 β 相对可辨识，A/c/γ 不可辨识；模块
交互显著且 seed-sensitive，不允许合并单公式。

## 自然语言 5gram（order=5）· 历史线（已退役）

> 详细记录：`docs/experiment-log.md` §19。以下 run 是历史记录，不再作为当前活动入口；旧 trainer 已删除。

| # | 线名 | 科学问题 | run_id | 状态 |
|---|---|---|---|---|
| N1 | 自然语言 5gram · +trigram 注入 | 5gram 表的 coincidental gap 在真实语料上是否出现 | `ngram5_order5_trigram_fixed` | ✅ done @2000（seed 42 gap −0.0067；seed 43 −0.0090） |
| N2 | 自然语言 5gram · 纯 transformer 对照 | 无表时 backbone 自己能否学 5gram、gap 多大 | `ngram5_order5_puretransformer_fixed` | ✅ done @2000（gap +0.0054） |
| N3 | 自然语言 5gram · LR 消融 ×1 | 表 LR ×1（=backbone lr）对 gap 的影响 | `ngram5_order5_trigram_lr1x_fixed` | ✅ done @2000（gap +0.0015） |
| N4 | 自然语言 5gram · LR 消融 ×4 | 表 LR ×4 对 gap 的影响 | `ngram5_order5_trigram_lr4x_fixed` | ✅ done @2000（gap −0.0092） |

**历史初步结论（seed 42/43 的 +trigram 主臂）**：全局 gap 极小（主臂 −0.0067、
−0.0090；其余 seed 42 消融臂 −0.0092 到 +0.0054），与合成 markov 完全不同——43M
distinct 5gram contexts 挤 1M 行表，collision 可能稀释 gap。表仍有效降低 train loss
（seed 42 的 ×2/×4≈0.71 < ×1≈0.77 < 无表≈0.83；seed 43 主臂 train≈0.695）。
trigram 主臂的 per-bucket gap 在中频段出现较大值（[21,51)≈+1.00、
[501,1001)≈+1.82），但这些较高频桶的 token fraction 很小，且高频端只有少数频次类，
因此暂不把“中频峰”写成稳健定律；seed 43 的频次图已加入同一脚本，但 LR 消融仍只有
seed 42。

设置要点（历史 N1–N4，不能作为当前主线默认）：order=5（5-gram context）、train shard 1（49.7M tokens）/ val shards 2-10,6542 不重叠、
43M distinct contexts、input 注入、RMSProp 表 `(0.0, 0.99)`、table LR scale=2.0、AdamW lr 0.004、
batch 72×2048、2000 步、seed 42（并以完全相同口径复现 seed 43）、bf16 不 compile。
`code/make_ngram_blocks.py` + `code/train.py`。

---

## 已知待办

完整的可执行方案见 **`plans/plan-3-fix-and-backfill.md`**（面向 agent 阵列，含验收标准）。
摘要：

| # | 任务 | 优先级 | 状态 |
|---|---|---|---|
| T1 | 回填 `experiment-log.md`：pre-fix 数值**直接覆盖**为 `_fixed`，不保留旧值 | P0 | 远端 summary 已回收；字段级核对后覆盖 |
| T2 | 修正 β₂ 记录（标注无效 + 删无支撑结论，**不补跑实验**，用户已明确不重要） | P0 | 历史口径已标记；数字覆盖随 T1 |
| T3 | 用 `_fixed` 数据重生成全部图 | P0 | 索引与 fixture 已完成；数据图待 host-scoped run 证据 |
| T4 | 修 `experiment-log.md` 结构（§10 重复、§5 缺失、90 行逐字重复） | P1 | 章节编号与 TOC 已修；数字/表格复核仍开放 |
| T5 | 旧受控数据 trainer 的模型绑定审计 | P1 | 已退役；历史提交可追溯，当前不再执行 |
| T6 | 补 M6 缺口（4x/5x/6x/8x）或显式限定结论覆盖范围 | P1 | 历史缺口已显式限定；不再是当前 TODO |
| T7 | full-163 线：脚本已删除，数据坐标入库 `docs/notes/data/full-corpus-full163.md` | — | ✅ 已解决 |
| T8 | 长时程 no-ngram 对照（**缩小数据量前的保险**，不做缩小可缓跑） | P2 | 已有 `nglab1x_nogram_long_v5_fixed` 8000-step 候选；曲线/协议复核开放 |
| T9 | 固定 train 采样集合的 loss 曲线（测 ρ，与 T8 可并行） | P2 | 已有 `nglab{1,2}x_input_rho_v5_fixed` 候选；fixed-probe 语义与 host 冲突复核开放 |
| T10 | 缩小单 epoch 数据量以放大 gap（**唯一必须等 T8 的动作**） | P2 | S1 epoch-length 阵列已覆盖相邻问题；重新注册协议前不启动 |
| T11 | 工程整理：`RUNS_DIR` 环境变量化、抽 `table_opt_common.py`、包改名 | P3 | 随手做 |
| T12 | 干预机制已接线（CLI + 主循环 + `run_causal_minimal.sh`） | P1 | 代码就绪，待 GPU 实跑 |
