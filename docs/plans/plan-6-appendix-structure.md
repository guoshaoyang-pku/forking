# Plan 6 · Appendix C–H 结构设计与图片分配

> **目的**：将飞书论文初稿 v2 中预留的「C–H. 完整实验与推导」落成 ICLR 惯例的具体小节，完成正文/appendix 图片分配，标注缺口。
> **输入**：飞书论文初稿 v2（只读）、`docs/figs/` 全量图片资产、`docs/report/experiment-registry.html`、`docs/experiment-lines.md`、`agents.md`。
> **约束**：不修改飞书文档；不 git push；正文图少而精，其余下沉 appendix。

---

## 1. Appendix 完整结构树

```
Appendix
├── A. 原始发现与 observable 筛选（已有，保持）
│   ├── A.1 原始任务与发现阶段的实验设置
│   └── A.2 通过跨实验 observables 定位 n-gram
│       ├── A.2.1 四组对照实验
│       ├── A.2.2 五条 observable 筛选规则
│       └── A.2.3 人工审查与组件移除验证
│
├── B. 标准设置与测量合同（已有，保持）
│
├── C. 三轴 Scaling 完整数据
│   ├── C.1 Table Size 轴
│   │   ├── C.1.1 Clean 单表 R 扫描（bigram / trigram 各 31 点）
│   │   ├── C.1.2 Gap-R 双对数斜率与分段拟合
│   │   └── C.1.3 Collision mass 与 dilution 补充图
│   ├── C.2 Dose（唯一数据量 × 重播次数）轴
│   │   ├── C.2.1 Shard 大小扫描 12 点完整表
│   │   ├── C.2.2 Fixed-step vs fixed-epoch 对照
│   │   └── C.2.3 Dose-frequency heatmap 补充
│   ├── C.3 Epoch 长度轴
│   │   ├── C.3.1 Epoch length scaling 轨迹族
│   │   ├── C.3.2 Staircase 台阶形状与 per-pass 增量
│   │   ├── C.3.3 Prefix-slice shard 事实与 epfix 注册
│   │   └── C.3.4 Long replay 20-epoch 线性增长
│   └── C.4 三轴交叉总结表
│
├── D. 优化器与 LR / β₂ 消融
│   ├── D.1 Table 优化器选择（RMSProp / AdamW / SGD）
│   │   ├── D.1.1 完整曲线（v5-refresh 11 run）
│   │   └── D.1.2 Final gap 汇总与 seed 方差（X1 三 seed）
│   ├── D.2 Table LR 扫描
│   │   ├── D.2.1 LR scale 8–1024 gap 响应
│   │   └── D.2.2 β₀.₉₉ gap-vs-table-LR 曲线
│   ├── D.3 β₂ 消融
│   │   ├── D.3.1 β₂ sweep 1k 步（0.9 / 0.95 / 0.98 / 0.99 / 0.999）
│   │   ├── D.3.2 β₂ × table LR 交互面（optv5f facets）
│   │   └── D.3.3 短 epoch β₂ 台阶清晰度（short_epoch_b2）
│   ├── D.4 Backbone LR 与绝对 table LR 解耦（M9）
│   │   ├── D.4.1 Epoch dynamics 面板
│   │   └── D.4.2 Recurrence diagnostics
│   └── D.5 Warmup 与 schedule 消融
│       ├── D.5.1 Warmup start grid
│       └── D.5.2 Warmup vs constant live 对比
│
├── E. 因果干预全曲线与 Mask 阈值扫描
│   ├── E.1 因果干预臂完整轨迹
│   │   ├── E.1.1 freeze-table / freeze-backbone / freeze-both / hash_reseed
│   │   ├── E.1.2 Frequency effect 分解
│   │   └── E.1.3 Freeze forking 时序
│   ├── E.2 Mask 阈值扫描
│   │   ├── E.2.1 mask_low（f ≤ t）阈值扫描完整曲线
│   │   ├── E.2.2 mask_high（f ≥ t）阈值扫描完整曲线
│   │   └── E.2.3 Mask freq-tradeoff 与 last-epoch 切片
│   ├── E.3 Net-val benefit 判决图
│   └── E.4 Replay 读出动力学（M8）
│       ├── E.4.1 10-epoch kernel dynamics
│       └── E.4.2 Freeze-backbone e1/e2/e3 gap 序列
│
├── F. 频率分桶与 Seen / Novel 分解细节
│   ├── F.1 Exact Frequency 分析
│   │   ├── F.1.1 Gap vs exact-f（log-x / log-log / steps 三视图）
│   │   ├── F.1.2 Bigram / trigram 频率分布
│   │   └── F.1.3 Hit-count 分布与 token mass
│   ├── F.2 Seen / Novel 分解
│   │   ├── F.2.1 f × pass 全账目（seen/novel × table-direct/backbone-carried）
│   │   ├── F.2.2 Novel continuation 占 gap 大头证据
│   │   └── F.2.3 Injection frequency tokens（raw / scaled 双视图）
│   ├── F.3 Good-Turing 缺失质量核
│   │   ├── F.3.1 g(f) 与 C·M(f) 对齐
│   │   └── F.3.2 Val damage vs missing mass M_f
│   └── F.4 Frequency bin 诊断方法说明
│
├── G. 多 Epoch 动力学与 Logit 锐化补充
│   ├── G.1 多 Epoch 台阶现象
│   │   ├── G.1.1 Within-epoch aligned 轨迹（main + toy 对照）
│   │   ├── G.1.2 Per-epoch gap 增量与 staircase 形状
│   │   └── G.1.3 Epoch multiseed 方差
│   ├── G.2 Logit 锐化理论补充
│   │   ├── G.2.1 2×2 锐化实验面板（已由 §53 logit-stats 素材覆盖；2×2 原设计降级为可选补强）
│   │   ├── G.2.2 Bias split 与 paired decomposition
│   │   └── G.2.3 Optimizer exponent 拟合
│   ├── G.3 状态空间分割补充
│   │   ├── G.3.1 Table cosine & margin 逐 pass 演化
│   │   ├── G.3.2 Hash collision mass 与 interference model vs data
│   │   └── G.3.3 Zipf context α 与 sampling regimes
│   └── G.4 Toy 模型对照
│       ├── G.4.1 Markov unigram gap epochs / excess vs freq / H
│       ├── G.4.2 Synth powerlaw gap vs freq / step
│       ├── G.4.3 L6 residual response
│       └── G.4.4 Toy β scan per-epoch / step-level
│
└── H. 测量管线与可复现性
    ├── H.1 Run ID 命名规范
    │   ├── H.1.1 前缀约定（nglab / optv5 / causalv5 / s1v5 / blrabs / ffqv5 / mixv5）
    │   ├── H.1.2 后缀约定（_fixed / _fd / _s43 / _s44）
    │   └── H.1.3 完整 run_id 索引表（指向 experiment-registry.html）
    ├── H.2 Fast-Diag 管线
    │   ├── H.2.1 diag_worker.py 异步 CPU 聚合契约
    │   ├── H.2.2 bf16 forward + eval block ~1.5s 基准
    │   └── H.2.3 fd vs old curves 一致性验证
    ├── H.3 代码版本与口径锁定
    │   ├── H.3.1 关键 commit hash（freq-bin fix / β₂ fix / clean-table rework）
    │   ├── H.3.2 md5sum 跨机核对清单
    │   └── H.3.3 Deprecated setting 标记规则
    ├── H.4 数据与 val set 合同
    │   ├── H.4.1 Train / val shard 不重叠保证
    │   ├── H.4.2 Fixed val batches 与 VAL_LOSS_INTERVAL_STEPS=10
    │   └── H.4.3 Frequency index 同步机制
    └── H.5 复现检查清单（checklist）
```

---

## 2. 图片分配表

### 2.1 正文图片（~12 张，精简）

| 正文节 | 图号 | 文件名 | 理由 | 新版本？ |
|--------|------|--------|------|----------|
| §1 Intro | 图 1 | `fig_v5_128x_injection_curves.png` | 主现象一览：四臂 gap 分叉 | 否 |
| §3.2 主现象 | 图 3-0 | `fig_v5_128x_injection_bars.png` | 末端 gap bar 对比 | 否 |
| §3.2 主现象 | 图 3-1 | `fig_v5_128x_injection_curves.png` | 完整 train/val/gap 曲线 | 否 |
| §3.3 幂律 | 图 4-3 | `fig_v5_128x_injection_frequency_bigram.png` | Gap vs exact-f 幂律 | 否 |
| §3.3 幂律 | 图 7-1 | `fig_v5_s1_table_size_loglog_clean.png` | Gap-R 双对数 | 否 |
| §3.3 幂律 | 图 7-2 | `fig_v5_s1_epoch_length_scaling.png` | Epoch 长度轴 | 否 |
| §4 干预 | 图 5-1 | `fig_v5_128x_causal_losses.png` | 因果干预臂轨迹 | 否 |
| §4 干预 | 图 4-5 | `fig_v5_128x_mask_low_le_scan.png` | Mask low 阈值 | 否 |
| §5 理论 | 图 4-7 | `fig_v5_good_turing_kernel.png` | Good-Turing 核 | 否 |
| §5 理论 | 图 5-3 | `fig_v5_epoch_kernel_dynamics.png` | 多 epoch 线性增长 | 否 |
| §6 验证 | 图 6-1 | `fig_v5_table_cosine_and_margin.png` | 表重写与 margin | 否 |
| §6 验证 | 图 6-2 | `fig_v5_f_by_pass_accounting.png` | f×pass 全账目 | 否 |

**正文合计：12 张**（Intro 1 + 现象 3 + 干预 2 + 理论 2 + 验证 2 + 幂律 2）

### 2.2 Appendix 图片分配

| Appendix 节 | 图片文件 | 来源目录 | 备注 |
|-------------|----------|----------|------|
| A.1 | （飞书已有嵌入） | — | 保持 |
| A.2 | （飞书已有嵌入） | — | 保持 |
| C.1.1 | `fig_v5_s1_table_size_loglog_clean.{png,svg}` | main | 正文用 png，appendix 附 svg |
| C.1.2 | `fig_v5_s1_table_load_collision.png`, `fig_v5_s1_table_load_proxy.png` | main | 下沉 |
| C.1.3 | `fig_v5_hash_collision_mass.{png,svg}`, `fig_v5_dilution_surface.png` | theory | 下沉 |
| C.2.1 | `fig_v5_dose_trajectories.png`, `fig_v5_dose_fixedstep.png` | main | 下沉 |
| C.2.2 | `fig_sweep_v2_family.{png,svg}`, `fig_sweep_v2_dose_resp.{png,svg}` | epoch_scale | 下沉 |
| C.2.3 | `fig_v5_dose_frequency_heatmap.png` | main | 下沉 |
| C.3.1 | `fig_v5_s1_epoch_length_trajectories.png`, `fig_v5_s1_epoch_length_valid.{png,svg}` | main | 下沉 |
| C.3.2 | `fig_v5_s1_epoch_staircase.png`, `fig_v5_s1_epoch_number.png` | main | 下沉 |
| C.3.3 | `fig_v5_s1_epoch_prefix.png` | main | 下沉 |
| C.3.4 | `fig_v5_s1_long_replay.png` | main | 下沉 |
| C.4 | （新建汇总表，无图） | — | — |
| D.1.1 | `fig_v5_optimizer_full_curves.png`, `fig_v5_optimizer_existing_curves.png` | main | 下沉 |
| D.1.2 | `fig_theory_x1_optimizer.{png,svg}` | theory | 下沉 |
| D.2.1 | `fig_v5_optv5f_all_scale_gap_facets.png`, `fig_v5_optv5f_final_gap.png` | main | 下沉 |
| D.2.2 | `fig_v5_beta099_gap_step1000_vs_table_lr.png` | main | 下沉 |
| D.3.1 | `fig_v5_beta2_sweep_1k.{png,svg}` | main | 下沉 |
| D.3.2 | `fig_v5_optv5f_beta2_compare.png`, `fig_v5_optv5f_conv_curves.png` | main | 下沉 |
| D.3.3 | `short_epoch_b2_gap_v11.{png,svg}` | short_epoch_b2 | 下沉 |
| D.4.1 | `fig_v5_backbone_lr_epoch_dynamics.{png,svg}` | theory | 下沉 |
| D.4.2 | `fig_v5_backbone_lr_recurrence_diagnostics.{png,svg}` | theory | 下沉 |
| D.5.1 | `fig_v5_warmup_start_grid.png` | main | 下沉 |
| D.5.2 | `fig_v5_warmup_vs_constant_live.png` | main | 下沉 |
| E.1.1 | `fig_v5_128x_causal_losses.png`（正文已用，appendix 放完整版含 no-gram） | main | 同图复用 |
| E.1.2 | `fig_v5_128x_causal_frequency_effect.png`, `fig_v5_causal_frequency_effect.png` | main | 下沉 |
| E.1.3 | `fig_v5_128x_freeze_forking.png` | main | 下沉 |
| E.2.1 | `fig_v5_128x_mask_low_le_scan.png`（正文已用） | main | 同图复用 |
| E.2.2 | `fig_v5_128x_mask_high_threshold_scan.png` | main | 下沉 |
| E.2.3 | `fig_v5_mask_freq_tradeoff.{png,svg}`, `fig_v5_mask_sweep_lastep.{png,svg}` | main | 下沉 |
| E.3 | `fig_v5_netval_benefit.{png,svg}` | main | 下沉 |
| E.4.1 | `fig_v5_epoch_kernel_dynamics.{png,svg}`（正文已用简化版） | theory | appendix 放完整 SVG |
| E.4.2 | `fig_v5_epoch_kernel_rescale.{png,svg}` | theory | 下沉 |
| F.1.1 | `fig_gap_vs_frequency.{svg}`, `fig_gap_vs_frequency_logx.{svg}`, `fig_gap_vs_frequency_loglog.{svg}`, `fig_gap_vs_frequency_steps.{svg}` | main | 三视图全部下沉 |
| F.1.2 | `fig_freq_bigram.svg`, `fig_freq_trigram.svg` | main | 下沉 |
| F.1.3 | `fig_hitcount_dist.{html,svg}`, `fig_v5_s1_frequency_token_mass.png` | main | 下沉 |
| F.2.1 | `fig_v5_f_by_pass_accounting.png`（正文已用） | theory | 同图复用 |
| F.2.2 | `fig_v5_val_damage_vs_missing_mass_bigram.png` | theory | 下沉 |
| F.2.3 | `fig_v5_128x_injection_frequency_tokens_raw.{png,svg}`, `fig_v5_128x_injection_frequency_tokens_scaled.{png,svg}` | main | 下沉 |
| F.3.1 | `fig_v5_good_turing_kernel.png`（正文已用） | theory | 同图复用 |
| F.3.2 | `fig_v5_missing_mass_kernel.png` | theory | 下沉 |
| F.4 | （文字说明，无图） | — | — |
| G.1.1 | `fig_main_within_epoch_aligned.{png,svg}`, `fig_toy_within_epoch_aligned.{png,svg}` | toy | 下沉 |
| G.1.2 | `figs_v11_staircase_shape_comparison.{png,svg}` | toy | 下沉 |
| G.1.3 | `fig_theory_epoch_multiseed.{png,svg}` | theory | 下沉 |
| G.2.1 | §53 logit sharpening 素材（严格 2×2 为可选补强） | `docs/figs/main/fig_ls20ep_sharpening_vs_ce.png` | 已完成素材 |
| G.2.2 | `fig_theory_bias_split.{png,svg}`, `fig_theory_paired_decomposition.{png,svg}` | theory | 下沉 |
| G.2.3 | `fig_theory_optimizer_exponent.{png,svg}` | theory | 下沉 |
| G.3.1 | `fig_v5_table_cosine_and_margin.png`（正文已用） | theory | 同图复用 |
| G.3.2 | `fig_v5_hash_collision_mass.{png,svg}`, `fig_v5_interference_model_vs_data.png` | theory | 下沉 |
| G.3.3 | `fig_v5_zipf_context_alpha.png`, `fig_theory_sampling_regimes.{png,svg}` | theory | 下沉 |
| G.4.1 | `fig_markov_unigram_gap_epochs.{png,svg}`, `fig_markov_unigram_excess_vs_freq.{png,svg}`, `fig_markov_unigram_h.{png,svg}` | theory | 下沉 |
| G.4.2 | `fig_toy_synth_gap_vs_freq.svg`, `fig_toy_synth_gap_vs_step.svg` | theory | 下沉 |
| G.4.3 | `fig_l6_residual_response.{png,svg}` | theory | 下沉 |
| G.4.4 | `figs_v11_toy_beta_scan_per_epoch.{png,svg}`, `figs_v11_toy_beta_scan_step_level.{png,svg}` | toy | 下沉 |
| H.* | （无图，纯文字/表格/代码片段） | — | — |

### 2.3 未分配 / 历史归档图

以下图保留在 `docs/figs/` 但不在正文或 appendix C–H 中引用，仅作历史溯源：

- `fig_cmp_*.png`（旧 probe 对比，被 v5-refresh 取代）
- `fig_online_b2_sweep_*.png`, `fig_online_lr_sweep_*.png`（旧 online 扫描，被 optv5f 取代）
- `fig_probe_artifact_*.png`（probe 伪影记录）
- `fig_uniform_*.png`（uniform 对照，历史）
- `fig_dtype_compare.png`, `fig_fp32_vs_bf16_samehp.png`（dtype 验证，已完成使命）
- `fig_ngram5_order5_*.{png,svg}`（order-5 探索，非主线）
- `fig_vanilla_repro_20260823.png`（vanilla 复现快照）
- `fig_lrdiag_*.png`（LR 诊断旧版，被 v5 取代）
- `fig_lrscan_clean_input*.png`（LR scan preview，被 D.2 取代）
- `epoch_scale/` 下旧 sweep 图（`fig_sweep_*.png/svg` 系列，被 v5 S1 取代）
- `table_opt/` 下旧 optimizer 图（被 optv5f / D 节取代）

---

## 3. 缺口清单（Top 3 + 次要）

### Top 3 最大缺口

| # | 缺口 | 所在小节 | 需要什么 | 优先级 | 状态 |
|---|------|----------|----------|--------|------|
| **1** | Logit 锐化 2×2 实验面板 | G.2.1 | §53 已提供 20-epoch sharpening-vs-CE 与 exemplar 素材；原 2×2 新实验降级为可选补强。 | P0 | ✅ 素材完成 |
| **2** | C.4 三轴交叉总结表 | C.4 | 已由 gen_appendix_c4_cross_table.py 从现有 CSV 生成覆盖表；无完整三维交叉，明确标注未测。 | P1 | ✅ 已生成 |
| **3** | H.1.3 完整 run_id 索引表 | H.1 | 已由 docs/tools/gen_run_id_index.py 生成静态导航表；完成性仍需按 fixed 目录验收。 | P1 | ✅ 已生成 |

### 次要缺口

| # | 缺口 | 所在小节 | 说明 |
|---|------|----------|------|
| 4 | D.3.2 β₂ × table LR 交互面高清版 | D.3.2 | `fig_v5_optv5f_beta2_compare.png` 分辨率偏低，建议重出 SVG |
| 5 | E.4.1 完整 10-epoch kernel SVG | E.4.1 | 正文用 PNG，appendix 应附 SVG 供审稿人放大 |
| 6 | F.4 Frequency bin 诊断方法伪代码 | F.4 | 需写一段伪代码说明 freq-bin 独立迭代器修复后的测量语义 |
| 7 | G.1.1 Main vs toy within-epoch 并排 | G.1.1 | 当前两张图独立，建议做一张 side-by-side 合成图 |
| 8 | DeepSeek-like 复现（§7.1 预留） | 正文 §7.1 | 完全空缺，但属于正文预留而非 appendix 任务 |
| 9 | 小数据 SFT 压力测试（§7.2 预留） | 正文 §7.2 | 完全空缺，同上 |

---

## 4. 执行建议

1. **Appendix 写作顺序**：H → C → D → E → F → G（H 是基础设施，先锁定口径；G 使用 §53 已完成素材，严格 2×2 设计仅作可选补强）。
2. **图片迁移**：正文确定 12 张后，其余图的引用路径统一改为 `appendix/C/figs/`、`appendix/D/figs/` 等子目录；物理文件可软链或复制，避免重复。
3. **缺口 #1 跟踪**：§53 已完成直接 logit-sharpening 测量；若需严格 sharp/flat × seen/novel 设计，另行登记新 run。
4. **缺口 #2 提取脚本**：`docs/plot_scripts/gen_appendix_c4_cross_table.py` 已生成覆盖表；新增交叉 run 前需重新登记。
5. **缺口 #3 生成脚本**：`docs/tools/gen_run_id_index.py` 已生成静态索引；每次 registry/log 更新后重跑。
6. **Commit 策略**：本计划文件先行 commit；后续每个 appendix 节写成独立 commit（前缀 `docs: appendix-X ...`）。

---

## 5. 元信息

- **创建时间**：2026-09-12
- **依据**：飞书论文初稿 v2（doc LQjudLhhpoThlQxShb8cwuGSnkb）、`docs/figs/` 278 张图片、`experiment-lines.md` v10+、`agents.md`
- **关联计划**：plan-5（S1 三轴）、plan-3（fix-and-backfill）
- **下次更新触发**：正文图号最终确定 / appendix 首节写完 / 严格 2×2 新实验获授权
