# 图片完整度清单 (2026-09-12)

> 盘点范围：`docs/figs/` 全部图片 + `docs/report/` 三份 HTML + 飞书初稿 v2 (LQjudLhhpoThlQxShb8cwuGSnkb)
> 标准依据：`agents.md` §1 极简 setting（128× table LR、v5 clean 表、warmup_constant(100)、fast-diag `_fd` 口径）
> 历史盘点状态定义：**current** = 当时人工判断可直接用 | **outdated** = 旧标准需重绘 | **broken** = 脚本/数据缺失 | **orphan** = 未被任何文档引用 | **missing** = 文档引用但文件不存在。当前机器生成索引使用 `referenced`/`outdated`/`orphan-review`；`referenced` 只表示引用关系，不等于科学验证。

---

## 统计摘要

| 指标 | 数量 |
|------|------|
| 图片总数（docs/figs/） | 290 |
| 唯一文件名 | 290 |
| 被 HTML 报告引用 | 52 |
| 被飞书初稿引用 | 14 |
| 被任一文档引用（去重） | ~63 |
| **current**（当前标准） | ~45 |
| **outdated**（旧标准） | ~35 |
| **broken**（脚本/数据缺失） | 1 |
| **orphan**（未被引用） | ~228 |
| **missing**（引用但不存在） | 0 |

### 子目录分布

| 子目录 | 图片数 | 主要内容 |
|--------|--------|----------|
| main/ | 94 | 主线 v5/128x 实验图 |
| theory/ | 68 | 理论模型与验证图 |
| epoch_scale/ | 45 | epoch/shard sweep 系列 |
| toy/ | 20 | toy model / within-epoch |
| table_opt/ | 18 | table optimizer ablation |
| short_epoch_b2/ | 5 | short epoch beta2 |
| 根目录散图 | 28 | 早期探索/probe/dtype |

---

## 飞书初稿 v2 已插图映射

| 章节 | 图注编号 | 文件名 | 脚本 | 源 run_id | Setting | 状态 | 备注 |
|------|----------|--------|------|-----------|---------|------|------|
| §3.2 | 图 1-1 | fig_paper1_forking_curves.png | plot_paper_fig1_forking.py | nglab1x_{input,y,v,nogram}_v5_128x_freq10_fixed | 128×, v5, warmup_const(100), seed 42, 2000 steps | **current** | 四臂主线曲线；左 train/val，右 raw gap |
| §3.3 | 图 4-3 | fig_v5_s1_frequency_exact_f.png | plot_v5_registry_figures.py | nglab1x_input_v5_freq10_r1 等 | v5 freq10 (非 _fd) | **outdated** | 非 fd 口径，需确认是否重绘 |
| §3.3 | 图 7-1 | fig_v5_s1_table_size_loglog_clean.png | plot_v5_128x_doc_figures.py | nglab1x_{arm}_v5_128x_freq10 | 128×, v5 | **current** | gap-R 双对数 |
| §3.3 | 图 3-4 | fig_s1_epoch20_epoch2_epoch3.png | plot_epoch20_scaling.py | s1v5_128_ep20_tri_*_3ep_v2_fixed | trigram-only, clean R=2²⁰, seed 42；20 长度 × 完整 3 epoch（每长度 1 run） | **current** | Figure 5 replacement；epoch 2/3 精确边界 |
| §3.3 | 图 3-8 | fig_v5_s1_epoch_staircase.png | plot_v5_epoch_staircase.py | nglab1x_input_v5_128x_freq10_fd + s1v5_128_ep*_20ep | 128×, _fd, 20ep | **current** | 多 epoch 台阶 |
| §5.1 | 图 5-1 | fig_v5_causal_losses.png | plot_v5_128x_doc_figures.py | causalv5c_*_128x 系列 | 128×, causal | **current** | 因果干预臂轨迹 |
| §4 | 图 4-5 | fig_v5_128x_mask_low_le_scan.png | plot_v5_128x_doc_figures.py | causalv5m2_mask_low_* | 128×, mask | **current** | mask_low 阈值扫描 |
| §4 | 图 4-6 | fig_v5_128x_mask_high_threshold_scan.png | plot_v5_mask_high_threshold_scan.py | causalv5m2_mask_high_t*_e1_fixed | 128×, F>=t 刷新，14 个阈值 | **current** | mask_high 阈值扫描 |
| §5 | 图 5-3 | fig_v5_s1_long_replay.png | plot_v5_registry_figures.py | s1v5_128_ep*_20ep 系列 | 128×, 20ep | **current** | 20-epoch 长程线性 |
| §5.1 | 图 4-7 | fig_v5_good_turing_kernel.png | plot_v5_good_turing_kernel.py | 离线计算 | N/A (理论) | **current** | Good-Turing 核 |
| §5.1 | 图 4-8 | fig_v5_val_damage_vs_missing_mass_bigram.png | analyze_v5_val_damage_vs_missing_mass.py | nglab1x_{input,nogram}_v5_128x_freq10_fd | 128×, _fd | **current** | val 伤害 vs 缺失质量 |
| §6 | 图 6-2 | fig_v5_f_by_pass_accounting.png | analyze_v5_f_by_pass_table.py | 多 run 汇总 | 128× | **current** | f x pass 全账目 |
| §6 | 图 6-1 | fig_v5_table_cosine_and_margin.png | analyze_v5_cosine_margin_posthoc.py | nglab1x_input_v5_128x_freq10_fd | 128×, _fd | **current** | 表 cosine + margin |
| §9 | 图 9-1 | fig_v5_netval_benefit.png | plot_v5_netval_benefit.py | 多变体汇总 | 128× | **current** | 净收益判决 |
| §3.1 | 图 3.1 | main/fig_transformer_ngram_injection_routes.svg | draw_transformer_ngram_injection.py | 无 run（结构图） | vanilla nanoGPT；三种 route 备选，每次 run 只启用一种 | **current** | 架构主图；hash table 逻辑留在附录伪代码 |

### 飞书图注编号问题

已知问题：飞书 §3.3 中三张图的图注编号仍为旧编号（图4-3 / 图7-1 / 图7-2），与章节位置不一致。建议统一为 §3.3 内部编号（如图3-3/3-4/3-5）或按全文顺序重排。

---

## HTML 报告引用图清单

### index.html (权威主汇报)

| 文件路径 | 主题 | 脚本 | 状态 |
|----------|------|------|------|
| main/fig_v5_128x_dose_gap.png | dose final gap | plot_v5_128x_batch.py | current |
| main/fig_v5_128x_injection_bars.png | 四臂 final gap bars | plot_v5_128x_batch.py | current |
| main/fig_v5_128x_injection_frequency_tokens_scaled.svg | frequency tokens scaled | plot_v5_128x_doc_figures.py | current |
| main/fig_v5_128x_mask_high_threshold_scan.png | mask_high scan | plot_v5_mask_high_threshold_scan.py | current |
| main/fig_v5_128x_mask_low_le_scan.png | mask_low scan | plot_v5_128x_doc_figures.py | current |
| main/fig_v5_128x_optimizer_gap.png | optimizer gap bars | plot_v5_128x_batch.py | current |
| main/fig_v5_beta099_gap_step1000_vs_table_lr.png | β0.99 gap vs table LR | plot_v5_beta099_gap_step1000.py | current |
| main/fig_v5_beta2_sweep_1k.svg | β2 sweep | plot_v5_fig17_beta2_sweep.py | current |
| main/fig_v5_blrv5_backbone_lr.png | backbone LR scan | plot_v5_blrv5_backbone_lr.py | current |
| main/fig_v5_causal_losses.png | causal arm losses | plot_v5_128x_doc_figures.py | current |
| main/fig_v5_optv5f_all_scale_gap_facets.png | optv5f scale facets | plot_v5_optv5f_readable.py | current |
| main/fig_s1_epoch20_epoch2_epoch3.svg | epoch-length scaling；exact epoch 1–3 endpoints | plot_epoch20_scaling.py | current |
| main/fig_v5_s1_epoch_number.png | epoch number | plot_v5_128x_doc_figures.py | current |
| main/fig_v5_s1_table_size_loglog_clean.svg | table size loglog | plot_v5_128x_doc_figures.py | current |
| theory/fig_v5_dilution_surface.png | dilution surface | plot_v5_dilution_surface.py | current |
| theory/fig_v5_epoch_kernel_dynamics.png | epoch kernel dynamics | plot_v5_epoch_kernel_dynamics.py | current |
| theory/fig_v5_good_turing_kernel.png | Good-Turing kernel | plot_v5_good_turing_kernel.py | current |

### experiment-registry.html

引用 37 张图，大部分与 index.html 重叠。额外引用包括：
- `appendices/s1_scaling_three_axis/figs/backbone_safety_L1_nogram_long.png` (current)
- `fig_ngram5_order5_gap*.svg` (ngram5 toy, current)
- `theory/fig_gap_vs_samples_*.svg` (理论采样, current)
- `theory/fig_markov_unigram_gap_epochs.svg` (Markov 理论, current)
- `toy/figs_v11_toy_beta_scan_per_epoch.svg` (toy beta scan, current)

### clean-data-report.html

引用 7 张图，均为 epoch_scale/ 和 main/ 下的 sweep 系列：
- `epoch_scale/fig_sweep_v2_*.svg` (4张, v2 sweep, current)
- `main/fig_freq_bigram.svg`, `fig_freq_trigram.svg`, `fig_gap_vs_frequency.svg` (频率分布, current)

---

## Outdated 图片（旧标准，需重绘）

以下图片使用旧 setting（2× table LR / β₂=0.999 / warmdown / 非 fd 口径），不符合 agents.md §1 当前标准：

| 文件 | 旧标准类型 | 脚本 | 建议 |
|------|-----------|------|------|
| main/fig_v5_2x_128x_injection_curves.png | 2× 历史对比 | plot_v5_128x_doc_figures.py | 保留作历史对照，标注 [DEPRECATED SETTING] |
| table_opt/fig_table_opt_2x.* | 2× table LR | analyze_table_opt_2x.py | outdated，除非专门讨论 2x |
| table_opt/fig_table_opt_1x_vs_2x.* | 1x vs 2x 对比 | analyze_table_opt_1x_vs_2x.py | outdated |
| table_opt/fig_beta2_ablation.* | β₂ ablation (含 0.999) | fig_beta2_lr_ablation.py | 部分 outdated |
| table_opt/fig_beta2_curves.* | β₂ curves | fig_beta2_lr_curves.py | 部分 outdated |
| short_epoch_b2/short_epoch_b2_gap*.png | β₂=0.999 short epoch | gen_short_epoch_b2_figs.py | outdated |
| main/fig_v5_causal_existing_losses.png | 旧 v5 (非 128x) | plot_v5_registry_figures.py | outdated |
| main/fig_v5_dose_trajectories_existing.png | 旧 dose | plot_v5_registry_figures.py | outdated |
| main/fig_v5_causal_frequency_effect.png | 可能非 fd | plot_v5_128x_doc_figures.py | 需确认 |
| epoch_scale/ 系列 (部分) | 旧 shard sweep | gen_shard_sweep_figs.py | 需逐张确认 |
| 根目录 fig_cmp_*.png (12张) | probe/b2 旧实验 | gen_all_figures.py | outdated (probe artifact) |
| 根目录 fig_online_*.png (3张) | 旧 online b2 sweep | gen_all_figures.py | outdated |
| 根目录 fig_uniform_*.png (2张) | 旧 uniform 实验 | gen_all_figures.py | outdated |
| 根目录 fig_probe_artifact_*.png (2张) | probe artifact | gen_all_figures.py | outdated |
| main/fig_lrdiag_*.png (7张) | fp32 diag / warmdown | plot_lrdiag_*.py | outdated (非 fd 口径) |
| main/fig_lrscan_clean_input*.png | LR scan (可能旧) | plot_lrscan_clean_input.py | 需确认 |

**小计：~35 张 outdated**

---

## Broken 图片

| 文件 | 问题 |
|------|------|
| （无） | 所有被引用的图片文件均存在 |

注：`fig_v5_netval_benefit.png` 的脚本 `plot_v5_netval_benefit.py` 存在且可生成，标记为 current。

---

## Orphan 图片（未被任何文档引用）

共 ~216 张图片未被 index.html / experiment-registry.html / clean-data-report.html / 飞书初稿 引用（2026-09-12 人工快照；当前生成索引请以 `docs/figure-index.json` 为准）。

主要类别：
1. **epoch_scale/ 大量 sweep 中间产物** (~30张)：fig_sweep_*, gap_vs_epoch_*, shard_curve_* 等
2. **theory/ 理论推导中间图** (~40张)：fig_theory_*, fig_v5_epoch_*, fig_markov_* 等
3. **main/ 历史版本/变体** (~20张)：fig_v5_s1_*, fig_v5_injection_* 等非最终版
4. **toy/ 探索图** (~10张)：fig_main_within_epoch_*, fig_zipf_* 等
5. **根目录早期探索** (~28张)：fig_cmp_*, fig_dtype_*, fig_fp32_* 等

> 注：orphan 不等于无用。许多是实验过程中的诊断图或备选方案。建议用户 review 后决定归档或删除。

---

## Top 待办建议（需用户决策）

### 1. 飞书 §3.3 图注重编号
- **问题**：图4-3 / 图7-1 / 图7-2 编号与 §3.3 位置不一致
- **建议**：统一为 §3 内部编号（图3-3/3-4/3-5）或全文顺序重排
- **影响**：仅飞书文档编辑，不涉及重绘

### 2. fig_v5_s1_frequency_exact_f.png 口径确认
- **问题**：脚本使用 `nglab1x_input_v5_freq10_r1`（非 `_fd` 口径）
- **建议**：确认是否需要用 `_fd` 数据重绘；若当前数据已足够则标 current
- **影响**：可能需要 1 次重绘

### 3. Outdated 图片处置策略
- **问题**：~35 张旧标准图片仍存在于 figs/
- **选项**：
  - A) 移至 `_archive/figs/` 并标注
  - B) 保留但在文件名加 `_deprecated` 后缀
  - C) 删除（需确认无其他引用）
- **建议**：选 A，保持 figs/ 清洁

### 4. Orphan 图片清理
- **问题**：216 张 orphan 占总量 78%
- **建议**：分批 review
  - 第一批：根目录散图 (28张) — 多为早期探索，大概率可归档
  - 第二批：epoch_scale/ 中间产物 (30张) — 确认是否有未发表的引用
  - 第三批：theory/ 中间图 (40张) — 确认理论章节是否需要补充

### 5. SVG/PNG 双格式冗余
- **问题**：多处同时存在 .png 和 .svg 版本（如 fig_v5_beta2_sweep_1k）
- **建议**：确定单一输出格式标准
  - Blog/HTML 优先 SVG（矢量、体积小）
  - 飞书/演示优先 PNG（兼容性好）
- **影响**：减少 ~30% 冗余文件

---

## 完整图片索引（按状态分类）

### Current (~45 张)

| 文件路径 | 主题 | 源 run_id | 被谁引用 |
|----------|------|-----------|----------|
| main/fig_v5_128x_injection_curves.png | 四臂主线曲线 | nglab1x_*_v5_128x_freq10 | Lark§3.2, index |
| main/fig_v5_128x_injection_bars.png | 四臂 final gap | nglab1x_*_v5_128x_freq10 | Lark, index, registry |
| main/fig_v5_128x_dose_gap.png | dose final gap | nglab1x_*_v5_128x_freq10 | index, registry |
| main/fig_v5_128x_causal_losses.png | 因果干预轨迹 | causalv5c_*_128x | Lark§5.1, index, registry |
| main/fig_v5_128x_causal_gap.png | 因果 final gap | causalv5c_*_128x | registry |
| main/fig_v5_128x_causal_frequency_effect.png | mask 频率效应 | causalv5c_*_128x | registry |
| main/fig_v5_128x_mask_low_le_scan.png | mask_low 扫描 | causalv5m2_mask_low_* | Lark§4, index |
| main/fig_v5_128x_mask_high_threshold_scan.png | mask_high 扫描 | causalv5m2_mask_high_t*_e1 | Lark§4, index, registry |
| main/fig_v5_128x_optimizer_gap.png | optimizer gap | optv5f_* | Lark, index, registry |
| main/fig_v5_128x_rowwidth_gap.png | row width gap | X2 系列 | registry |
| main/fig_v5_128x_injection_frequency_bigram.png | bigram 频率 bin | nglab1x_*_v5_128x_freq10 | registry |
| main/fig_v5_128x_injection_frequency_trigram.png | trigram 频率 bin | nglab1x_*_v5_128x_freq10 | registry |
| main/fig_v5_128x_injection_frequency_tokens_scaled.svg | tokens scaled | nglab1x_*_v5_128x_freq10 | index |
| main/fig_v5_s1_table_size_loglog_clean.png | gap-R 双对数 | nglab1x_*_v5_128x_freq10 | Lark§3.3, registry |
| main/fig_v5_s1_table_size_loglog_clean.svg | 同上 SVG | 同上 | index, registry |
| main/fig_v5_s1_table_size.png | table size | nglab1x_*_v5_128x | registry |
| main/fig_s1_epoch20_epoch2_epoch3.png | epoch 长度 scaling；20 lengths | s1v5_128_ep20_tri_*_3ep_v2_fixed | Lark§3.3 Figure 5, index, registry |
| main/fig_s1_epoch20_epoch2_epoch3.csv | epoch-length source table | plot_epoch20_scaling.py | registry provenance |
| main/fig_v5_s1_epoch_length_trajectories.png | epoch 轨迹 | s1v5_128_ep* | registry |
| main/fig_v5_s1_epoch_length_valid.svg | epoch length valid（旧图3-4 候选） | s1v5_128_ep* | superseded（图3-4 已改用 epoch20 资产） |
| main/fig_v5_s1_epoch_number.png | epoch number | s1v5_128_ep* | Lark, index, registry |
| main/fig_v5_s1_long_replay.png | 20-epoch 长程 | s1v5_128_ep*_20ep | Lark§5, registry |
| main/fig_v5_s1_epoch_staircase.png | 多 epoch 台阶 | nglab1x_*_fd + s1v5_128_ep*_20ep | Lark§3.3 |
| main/fig_v5_s1_frequency_exact_f.png | exact freq f | nglab1x_*_v5_freq10 | Lark§3.3, registry |
| main/fig_v5_s1_frequency_token_mass.png | token mass | nglab1x_*_v5_128x | registry |
| main/fig_v5_s1_table_load_proxy.png | table load proxy | s1v5_128_tbl* | registry |
| main/fig_v5_beta099_gap_step1000_vs_table_lr.png | β0.99 gap vs LR | optv5c/f_rms_b099_s* | index, registry |
| main/fig_v5_beta2_sweep_1k.svg | β2 sweep | optv5f_rms_b099_s* | index |
| main/fig_v5_blrv5_backbone_lr.png | backbone LR scan | blrv5_*_lr* | index |
| main/fig_v5_optv5f_readable_overview.png | optv5f overview | optv5f_rms_b099_s* | registry |
| main/fig_v5_optv5f_readable_2000_facets.png | optv5f 2000 facets | optv5f_*_2k | registry |
| main/fig_v5_optv5f_all_scale_gap_facets.png | all scale facets | optv5f_* | index |
| main/fig_v5_optimizer_frequency.png | optimizer frequency | optv5f_* | registry |
| main/fig_v5_optimizer_full_curves.png | optimizer full curves | optv5f_* | registry |
| main/fig_v5_dose_frequency_heatmap.png | dose frequency heatmap | nglab1x_*_v5_128x | registry |
| main/fig_v5_netval_benefit.png | 净收益判决 | 多变体汇总 | Lark§9 |
| main/fig_freq_bigram.svg | bigram 频率分布 | freq_index | clean-data-report |
| main/fig_freq_trigram.svg | trigram 频率分布 | freq_index | clean-data-report |
| main/fig_gap_vs_frequency.svg | gap vs frequency | nglab1x_*_v5_128x | clean-data-report |
| theory/fig_v5_good_turing_kernel.png | Good-Turing 核 | 离线计算 | Lark§5.1, index |
| theory/fig_v5_val_damage_vs_missing_mass_bigram.png | val 伤害 vs M_f | nglab1x_*_v5_128x_freq10_fd | Lark§5.1 |
| theory/fig_v5_f_by_pass_accounting.png | f x pass 账目 | 多 run 汇总 | Lark§6 |
| theory/fig_v5_table_cosine_and_margin.png | cosine + margin | nglab1x_input_v5_128x_freq10_fd | Lark§6 |
| theory/fig_v5_dilution_surface.png | dilution surface | s1v5_128_tbl_bi1_R* | index |
| theory/fig_v5_epoch_kernel_dynamics.png | epoch kernel | s1v5_128_ep*_20ep | index |
| theory/fig_gap_vs_samples_bc11.svg | BC11 采样 | 理论 | registry |
| theory/fig_gap_vs_samples_exact.svg | exact 采样 | 理论 | registry |
| theory/fig_gap_vs_samples_longtail.svg | longtail 采样 | 理论 | registry |
| theory/fig_markov_unigram_gap_epochs.svg | Markov unigram | 理论 | registry |
| toy/figs_v11_toy_beta_scan_per_epoch.svg | toy beta scan | toy | registry |
| fig_ngram5_order5_gap.svg | ngram5 order5 | ngram5_order5_* | registry |
| fig_ngram5_order5_gap_freq.svg | ngram5 freq | ngram5_order5_* | registry |
| appendices/s1_scaling_three_axis/figs/backbone_safety_L1_nogram_long.png | backbone safety | L1 nogram | registry |
| epoch_scale/fig_sweep_v2_family.svg | v2 sweep family | shard sweep v2 | clean-data-report |
| epoch_scale/fig_sweep_v2_meta.svg | v2 sweep meta | shard sweep v2 | clean-data-report |
| epoch_scale/fig_sweep_v2_dose_resp.svg | v2 dose response | shard sweep v2 | clean-data-report |
| epoch_scale/fig_sweep_v2_injpos.svg | v2 injpos | shard sweep v2 | clean-data-report |

### Outdated (~35 张)

见上方 "Outdated 图片" 章节详细列表。

### Orphan (~216 张)

完整列表见 `/tmp/orphan_figs.txt`。主要分布在：
- epoch_scale/ (~30张 sweep 中间产物)
- theory/ (~40张理论推导图)
- main/ (~20张历史版本)
- toy/ (~10张探索图)
- 根目录 (~28张早期实验)

---

## 附录：脚本-图片映射速查

| 脚本 | 输出图片 | 状态 |
|------|----------|------|
| plot_v5_128x_batch.py | fig_v5_128x_injection_curves/bars/dose_gap/causal_gap/rowwidth_gap/optimizer_gap | current |
| plot_v5_128x_doc_figures.py | fig_v5_128x_injection_frequency_*, causal_losses/frequency_effect, mask_low_le_scan, s1_table_size_loglog_clean, s1_epoch_number | current |
| plot_v5_registry_figures.py | fig_v5_injection*, frequency_*, s1_frequency_exact_f, s1_epoch_length_scaling, s1_long_replay, causal_losses (旧版) | mixed |
| plot_v5_epoch_staircase.py | fig_v5_s1_epoch_staircase | current |
| plot_v5_good_turing_kernel.py | fig_v5_good_turing_kernel | current |
| analyze_v5_val_damage_vs_missing_mass.py | fig_v5_val_damage_vs_missing_mass_bigram | current |
| analyze_v5_f_by_pass_table.py | fig_v5_f_by_pass_accounting | current |
| analyze_v5_cosine_margin_posthoc.py | fig_v5_table_cosine_and_margin | current |
| plot_v5_mask_high_threshold_scan.py | fig_v5_128x_mask_high_threshold_scan | current |
| plot_v5_blrv5_backbone_lr.py | fig_v5_blrv5_backbone_lr | current |
| plot_v5_beta099_gap_step1000.py | fig_v5_beta099_gap_step1000_vs_table_lr | current |
| plot_v5_optv5f_readable.py | fig_v5_optv5f_readable_*, all_scale_gap_facets | current |
| plot_v5_dilution_surface.py | fig_v5_dilution_surface | current |
| plot_v5_epoch_kernel_dynamics.py | fig_v5_epoch_kernel_dynamics | current |
| plot_v5_netval_benefit.py | fig_v5_netval_benefit | current |
| plot_v5_epoch_length_valid.py | fig_v5_s1_epoch_length_valid | superseded |
| draw_transformer_ngram_injection.py | main/fig_transformer_ngram_injection_routes.svg | current；结构图，无实验数值 |
| plot_v5_fig17_beta2_sweep.py | fig_v5_beta2_sweep_1k | current |
| gen_shard_sweep_figs.py | epoch_scale/fig_sweep_* | mixed |
| gen_all_figures.py | 根目录 fig_cmp_*, fig_uniform_*, fig_probe_* | outdated |
| plot_lrdiag_*.py | main/fig_lrdiag_* | outdated (非 fd) |
| fig_beta2_lr_ablation.py | table_opt/fig_beta2_ablation, fig_table_lr_ablation | outdated |
| analyze_table_opt_2x.py | table_opt/fig_table_opt_2x | outdated |

---

*Generated: 2026-09-12 by figure-inventory agent*
*Repo: ngram-gap-lab @ main*
