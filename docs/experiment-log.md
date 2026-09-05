# ngram-gap-lab · 实验日志

> 创建：2026-08-05
> **本文件是多 Agent 的唯一实验登记簿**：先登记（`planned`）→ 占 GPU 开跑 → 回填（`done`）。
> 工作原则、run_id 约定与并行规则见 `agents.md` §0；**极简 setting（SSOT）见 `agents.md` §1**。

## 实验登记总表

| run_id | 日期 | 实验 | 状态 | gap 关键值 | 详情 |
|---|---|---|---|---|---|
| `blrabs_{input,nogram}_lr*_tlr0p0768_*` | 2026-08-31 | **纯 backbone LR 扫描：绝对 table LR 锁定 0.0768（14 run）** | 🔄 running（ophis GPU2/4/5） | 待测：短程高 LR 判混淆；长程拟合时间常数/准平衡 | §42 |
| `blrv5_{input,nogram}_lr{0p0001..0p0040}` | 2026-08-30 | **耦合 LR 扫描（原误标 backbone-only；12 run）** | ⚠️ done but confounded | scale=128 导致绝对 table LR 0.0128→0.512；不得作纯 backbone 因果结论 | §39、§42 |
| `s1v5_128_epfx_tri_{2p0,3p0,4p0}xL4_3ep` | 2026-08-31 | **V5 epoch 长度轴修复批（多 shard 真实池，trigram 单支路）** | ✅ done | 1.955/1.519/1.267；U 形确认为 wrap 假象 | §40 |
| `s1v5_128_ep_tri_1xL4_20ep` / `s1v5_128_ep1xL4_20ep_{both,nogram}` | 2026-08-31 | **V5 20-epoch 长 replay 批（shard-1 池 ×20 遍，tri-only / both / nogram 三臂）** | 🔄 running | 寻长程渐进行为（10ep 递推外推平台此前未验证） | §41 |
| `causalv5m3_hash_reseed_e2{,e3}` | 2026-08-31 | **V5 二次 reseed 判决批（2022 步 6 pass：e2 单次 vs e2+e3 双次 reseed）** | 🔄 running | 检验重对齐是否可再学、二次 reseed 是否再击落 gap | §41 |
| `optv5h_rms_b099_s2p0_warmstart0p1` | 2026-08-26 | V5 warmup 起始倍率 · 0.1× | ✅ done | +1.504498 @1000 | §31 |
| `optv5h_rms_b099_s2p0_warmstart0p5` | 2026-08-26 | V5 warmup 起始倍率 · 0.5× | ⚠️ failed at initialization（360-1 GPU7 CUDA launch failure） | 无结果 | §31 |
| `optv5h_rms_b099_s2p0_warmstart0p5_r1` | 2026-08-26 | V5 warmup 起始倍率 · 0.5× retry | ⚠️ failed at initialization（GPU7 large-allocation launch failure） | 无结果 | §31 |
| `optv5h_rms_b099_s2p0_warmstart0p5_r2` | 2026-08-26 | V5 warmup 起始倍率 · 0.5× retry on GPU5 | ✅ done | +1.441762 @1000 | §31 |
| `optv5g_rms_b099_s2p0_constant` | 2026-08-26 | V5 schedule 消融 · zero-warmup constant | ✅ done | +0.533989 @1000 | §30 |
| `optv5f_rms_b099_s8p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 8 × β₂.99 · 2000 步** | ✅ done | 1.055 / 7.076 / +6.020 @2000 | §32 |
| `optv5f_rms_b0999_s8p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 8 × β₂.999 · 2000 步** | ✅ done | 1.156 / 7.056 / +5.899 @2000 | §32 |
| `optv5f_rms_b099_s16p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 16 × β₂.99 · 2000 步** | ✅ done | 1.274 / 7.035 / +5.761 @2000 | §32 |
| `optv5f_rms_b0999_s16p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 16 × β₂.999 · 2000 步** | ✅ done | 1.376 / 6.951 / +5.575 @2000 | §32 |
| `optv5f_rms_b099_s32p0` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 32 × β₂.99 · 1000 步** | ✅ done | +2.647 @1000 | §32 |
| `optv5f_rms_b099_s32p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 32 × β₂.99 · 2000 步** | ✅ done | 1.263 / 6.921 / +5.659 @2000 | §32 |
| `optv5f_rms_b0999_s32p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 32 × β₂.999 · 2000 步** | ⚠️ failed @1830（CUDA peer-GPU/hardware error） | 无 final；partial gap +5.096 @1830 | §32 |
| `optv5f_rms_b0999_s32p0_2k_r1` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 32 × β₂.999 · 2000 步 retry** | ✅ done（360-2 GPU0） | 1.272 / 6.833 / +5.562 @2000 | §32 |
| `optv5f_rms_b099_s64p0` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 64 × β₂.99 · 1000 步** | ✅ done | +2.629 @1000 | §32 |
| `optv5f_rms_b0999_s64p0` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 64 × β₂.999 · 1000 步** | ✅ done | +2.607 @1000 | §32 |
| `optv5f_rms_b099_s64p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 64 × β₂.99 · 2000 步** | ✅ done | 1.225 / 6.965 / +5.740 @2000 | §32 |
| `optv5f_rms_b0999_s64p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 64 × β₂.999 · 2000 步** | ✅ done | 1.329 / 6.755 / +5.425 @2000 | §32 |
| `optv5f_rms_b099_s128p0` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 128 × β₂.99 · 1000 步** | ✅ done | +2.606 @1000 | §32 |
| `optv5f_rms_b0999_s128p0` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 128 × β₂.999 · 1000 步** | ✅ done | +2.613 @1000 | §32 |
| `optv5f_rms_b099_s128p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 128 × β₂.99 · 2000 步** | ✅ done | 1.229 / 6.953 / +5.724 @2000 | §32 |
| `optv5f_rms_b0999_s128p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 128 × β₂.999 · 2000 步** | ✅ done | 1.359 / 6.758 / +5.399 @2000 | §32 |
| `optv5f_rms_b099_s256p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 256 × β₂.99 · 2000 步** | ✅ done | 1.301 / 6.844 / +5.543 @2000 | §32 |
| `optv5f_rms_b099_s512p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 512 × β₂.99 · 2000 步** | ✅ done | 1.415 / 6.889 / +5.474 @2000 | §32 |
| `optv5f_rms_b099_s1024p0_2k` | 2026-08-26 | **高 table-LR β₂ 收敛批 · scale 1024 × β₂.99 · 2000 步** | ✅ done | 1.462 / 6.797 / +5.335 @2000 | §32 |
| `s1v5_128_tbl_bi2_R{16000..2347000}` | 2026-08-26 | **S1 table-size · bigram-R single-variable scaling · table LR 128×** | ✅ done | 18/18；gap@1000 = 2.305–2.771 | §33 |
| `s1v5_128_tbl_tri2_R{16000..2347000}` | 2026-08-26 | **S1 table-size · trigram-R single-variable scaling · table LR 128×** | ✅ done | 18/18；gap@1000 = 1.044–3.264 | §33 |
| `s1v5_128_frequency_main` | 2026-08-26 | **S1 frequency-bin · main input double-branch · table LR 128×** | ✅ done | gap@1000 = 2.736 | §33 |
| `s1v5_128_ep{0p125..2p0}xL4_3ep` | 2026-08-26 | **S1 epoch-length · 12 L4 multiples × 3 epochs · table LR 128×** | ✅ done | 12/12；gap = 4.417→2.728→5.661 | §33 |
| `s1v5_128_ep1xL4_10ep_{both,nogram}` | 2026-08-26 | **S1 L4 long replay · 10 epochs · table LR 128×** | ✅ done | both=8.917；nogram=0.480 @3370 | §33 |
| `vanilla_input_1000_seed42` | 2026-08-23 | 干净 vanilla 复现 · input 注入 · 1000 步 | ✅ done | **+0.858 @1000** | §14 |
| `vanilla_nogram_1000_seed42` | 2026-08-23 | 干净 vanilla 复现 · 无 n-gram 对照 · 1000 步 | ✅ done | **+0.038 @1000** | §14 |
| `nglab1x_input_reset_e2` | 2026-08-24 | P1 因果 · e2 边界全 table 回滚 | ✅ done | **+0.054 @1000（−94%）** | §15 |
| `nglab1x_input_reset_e1` | 2026-08-24 | P1 因果 · e1 边界全 table 回滚 | ✅ done | +0.351 @1000（−59%） | §15 |
| `nglab1x_input_mask_e1` | 2026-08-24 | P2 因果 · e1 边界屏蔽 readout | ✅ done | **+0.058 @1000（−93%）** | §15 |
| `nglab1x_input_freeze_table_e1` | 2026-08-24 | P2 因果 · e1 边界冻结 table | ✅ done | +0.601 @1000（−30%） | §15 |
| `nglab1x_input_freeze_backbone_e1` | 2026-08-24 | P2 因果 · e1 边界冻结 backbone | ✅ done | +0.780 @1000（−9%） | §15 |
| `nglab_v` | 2026-08-05 | 注入点消融 · v | ✅ done | 0.33 @999 | §2 |
| `nglab_y` | 2026-08-05 | 注入点消融 · y | ✅ done | 3.50 @999 | §2 |
| `nglab_input` | 2026-08-05 | 注入点消融 · input | ✅ done | 0.79 @999 | §2 |
| `nglab2x_v` | 2026-08-06 | 双倍训练集 · v | ✅ done | 1.169 @2000 | §4 |
| `nglab2x_y` | 2026-08-06 | 双倍训练集 · y | ✅ done | 3.101 @2000 | §4 |
| `nglab2x_input` | 2026-08-06 | 双倍训练集 · input | ✅ done | 0.687 @2000 | §4 |
| `nglab2x_input_v10` | 2026-08-06 | 双倍训练集 · input · v10 细曲线 | ⛔ superseded | val 移动窗，已停 | §6 |
| `nglab0_5x_input` | 2026-08-06 | 半 epoch 训练集 · input · v10 | ⛔ superseded | val 移动窗，已停 | §7 |
| `nglab2x_input_v10_fv` | 2026-08-06 | 双倍训练集 · input · v10 · **fixed-val** | ✅ done | 0.502 @2000 | §6 |
| `nglab0_5x_input_fv` | 2026-08-06 | 半 epoch 训练集 · input · v10 · **fixed-val** | ✅ done | 4.952 @2000 | §7 |
| `nglab1x_v10_v` | 2026-08-06 | 标准 1x · v 注入 · v10 重跑 | ✅ done | 5.041@2000 | §8 |
| `nglab1x_v10_y` | 2026-08-06 | 标准 1x · y 注入 · v10 重跑 | ✅ done | 5.049@2000 | §8 |
| `nglab1x_v10_input` | 2026-08-06 | 标准 1x · input 注入 · v10 重跑 | ✅ done | 1.931@2000 | §8 |
| `nglab1x_v10_nogram` | 2026-08-06 | 标准 1x · 无 n-gram 对照 · v10 重跑 | ✅ done | 0.231@2000 | §8 |
| `nglab1x_opt_rmsprop_2x` | 2026-08-06 | table 优化器消融 · RMSProp lr×2（2000 步）| ✅ done | 2.376@2000 | §9/9a/9b |
| `nglab1x_opt_adamw_090999` | 2026-08-06 | table 优化器消融 · AdamW(0.9,0.999) | ✅ done | 0.912@1000 | §9/9a |
| `nglab1x_opt_adamw_080950` | 2026-08-06 | table 优化器消融 · AdamW(0.8,0.95) | ✅ done | 0.709@1000 | §9/9a |
| `nglab1x_opt_sgd_09` | 2026-08-06 | table 优化器消融 · SGD momentum 0.9 | ✅ done | −0.002@1000（table 未学）| §9/9a/9b |
| `nglab1x_opt_rmsprop_4x` | 2026-08-06 | table 优化器消融 · RMSProp lr×4（剂量上限，2000 步对照）| ✅ done | 4.742@2000 | §9a/9b/9d |
| `nglab1x_opt_rmsprop_4x_b2_099` | 2026-08-07 | β2 反向扫描 · RMSProp 4x · b2=0.99 | ✅ done | 5.143@2000 | §9d |
| `nglab1x_opt_rmsprop_4x_b2_098` | 2026-08-07 | β2 反向扫描 · RMSProp 4x · b2=0.98 | ✅ done | 5.155@2000 | §9d |
| `nglab025x_b2_099` | 2026-08-07 | 短 epoch × β2 · 0.25x · b2=0.99 | ✅ done | 13.577@2000 | §11 |
| `nglab05x_b2_099` | 2026-08-07 | 短 epoch × β2 · 0.5x · b2=0.99 | ✅ done | 5.017@2000 | §11 |
| `nglab1x_opt_rmsprop_2x_s43` | 2026-08-06 | RMSProp lr×2 · seed43 | ✅ done | 2.111@1000 | §9a/9b |
| `nglab1x_opt_adamw_090999_s43` | 2026-08-06 | AdamW(0.9,0.999) · seed43 | ✅ done | 1.443@1000 | §9a/9b |
| `nglab1x_opt_rmsprop_2x_s44` | 2026-08-06 | RMSProp lr×2 · seed44 | ✅ done | 2.089@1000 | §9a/9b |
| `nglab1x_opt_adamw_090999_s44` | 2026-08-06 | AdamW(0.9,0.999) · seed44 | ✅ done | 1.672@1000 | §9a/9b |
| `nglab0_25x_input_fv` | 2026-08-07 | shard 扫描 · 0.25x | ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab0_75x_input_fv` | 2026-08-07 | shard 扫描 · 0.75x | ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab1_5x_input_fv` | 2026-08-07 | shard 扫描 · 1.5x | ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab2_5x_input_fv` | 2026-08-07 | shard 扫描 · 2.5x | ⛔ superseded | val 与 train 重叠 | §10 |
| `nglab3x_input_fv` | 2026-08-07 | shard 扫描 · 3x | ⛔ superseded | val 与 train 重叠 | §10 |
| `nglab4x_input_fv` | 2026-08-07 | shard 扫描 · 4x | ⛔ superseded | val 与 train 重叠 | §10 |
| `nglab2_5x_input_fv_v2` | 2026-08-07 | shard 扫描 · 2.5x（修正 val）| ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab3x_input_fv_v2` | 2026-08-07 | shard 扫描 · 3x（修正 val）| ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab4x_input_fv_v2` | 2026-08-07 | shard 扫描 · 4x（修正 val）| ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab5x_input_fv` | 2026-08-07 | shard 扫描 · 5x（360-2）| ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab6x_input_fv` | 2026-08-07 | shard 扫描 · 6x（360-2）| ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `nglab8x_input_fv` | 2026-08-07 | shard 扫描 · 8x（360-2）| ⛔ superseded | 被 v5 frequency-refresh 取代 | §10 |
| `t5z_zipf_s42/s43/s44` | 2026-08-07 | toy 严格 Zipf 分布（N_r∝1/r²）· per-bucket gap | ✅ done | 7.01/7.96/7.56 @2000 | §13 |
| `nglab_plot_baseline` | 2026-08-06 | 基础实验统计与图表归档 | ✅ done | 15 bins + log/log-log | §10 |
| `ngram5_order5_trigram_fixed` | 2026-08-24 | **自然语言 5gram（order=5）· +trigram 注入 · input · fixed** | ✅ done | −0.0067 @2000 | §19 |
| `ngram5_order5_puretransformer_fixed` | 2026-08-24 | **自然语言 5gram（order=5）· 纯 transformer 对照 · fixed** | ✅ done | +0.0054 @2000 | §19 |
| `ngram5_order5_trigram_lr1x_fixed` | 2026-08-24 | **自然语言 5gram（order=5）· +trigram · 表 LR ×1** | ✅ done | +0.0015 @2000 | §19 |
| `ngram5_order5_trigram_lr4x_fixed` | 2026-08-24 | **自然语言 5gram（order=5）· +trigram · 表 LR ×4** | ✅ done | −0.0092 @2000 | §19 |
| `ngram5_order5_trigram_s43_fixed` | 2026-08-24 | **自然语言 5gram（order=5）· +trigram · seed 43 复现** | ✅ done | −0.0090 @2000 | §19 |
| `l6_counttable_freq_exact_v1` | 2026-08-25 | **二元计数表 · 频率扫描 · 精确枚举** | ✅ done | slope −1.0000（f≥512） | §23 |
| `l6_response_moments_exact_v1` | 2026-08-25 | **残差—响应映射 · 二/四/绝对矩 · 精确枚举** | ✅ done | −1.0000/−0.4995/−1.9986 | §23 |
| `optv5d_rms_b095_s8p0` | 2026-08-26 | **高 table-LR β₂ gate · RMSProp β₂=.95，scale=8** | ✅ done | +2.228 @1000 | §24c |
| `optv5d_rms_b099_s8p0` | 2026-08-26 | **高 table-LR β₂ gate · RMSProp β₂=.99，scale=8** | ✅ done | +2.432 @1000 | §24c |
| `optv5d_rms_b0995_s8p0` | 2026-08-26 | **高 table-LR β₂ gate · RMSProp β₂=.995，scale=8** | ✅ done | +2.429 @1000 | §24c |
| `optv5e_rms_b095_s16p0` | 2026-08-26 | **高 table-LR β₂ gate · RMSProp β₂=.95，scale=16** | ✅ done | +2.489653 @1000 | §24c |
| `optv5e_rms_b099_s16p0` | 2026-08-26 | **高 table-LR β₂ gate · RMSProp β₂=.99，scale=16** | ✅ done | +2.599353 @1000 | §24c |
| `optv5e_rms_b0995_s16p0` | 2026-08-26 | **高 table-LR β₂ gate · RMSProp β₂=.995，scale=16** | ⚠️ failed | GPU3 CUDA peer-memory hardware error @800 | §24c |
| `optv5e_rms_b0995_s16p0_r1` | 2026-08-26 | **高 table-LR β₂ gate retry · RMSProp β₂=.995，scale=16** | ✅ done | +2.598872 @1000 | §24c |
| `ngram5_order5_sample285_v5_transformer_s42` | 2026-08-26 | **5-gram condition · sample285 受控样本量 · Transformer · v5** | planned | 待填 | §26 |
| `optv5c_{rms,adamw,sgd}_*_s{43,44}` | 2026-08-27 | **X1 optimizer × seed 复现 · v5** | ✅ done | 三臂三 seed 均值/SD见§27 | §27 |
| `ctbl_dim{192,48,12}_input_v5` | 2026-08-27 | **X2 clean 表行宽扫描 · v5** | ✅ done | gap 0.803/0.364/0.157 @1000 | §28 |

状态约定：`planned` 已登记未开跑 / `running` 运行中 / `done` 已回填 / `stalled` 超期未回填。
新实验流程：总表加一行拿到唯一 `run_id` → 正文新建 section 按 `agents.md` §3 / `docs/plan.md` 模板填写
→ 占 GPU 开跑 → 结果回填并改状态。

---

## 1. 注入点消融（2026-08-05，OPHIS 旧 run 迁移）

以下数据来自旧代码库的历史 run（`injpos_*_freq2`），
作为新 repo 的历史对照基线。新 repo 的 `train.py` 精简重写后需复现这些数值。

### 1.1 1000 步消融

| run | 注入点 | n-gram | optimizer | steps | seed | gap@999 | norm 诊断 |
|---|---|---|---|---|---|---|---|
| `injpos_v_freq2` | v | bi+tri | mixed rmsprop | 1000 | 42 | 0.60 | n-gram residual = V 的 6.5% |
| `injpos_y_freq2` | y | bi+tri | mixed rmsprop | 1000 | 42 | 1.82 | — |
| `injpos_input_freq2` | input | bi+tri | mixed rmsprop | 1000 | 42 | 0.64 | n-gram residual = wte 的 4.77x |
| `injpos_baseline_no_ngram` | — | none | mixed rmsprop | 1000 | 42 | 0.03 | — |

### 1.2 2000 步延长

| run | 注入点 | gap@999 | gap@1999 |
|---|---|---|---|
| `injpos_v_long2000` | v | 0.60 | 4.70 |
| `injpos_y_long2000` | y | 2.10 | 4.65 |
| `injpos_input_long2000` | input | 0.75 | 2.98 |

### 1.3 Table norm × gap（theory obs，103 个点）

| step | v gap | y gap | input gap | v bg_rms | y bg_rms | input bg_rms |
|---|---|---|---|---|---|---|
| 10 | 0.00 | 0.00 | 0.00 | 0.037 | 0.037 | 0.037 |
| 100 | -0.004 | 0.011 | 0.002 | 0.042 | 0.083 | 0.068 |
| 337(e2) | -0.047 | -0.048 | -0.093 | 0.121 | 0.119 | 0.113 |
| 686(e3) | -0.077 | -0.790 | -0.342 | 0.160 | 0.149 | 0.150 |
| 999 | -0.542 | -1.900 | -0.672 | — | — | — |

注：gap = val - train（正值=gap）。bg_rms = bigram table layer_1 table_0 的 param rms。

### 1.4 关键结论

1. **v 注入无 gap 的原因是数值尺度问题**：n-gram value norm 只有 V 的 6.5%，信号被 V 淹没。
2. **y/input 注入都能产生 gap**：只要 n-gram 信号不走 attention 混合、能有效到达输出。
3. **gap 仅依赖 n-gram memory**：不需要 current shell / Muon / RoPE / RMSNorm。
4. **Table norm 增长速度不是 gap 的决定因素；注入点（信号能否到达输出）才是。**

## 2. 新 repo 复现验证（2026-08-05 完成）

用 `code/train.py`（精简版）重跑 v/y/input 三注入点，核对 gap 数值是否与 §1 一致。

| run | 注入点 | steps | gap@999（目标）| gap@999（实测）| 状态 |
|---|---|---|---|---|---|
| `nglab_v` | v | 1000 | 0.60 | 0.33 | ✅ |
| `nglab_y` | y | 1000 | 1.82 | 3.50 | ✅ |
| `nglab_input` | input | 1000 | 0.64 | 0.79 | ✅ |

**相对顺序一致**：y > input > v。绝对数值与旧实验有差异（LR schedule 不同），但现象完全可复现。

### 2.1 频率 bin 分解验证

- bigram novel frac: 4.3%（旧实验 ~4%）✅
- trigram novel frac: 31.2%（旧实验 ~30%）✅
- novel + 低频 bucket 主导 gap（详见 `fig_gap_by_freq.html`）

## 3. 频率 bin 分解（2026-08-05 完成）

用 `code/ngram_freq.py` 构建频率索引，统计 per-bin 的 mean loss 与 total contribution。

结果：novel + 低频 bucket（1-5）主导 gap；高频 bucket（5k+）gap 贡献 ≈ 0。与旧实验一致。

## 4. 双倍 training size 延长实验（2026-08-06，ophis-gpu）

目的：把 fixed replay 的 train 数据从 shard 1 扩大到 shard 1+2，观察更长的 epoch 平台是否能让 replay gap 更清楚。三种注入点使用完全相同的 setting，并行跑到 2000 steps。

### 4.1 Setting

| 项目 | setting |
|---|---|
| train shards | `1,2`（约 2x，约 600 steps / epoch） |
| validation shards | `3,4,5,6,7,8,9,10,6542` |
| model | vanilla nanoGPT, 8L / 6H / 768D |
| n-gram | trainable bigram + trigram |
| optimizer | backbone AdamW + table RMSProp (`beta1=0`, `beta2=0.999`) |
| learning rate | `0.004`，沿用原 warmup / warmdown schedule |
| seed | `42` |
| steps | `2000` |
| validation / freq eval | 每 50 steps，4 batches |
| table norm | 每 10 steps |
| runs | `nglab2x_v`, `nglab2x_y`, `nglab2x_input` |

双倍训练集对应的 exact-context frequency index 为 `data/freq_index_train2x.npz`。每个 run 均保留 `train_log.jsonl`、`table_norm.jsonl`、`freq_bin_loss.jsonl`、`summary.json` 和原始 `train.log`；本地备用归档为 `data/nglab2x_runs.tar.gz`。

### 4.2 Gap 结果

| run | 注入点 | gap@1000 | gap@1200 | gap@1500 | gap@2000 |
|---|---:|---:|---:|---:|---:|
| `nglab2x_v` | v | 0.001 | 0.068 | 0.482 | **1.169** |
| `nglab2x_y` | y | 0.220 | 0.752 | 2.174 | **3.101** |
| `nglab2x_input` | input | 0.152 | 0.213 | 0.460 | **0.687** |

### 4.3 Epoch 平台统计

| run | epoch | step range | gap mean | gap min–max | final gap |
|---|---:|---:|---:|---:|---:|
| v | 1 | 50–600 | -0.005 | -0.053–0.069 | 0.003 |
| v | 2 | 650–1200 | 0.032 | -0.015–0.068 | 0.068 |
| v | 3 | 1250–1800 | 0.434 | 0.199–0.654 | 0.614 |
| v | 4 | 1850–2000 | 0.950 | 0.787–1.169 | 1.169 |
| y | 1 | 50–600 | 0.001 | -0.044–0.053 | -0.009 |
| y | 2 | 650–1200 | 0.394 | 0.018–0.752 | 0.752 |
| y | 3 | 1250–1800 | 1.872 | 0.993–2.345 | 2.308 |
| y | 4 | 1850–2000 | 2.903 | 2.603–3.101 | 3.101 |
| input | 1 | 50–600 | -0.008 | -0.045–0.021 | 0.006 |
| input | 2 | 650–1200 | 0.140 | -0.015–0.220 | 0.213 |
| input | 3 | 1250–1800 | 0.416 | 0.255–0.582 | 0.507 |
| input | 4 | 1850–2000 | 0.618 | 0.538–0.687 | 0.687 |

Epoch boundaries occur around steps 600, 1200, 1800. The platform effect is substantially clearer than in the original one-shard run: y shows a stepwise progression `~0 → 0.75 → 2.3 → 3.1`, input shows `~0 → 0.21 → 0.51 → 0.69`, and v only becomes visibly positive after the third replay.

### 4.4 Final table norm and frequency coverage

Final table RMS:

| run | representative bigram RMS | representative trigram RMS | all norm rows |
|---|---:|---:|---:|
| v | 0.1202 (`layer_01`) | 0.1371 (`layer_01`) | 200 |
| y | 0.1449 (`layer_01`) | 0.1611 (`layer_01`) | 200 |
| input | 0.0860 (`layer_01`) | 0.0876 (`layer_01`) | 200 |

At step 2000, every frequency file has 40 checkpoints and every checkpoint covers all 15 buckets for both bigram and trigram. Bucket fractions sum to exactly 1.0 for train and validation, with 589,824 evaluated tokens per split.

| branch | train novel | val novel | train hit=1 | val hit=1 |
|---|---:|---:|---:|---:|
| bigram | 0.0% | 2.85% | 2.34% | 1.61% |
| trigram | 0.07% | 25.58% | 22.38% | 7.57% |

The novel fractions decrease relative to the one-shard index because the doubled training set covers more contexts, while the low-frequency trigram mass remains substantial. The full raw outputs are retained for future plots and alternative optimizer comparisons.

## 10. 基础实验统计与图表归档（2026-08-06）

目的：把已经完成并确认口径的基础实验，连同完整统计和图表生成方法登记为
可复用的干净基线。该 section 不启动新训练，不覆盖其他 Agent 的 running
实验。

> 口径说明：早期 `nglab_v/y/input` 是 validation/freq eval 每 50 步的历史
> 基线；后续 canonical 主线统一为每 10 步（v10），以获得更密集、更清晰的
> epoch replay 曲线。代码默认值、`run_injpos.sh` 和 `docs/plan.md` 均采用 v10。

### 数据与统计口径

基础图表使用 `nglab_v`、`nglab_y`、`nglab_input` 的完整日志：

| 统计层级 | 产物 | 内容 |
|---|---|---|
| global step | `train_log.jsonl` | train loss、fixed validation loss、global gap、epoch |
| table memory | `table_norm.jsonl` | 每个 table 的 RMS/norm 随 step 变化 |
| frequency bucket | `freq_bin_loss.jsonl` | bigram/trigram、train/val、15 个真实 bucket 的 token count、fraction、mean token loss、total contribution |
| summary | `summary.json` | run 的最终摘要 |

frequency loss 使用 unreduced per-token cross-entropy：先得到每个 token 的
loss，再按照真实 context 的 training hit-count bucket 聚合。运行文件保存
每个 bucket 的聚合统计，而不是保存全部逐 token loss 数组。

`novel` 是 training hit count 为 0 的 context。它可以有 validation loss，
但 train 侧没有对应 token，因此没有 train mean loss，不能定义标准的
`val loss - train loss`。因此 novel 保留在 raw/fraction 图中，并从 gap
和 log/log-log 图中排除。

### 图表思想

| 图表 | 目的 |
|---|---|
| global loss/gap | 显示 fixed-order replay 后 train 与 validation 的分叉，以及 epoch boundary |
| table norm × loss | 确认 n-gram table 确实被写入，并对齐 memory growth 与 gap |
| frequency-bin timeline | 观察每个频率 bucket 的 per-token loss、gap 和 total contribution 如何随 replay 演化 |
| frequency histogram + final gap | 柱表示 train/val token fraction，曲线表示末态 per-bin gap；同时看“占多少”和“差多少” |
| log-x / log-log frequency-to-gap | 用 bucket 命中次数的几何中点定量观察频率与末态 gap 的关系；横向误差线表示 bucket 范围 |

### 代码与产物

- canonical generator：`docs/plot_scripts/gen_all_figures.py`
- 作图目录说明：`docs/plot_scripts/README.md`
- summary/norm JSON builder：`docs/plot_scripts/build_injpos_data_json.py`
- epoch-length comparison：`docs/plot_scripts/gen_epoch_scale_figs.py`
- 早期 provenance generator：`docs/plot_scripts/gen_injpos_plot.py`
- 输出目录：`docs/figs/`
- public guide mirror：
  `guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide/`
- 运行数据：历史结果目录（未修正版，不作为当前权威数据）

本次归档的图表生成 commit：

- plotting pipeline：`10cb1b0`
- public guide log/log-log view：`d59111c`


---

## 6. 双倍训练集 v10 细曲线（2026-08-06，input）

目的：§4 的 nglab2x 批是 v50（每 50 步），无法与 v10 标准曲线对齐；本 run 用 v10 重跑 input 注入，保证 epoch 平台与 gap 曲线可与标准 1x（v10）逐点比较。

### Setting

| 项 | 值 |
|---|---|
| train shards | `1,2`（约 2x，~674 steps/epoch）|
| validation shards | `3,4,5,6,7,8,9,10,6542`（与 §4 一致）|
| steps / seed | 2000 / 42 |
| validation / freq eval | **每 10 steps**（v10）|
| freq index | `data/freq_index_train2x_fine.npz` |
| run | `nglab2x_input_v10`（首跑）→ **`nglab2x_input_v10_fv`**（fixed-val 重跑，当前） |

> **val-fix（2026-08-06 20:3x）**：首跑时 `evaluate_val` 从 `val_iter` 顺序取批，
> v10 下 200 次 eval × 4 批会让 val 曲线在 val 集上滑动（移动窗），
> 不满足「val 数据始终同一套」。已在 `code/train.py` 修复：启动时一次性捕获
> `fixed_val_batches`（val loss）与 `fixed_freq_val_batches`（val 侧 freq-bin），
> 每次 eval 复用同一批 val 数据；train 仍是唯一移动队列。首跑（移动窗）已停，
> 首轮数据为历史未修正版结果，正式结果以 `_fixed` run 为准。

### 结果

| run | final train | final val | gap@2000 | 观测 epoch 长（边界步）|
|---|---|---|---|---|
| `nglab2x_input_v10_fv` | 3.041 | 3.543 | **+0.502** | ~450（460, 900, 1350, 1800）|

- v10 细曲线 200 个点（每 10 步），fixed-val：val loss 每次都测同一批 val 数据。
- 关键现象：2x 下 2000 步只走 ~4.4 个 epoch，train 下降但 gap 到 2000 步仍只有 +0.5
  （对照 0.5x 同预算走 18 个 epoch，gap +4.95；见 §7）。

## 7. 半 epoch 训练集（2026-08-06，input）

目的：把 fixed replay 的 epoch 长度减半，与 1x（337 steps/epoch）和 2x（674 steps/epoch）在相同 2000 步预算下对比「epoch 平台长度 → replay gap」的剂量关系。

### Setting

| 项 | 值 |
|---|---|
| train shards | `60`（shard_00060 = shard_00001 前 12132 行，~168 steps/epoch）|
| validation shards | `2,3,4,5,6,7,8,9,10,6542`（与标准一致）|
| steps / seed | 2000 / 42 |
| validation / freq eval | 每 10 steps |
| freq index | `data/freq_index_train0_5x.npz` |
| run | `nglab0_5x_input`（首跑）→ **`nglab0_5x_input_fv`**（fixed-val 重跑，当前） |

> **val-fix（2026-08-06 20:3x）**：同上 §6——首跑 val 为移动窗，已停；
> 正式结果以 `_fv` run 为准（train 侧行为完全一致，仅 val 改为固定批次）。

### 结果

| run | final train | final val | gap@2000 | 观测 epoch 长（边界步）|
|---|---|---|---|---|
| `nglab0_5x_input_fv` | 1.776 | 6.728 | **+4.952** | ~110–120（120, 230, 350, …）|

- v10 细曲线 200 个点，fixed-val。
- 关键现象：epoch 减半后，train 塌到 1.78 而 val 升到 6.73，gap 几乎是 2x（+0.50）的 **10 倍**；
  gap 从 epoch 2（~step 120）就开始转正，符合「epoch 平台越短 → replay 越早、越猛」的剂量关系。
- 与 1x（`nglab1x_v10_input`，parallel agent，fixed-val，观测 ~230 steps/epoch）对照见
  `docs/figs/epoch_scale/epoch_scale_train_val_gap.png`（3 条曲线，均已 2000 步完成）。

### 剂量关系汇总（gap@2000，input 注入，v10，fixed-val，seed 42）

| epoch 长 | run | 观测 steps/epoch | train@2000 | val@2000 | gap@2000 |
|---|---|---|---|---|---|
| 0.5x | `nglab0_5x_input_fv` | ~110–120 | 1.776 | 6.728 | **+4.95** |
| 1x | `nglab1x_v10_input` | ~230 | 2.707 | 4.669 | **+1.96** |
| 2x | `nglab2x_input_v10_fv` | ~450 | 3.041 | 3.544 | **+0.50** |

结论：epoch 平台越短，train 塌缩越深、val 翘起越早越强；2000 步预算内
0.5x 的 gap 是 2x 的 ~10 倍。

## 8. 标准 1x v10 重跑（2026-08-06，blog 克隆任务）

目的：博客 `ngram-gap-mechanism-guide` 主线的 v/y/input 消融原为 1000 步、
validation 每 50 步；本批用 **v10 标准（validation + freq eval 每 10 步）重跑到 2000 步**，
重做 v/y/input 三注入点 + 无 n-gram 对照，产出更细的 loss/gap 曲线，
并克隆一份博客文档。

### Setting

| 项 | 值 |
|---|---|
| train shards | `1`（标准 1x，约 250–316 steps/epoch，含 freq eval 消耗）|
| validation shards | `2,3,4,5,6,7,8,9,10,6542`（与标准一致）|
| steps / seed | 2000 / 42 |
| validation / freq eval | **每 10 steps**（v10）|
| table norm | 每 10 steps |
| freq index | `data/freq_index.npz` |
| runs | `nglab1x_v10_v` / `nglab1x_v10_y` / `nglab1x_v10_input` / `nglab1x_v10_nogram` |

### 结果（ophis-gpu 首波 · 23 桶 freq 统计，已 done；360-1 15 桶重跑进行中）

- ophis-gpu 首波（2026-08-06 21:41–22:41 CST，run_id 同上，使用当时 23 桶版 `ngram_freq.py`）：
  - `nglab1x_v10_v`：final_gap **4.9497**（train 1.3536 / val 6.3033）
  - `nglab1x_v10_y`：final_gap **5.0552**（train 1.3601 / val 6.4153）
  - `nglab1x_v10_input`：final_gap **1.9615**（train 2.7072 / val 4.6687）
  - `nglab1x_v10_nogram`：final_gap **0.2253**（train 3.0167 / val 3.2420）
  - freq bin 统计为 23 桶（另一 agent 的未提交改动在开跑前已同步到 ophis），train/val gap 与桶数无关。
- 360-1 重跑（2026-08-06 22:33–23:51 CST，15 桶提交版 `ngram_freq.py`，4 卡并行；与博客口径一致，作为克隆文档的权威数据）：
  - `nglab1x_v10_v`：final_gap **5.0406**（train 1.2377 / val 6.2783）
  - `nglab1x_v10_y`：final_gap **5.0493**（train 1.3698 / val 6.4191）
  - `nglab1x_v10_input`：final_gap **1.9308**（train 2.7082 / val 4.6390）
  - `nglab1x_v10_nogram`：final_gap **0.2312**（train 3.0033 / val 3.2345）
  - epoch 边界（每 10 步 val 记录）：230 / 460 / 680 / 910 / 1130 / 1360 / 1580 / 1810（约 226 steps/epoch，freq eval 每 10 步消耗 4 个 train batch）。
  - 与 ophis 首波交叉验证：四 run gap 差 < 0.09（桶数不影响 train/val loss）。
- 产物：图 `docs/figs/main/`；数据 `data/injpos_ablation_data.json`；克隆博客 `guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide-v10/`（validation 每 10 步 · 2000 steps）。


## 9. Table 优化器消融（2026-08-06，input，计划）

目的：记录历史实验中 n-gram table 使用 RMSProp（β=(0.0,0.999)，无一阶矩）的行为；当前标准已切换为 β₂=0.99、表学习率 ×2。
怀疑「table 学得慢/滞后」；本批测试替代优化器是否让 table 更快写入、以及 gap 曲线如何变化。

### Setting

| 项 | 值 |
|---|---|
| injection | `input`（blog 主线默认）|
| train shards / val | `1` / `2,3,4,5,6,7,8,9,10,6542`（与标准一致）|
| steps / seed | 1000 / 42 |
| val / freq / norm | 每 10 steps（v10，与 `nglab1x_v10_*` 对齐）|
| 只变 | table optimizer（backbone 恒为 AdamW lr=0.004）|

| arm | table optimizer | table betas | table lr |
|---|---|---|---|
| `rmsprop_2x` | RMSProp | (0.0, 0.999) | 0.008（lr×2）|
| `adamw_090999` | AdamW | (0.9, 0.999) | 0.004 |
| `adamw_080950` | AdamW | (0.8, 0.95) | 0.004 |
| `sgd_09` | SGD+momentum | momentum=0.9 | 0.004 |

launcher：`code/cluster/run_table_opt.sh <arm> <gpu>`。
代码：`code/train.py` 新增 `--table_optimizer / --table_lr_scale / --table_betas`（默认不变）。
对照基线：`nglab1x_v10_input`（同 flags，v10/2000）与 `nglab_input`（v50/1000）。

### 结果（ophis-gpu 首波 · 23 桶 freq 统计，已 done；360-1 15 桶重跑进行中）

- ophis-gpu 首波（2026-08-06 21:41–22:41 CST，run_id 同上，使用当时 23 桶版 `ngram_freq.py`）：
  - `nglab1x_v10_v`：final_gap **4.9497**（train 1.3536 / val 6.3033）
  - `nglab1x_v10_y`：final_gap **5.0552**（train 1.3601 / val 6.4153）
  - `nglab1x_v10_input`：final_gap **1.9615**（train 2.7072 / val 4.6687）
  - `nglab1x_v10_nogram`：final_gap **0.2253**（train 3.0167 / val 3.2420）
  - freq bin 统计为 23 桶（另一 agent 的未提交改动在开跑前已同步到 ophis），train/val gap 与桶数无关。
- 360-1 重跑（2026-08-06 22:33–23:51 CST，15 桶提交版 `ngram_freq.py`，4 卡并行；与博客口径一致，作为克隆文档的权威数据）：
  - `nglab1x_v10_v`：final_gap **5.0406**（train 1.2377 / val 6.2783）
  - `nglab1x_v10_y`：final_gap **5.0493**（train 1.3698 / val 6.4191）
  - `nglab1x_v10_input`：final_gap **1.9308**（train 2.7082 / val 4.6390）
  - `nglab1x_v10_nogram`：final_gap **0.2312**（train 3.0033 / val 3.2345）
  - epoch 边界（每 10 步 val 记录）：230 / 460 / 680 / 910 / 1130 / 1360 / 1580 / 1810（约 226 steps/epoch，freq eval 每 10 步消耗 4 个 train batch）。
  - 与 ophis 首波交叉验证：四 run gap 差 < 0.09（桶数不影响 train/val loss）。
- 产物：图 `docs/figs/main/`；数据 `data/injpos_ablation_data.json`；克隆博客 `guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide-v10/`（validation 每 10 步 · 2000 steps）。

### 9a. Table 优化器消融（郭绍阳，wave1 done / wave2 running）

> 与 §9 的 v10 主线重跑并行进行；只换 table optimizer，backbone 恒为 AdamW lr=0.004，input 注入、1x shard、seed42、v10。

**Wave1 结果（1000 步，2026-08-06 20:41–22:05，ophis）**

| arm | gap@1000 | norm@1000 | norm_growth(10→1000) |
|---|---:|---:|---:|
| RMSProp 1x（`nglab1x_v10_input`）| +0.953 | 0.0840 | +133% |
| RMSProp 2x（`rmsprop_2x`）| +0.939 | **0.1335** | **+270%** |
| AdamW (0.9,0.999) | +0.912 | 0.0720 | +99% |
| AdamW (0.8,0.95) | +0.709 | 0.0623 | +73% |
| RMSProp v50 基线（`nglab_input`）| +0.785 | 0.0781 | +117% |

- 结论：**加速 table 写入最直接有效的是把 RMSProp lr 提到 2×**（norm@1000 +59%）；AdamW 一阶矩对「每 epoch 只被读一次」的稀疏行帮助有限，norm 反而更小、gap 更小（0.8/0.95 时 0.709）。norm 增速 ↑ ↔ gap 大小 ↑ 的对应仍成立。
- 图：`docs/figs/table_opt/fig_table_opt.{svg,png}`；脚本 `docs/plot_scripts/analyze_table_opt.py`。

**Wave2（2026-08-06 23:53 启动，多机并行）**

- ophis GPU6 `sgd_09`(1000)、GPU7 `rmsprop_2x`(2000，验证更快写表→epoch3/4 更大 gap)
- 360-1 GPU4/5/6：`rmsprop_2x_s43`、`adamw_090999_s43`、`rmsprop_4x`(1000)
- 360-2 GPU0/1：`rmsprop_2x_s44`、`adamw_090999_s44`
- 分析：跑完自动 rsync + `analyze_table_opt.py` 汇总。

### 9b. Wave2：多 seed × LR 剂量 × SGD 机制验证（2026-08-07 回填）

> 承接 §9a wave1；wave2 目标：(i) 用 s43/s44 重复 RMSProp2x / AdamW(0.9,0.999)，估计 seed 方差；(ii) RMSProp 2x 延到 2000 步验证「更快写表 → epoch3/4 gap 更大」；(iii) 新增 RMSProp 4x 验证 LR 剂量效应；(iv) SGD+momentum 验证「无自适应归一化时 table 是否学得动」。

**Wave2 结果（360-1 / 360-2 / ophis，2026-08-06 23:53–08-07 ~10:20）**

| arm | seed | steps | gap@1000 | final_gap | norm@1000 | norm_growth(10→1000) |
|---|---:|---:|---:|---:|---:|---:|
| RMSProp 1x（`nglab1x_v10_input`）| 42 | 2000 | +0.953 | +1.931 | 0.0838 | +132% |
| RMSProp 2x | 42 | 2000 | +1.145 | +2.376 | 0.1544 | +328% |
| RMSProp 2x | 43 | 1000 | +2.111 | +2.111 | 0.1343 | +272% |
| RMSProp 2x | 44 | 1000 | +2.089 | +2.089 | 0.1350 | +274% |
| RMSProp 4x | 42 | 1000 | +2.182 | +2.182 | 0.2502 | +591% |
| AdamW (0.9,0.999) | 42 | 1000 | +0.912 | +0.912 | 0.0720 | +99% |
| AdamW (0.9,0.999) | 43 | 1000 | +1.443 | +1.443 | 0.0713 | +98% |
| AdamW (0.9,0.999) | 44 | 1000 | +1.672 | +1.672 | 0.0712 | +97% |
| SGD mom0.9 | 42 | 1000 | **−0.002** | −0.002 | **0.0361** | **~0%** |

**多 seed 汇总（mean ± std）**：RMSProp 2x norm@1000 = 0.1413 ± 0.0093（n=3）；AdamW(0.9,0.999) norm@1000 = 0.0715 ± 0.0003（n=3）。

**关键结论**

1. **SGD 无 gap 的机制 = table 根本没学**：SGD mom0.9 下全部 7 张 table（4 bigram + 3 trigram 层）的 param RMS 在 1000 步内纹丝不动（0.03608，精确到 1e-5 无变化），gap≈0 是「table 从未写入」而非「table 学好了但无过拟合」。原因：SGD 的更新量正比于梯度幅值，而 table 行是稀疏命中（每 epoch 每行只被读几次）+ 147k token batch 平均，单步梯度 ~1e-5 量级，×lr=0.004 → 每步有效更新 ~4e-8，1000 步积累不可见。RMSProp/AdamW 用 per-param `g/√EMA(g²)` 把有效步长固定到 ≈lr（与梯度幅值无关），所以 table 才学得动。**这反证了：标准 RMSProp 的每参数自适应归一化是 table 学习的必要条件。**
2. **LR 剂量效应（超线性）**：RMSProp 1x→2x：norm@1000 0.084→0.144（+71%），gap@1000 +0.95→+1.78（s42/s43/s44 均值 1.782±0.451）；2x→4x：norm@1000→0.250（再 +73%），gap@1000→+2.18。即 table LR 每翻倍，norm@1000 约再 +70%，gap 相应放大。2000 步口径：RMSProp 2x gap +2.376 vs 1x +1.931（+23%），norm@2000 0.179 vs 0.097（+85%）——更快写表 → epoch 3/4 的 gap 显著更大，直接回应「table 学得慢/滞后」。
3. **AdamW 一阶矩对稀疏行帮助有限**：AdamW(0.9,0.999) norm@1000 0.072 反而 ≤ RMSProp 1x 的 0.084（多 seed 稳定），gap@1000 1.34±0.32 略高但受 norm 上限约束；(0.8,0.95) 更慢（0.062 / +0.709）。动量在「每 epoch 只被读一次」的稀疏行上累积价值低。
4. **norm 先行、gap 滞后**：RMSProp 4x 在 step500 norm 已达 0.22（≥2x@1000）但 gap 仍 +0.01，到 step1000 才跳到 +2.18——table 写入先完成、train/val 分化随后显现，与 wave1 的 norm↔gap 对应关系一致。

**产物**：图 `docs/figs/table_opt/fig_table_opt.{svg,png}`（含 s43/s44 多 seed mean±std）；脚本 `docs/plot_scripts/analyze_table_opt.py`（自动发现 `nglab1x_opt_*`，seed 42/43/44）。日志：`data/runs/nglab1x_opt_{rmsprop_2x_s43,rmsprop_4x,adamw_090999_s43,adamw_090999_s44,rmsprop_2x_s44,sgd_09}/`。

**历史 setting 说明**：本节结果使用 β₂=0.999 的历史配置；当前标准 table 使用 RMSProp（β=(0.0,0.99)，无 momentum，无 WD，表学习率 ×2），backbone 使用 AdamW（β=(0.8,0.95)，WD=0.1，lr=0.004）。

### 9c. Table 优化器消融 × 2x epoch（郭绍阳，2026-08-07 done）

> 承接 §9b（1x epoch）。用户假设：2x epoch（train shards 1,2，~450 步/epoch）下 2000 步
> 只走 ~4.4 epoch，之前 norm 曲线到 2000 步仍未平台，怀疑 beta2=0.999 不够合理。
> 本批在 2x epoch 下扫 table LR（1x/2x/4x）+ beta2（0.999 / 0.9999 / 0.99999），
> 并在后续补跑 beta2=0.99/0.98（见 §9d）。

**2x epoch · 2000 步 · seed42 · input · fixed-val（360-1/360-2，2026-08-07 12:03–13:30）**

| arm | table LR | beta2 | gap@2000 | norm@2000 | norm@1000→2000 增量 |
|---|---:|---:|---:|---:|---:|
| `rmsprop_1x`（§6 `nglab2x_input_v10_fv`）| 0.004 | 0.999 | +0.502 | 0.0851 | +0.012 |
| `rmsprop_2x` | 0.008 | 0.999 | +0.595 | 0.1576 | +0.024 |
| `rmsprop_4x` | 0.016 | 0.999 | **+2.071** | 0.3172 | +0.046 |
| `rmsprop_1x_b2_09999` | 0.004 | 0.9999 | +0.497 | 0.0850 | +0.011 |
| `rmsprop_2x_b2_09999` | 0.008 | 0.9999 | +0.608 | 0.1589 | +0.023 |
| `rmsprop_4x_b2_09999` | 0.016 | 0.9999 | **+2.110** | 0.3136 | +0.043 |
| `rmsprop_2x_b2_099999` | 0.008 | 0.99999 | +0.634 | 0.1593 | +0.024 |

**关键结论**

1. **beta2 往 1 方向（0.9999/0.99999）无效**：norm@2000 与 0.999 差 <0.004，gap 差 <0.04，
   曲线几乎重合。平台问题不是「beta2 不够接近 1」。
2. **LR 剂量效应在 2x epoch 下更强**：norm@2000 0.085→0.158→0.317（≈线性翻倍）；
   gap@2000 0.50→0.60→2.07。4x LR 在 2x epoch 下 gap 已追平 1x epoch 的 RMSProp 2x（+2.38）。
3. **平台仍未达，但增速放缓**：1500→2000 步 norm 只涨 +0.002~0.005（相对 ~1-3%/500 步）。
   按此斜率，平台可能需 4000–6000 步；加 LR 比调 beta2 有效得多。
4. 2000 步下 2x epoch 只走 ~4.4 epoch（1x 是 ~8.7），所以同臂同 norm 时 gap 普遍小于 1x：
   数据重放次数减半，记忆-过拟合路径没走完。

**产物**：图 `docs/figs/table_opt/fig_table_opt_2x.{svg,png}`、`fig_table_opt_1x_vs_2x.{svg,png}`；
脚本 `docs/plot_scripts/analyze_table_opt_2x.py`、`analyze_table_opt_1x_vs_2x.py`；
launcher `code/cluster/run_table_opt_2x.sh`（train shards 1,2 / val 3..10,6542 / 2000 步）。

### 9d. β2 反向扫描（0.99/0.98）+ 1x·RMS4x·b2=0.999@2000 对照（2026-08-07 done）

> 承接 §9c：beta2 往 1 方向（0.9999/0.99999）无效。用户问「b2 合理的值难道不是 0.99 吗」，
> 本批补跑 β2=0.99/0.98（1x·RMS4x + 1x·RMS2x），并补 1x·RMS4x·b2=0.999 的 2000 步对照
> （§9b 只有 1000 步值 +2.18）。360-2 GPU0 跑 `nglab1x_opt_rmsprop_4x`（b2=.999，2000 步）。

**1x epoch · 2000 步 · seed42 · input · fixed-val（360-2，2026-08-07 15:17–16:52）**

| arm | table LR | beta2 | gap@1000 | gap@2000 | norm@2000 |
|---|---:|---:|---:|---:|---:|
| `rmsprop_4x`（§9b 对照补跑）| 0.016 | 0.999 | +2.360 | **+4.742** | 0.4180 |
| `rmsprop_4x_b2_099` | 0.016 | 0.99 | +2.389 | **+5.143** | 0.4357 |
| `rmsprop_4x_b2_098` | 0.016 | 0.98 | +2.568 | **+5.155** | 0.4345 |
| `rmsprop_2x_b2_099` | 0.008 | 0.99 | — | +2.349 | — |
| `rmsprop_2x_b2_098` | 0.008 | 0.98 | — | +2.309 | — |

**结论**

1. **β2=0.99/0.98 在 1x·RMS4x 下确实有正效应**：gap@2000 +4.74 → +5.14/+5.16
   （+0.40~+0.41，约 +8.5%），norm@2000 0.418 → 0.434~0.436（+4%）。与「β2 越接近 1
   越无效」一致的反向方向：**降低 β2（更激进地除方差）让 table 写得更快**。
2. **但 1x·RMS2x 下 β2 影响很小**：0.99→+2.349 / 0.98→+2.309 vs 0.999 的 +2.376（§9a）
   —— LR 剂量低时 β2 的差异被 LR 上限掩盖。β2 只在 table LR 拉满（4x）时才显现。
3. 结合 §9c：β2 在 [0.98, 0.99999] 全区间内，对 gap@2000 的总影响 ≤ ±0.4，
   远小于 table LR 翻倍的效果（+0.10→+2.07）。**用户若想让 table 学更快，
   首选 table_lr_scale=2–4；β2=0.99 可作次级叠加（约 +8%）**。

**产物**：数据 `data/runs/nglab1x_opt_rmsprop_4x{,_b2_099,_b2_098}/`（已同步本地）；
launcher `code/cluster/run_table_opt_2x.sh` 风格。

## 10. shard 大小扫描（epoch 长度剂量，2026-08-07，彻夜批）

目的：用户假设「epoch 的 shard 越大，gap 越小」。§6/§7 已有 0.5x/1x/2x 三个点
（gap@2000 = 4.95 / 1.96 / 0.50），本批**连续采样 shard 大小**补 9 个点，
共 12 点（0.25x → 8x），横轴按 shard 大小（log）与「epoch 数」双尺度分析。

### Setting

| 项 | 值 |
|---|---|
| 注入点 / seed / steps | input / 42 / 2000（2.5x=3200、3x=3800、4x=5000 以便同跑 ~5.5 epoch）|
| validation / freq eval | v10，fixed-val（`fixed_val_batches`）|
| freq index | 每个 train set 单独建（`data/freq_index_train{0_25x,...,8x}.npz`）|
| 集群 | ophis-gpu GPU0-5（6 个）+ 360-2 GPU0-2（5x/6x/8x）|

| size | train shards（rows）| val shards | steps |
|---|---|---|---|
| 0.25x | 62（6066）| 2..10,6542 | 2000 |
| 0.5x ✅ §7 | 60（12132）| 2..10,6542 | 2000 |
| 0.75x | 63（18198）| 2..10,6542 | 2000 |
| 1x ✅ §8 | 1（24264）| 2..10,6542 | 2000 |
| 1.5x | 1,61（36396）| 3..10,6542 | 2000 |
| 2x ✅ §4/§6 | 1,2（48240）| 3..10,6542 | 2000 |
| 2.5x | 1,2,64（60372）| **4..10,6542** | 3200 |
| 3x | 1,2,3（72000）| **4..10,6542** | 3800 |
| 4x | 1,2,3,4（95760）| **5..10,6542** | 5000 |
| 5x | 1..5（119808）| 6..10,6542 | 2000 |
| 6x | 1..6（143568）| 7..10,6542 | 2000 |
| 8x | 1..8（191016）| 9,10,6542 | 2000 |

> 注：train 含 shard2 的 run（≥1.5x）val 用 3..10,6542 避免 val 与 train 重叠（与 §4 的 2x 一致）。
> **v1 bug（01:40 发现并修正）**：2.5x/3x/4x 首跑 val 仍用 3..10,6542，但它们的 train
> 已含 shard 3 → fixed val 前几批与 train 重叠（val 被剧透），gap 偏负
> （2.5x=−0.80 / 3x=−0.71 / 4x=−0.27 @2000）。已停，改 `_v2`（val 从最后一个 train shard 之后开始）。

### 结果（gap@2000，首批 9 点已回填；2.5x/3x/4x 为 v2 重跑，待补）

| size | run | gap@2000 | epoch@2000 |
|---|---|---|---|
| 0.25x | `nglab0_25x_input_fv` | **+12.99** | 36 |
| 0.5x | `nglab0_5x_input_fv` | **+4.95** | 18 |
| 0.75x | `nglab0_75x_input_fv` | **+2.12** | 12 |
| 1x | `nglab1x_v10_input` | **+1.96** | 9 |
| 1.5x | `nglab1_5x_input_fv` | **+0.87** | 6 |
| 2x | `nglab2x_input_v10_fv` | **+0.50** | 5 |
| 5x | `nglab5x_input_fv` | −0.05 | 2 |
| 6x | `nglab6x_input_fv` | −0.11 | 2 |
| 8x | `nglab8x_input_fv` | +0.03 | 2 |

**结论（用户假设成立）**：epoch shard 越大，gap@2000 单调变小——
0.25x→8x：+13 → +5 → +2 → +0.5 → ~0（≥5x 在 2000 步内只走 ~2 个 epoch，gap 尚未形成）。
0.25–2x 段呈近似幂律（log-log 斜率约 −1.5~−2）。

图：`docs/figs/epoch_scale/dose_response_gap2000.png`（gap@2000 vs shard size，log-x）、
`gap_vs_epochs.png`（gap vs 已过 epoch 数，横轴按 epoch 缩放）、`sweep_train_val_gap.png`。
脚本：`docs/plot_scripts/gen_shard_sweep_figs.py`。

### 结果（12 点全部完成，2026-08-07 凌晨→早上；2.5x/3x/4x 为 v2 重跑）

| size | run | gap@2000 | epoch@2000 | final gap（步数 / epoch）|
|---|---|---|---|---|
| 0.25x | `nglab0_25x_input_fv` | **+12.99** | 36 | +12.99（2000 / 36）|
| 0.5x | `nglab0_5x_input_fv` | **+4.95** | 18 | +4.95（2000 / 18）|
| 0.75x | `nglab0_75x_input_fv` | **+2.12** | 12 | +2.12（2000 / 12）|
| 1x | `nglab1x_v10_input` | **+1.96** | 9 | +1.96（2000 / 9）|
| 1.5x | `nglab1_5x_input_fv` | **+0.87** | 6 | +0.87（2000 / 6）|
| 2x | `nglab2x_input_v10_fv` | **+0.50** | 5 | +0.50（2000 / 5）|
| 2.5x | `nglab2_5x_input_fv_v2` | −0.03 | 4 | +0.69（3200 / 6）⚠️ |
| 3x | `nglab3x_input_fv_v2` | +0.11 | 3 | +1.94（3800 / 6）|
| 4x | `nglab4x_input_fv_v2` | −0.03 | 2-3 | +1.55（5000 / 6.2）|
| 5x | `nglab5x_input_fv` | −0.05 | 2 | −0.05（2000 / 2）|
| 6x | `nglab6x_input_fv` | −0.11 | 2 | −0.11（2000 / 2）|
| 8x | `nglab8x_input_fv` | +0.03 | 2 | +0.03（2000 / 2）|

**结论（用户假设成立）**：epoch shard 越大，gap@2000 单调变小——
0.25x→8x：+13 → +5 → +2 → +1.96 → +0.87 → +0.50 → ~0（≥2.5x 在 2000 步内
gap≈0，≥5x 只走 ~2 个 epoch、gap 尚未形成）。0.25–2x 段近似幂律
（log-log 斜率约 −1.5~−2）。横轴双尺度：`gap_vs_epochs.png` 按「已过 epoch 数」
缩放后可见 3x/4x 在 epoch 5-6 的 gap 反而高于 1.5x/2x → 大 shard 只是**延迟**了
gap 的出现（步数维度），并未消除（epoch 维度），支持「重播轮数 × 数据大小共同决定」。

**观测到的 epoch 边界**（fixed-val + freq-bin eval 每 10 步额外消耗 4 个 train batch，
故步数/epoch < 纯 rows/72）：0.5x≈120、1x≈240、2x≈450、2.5x=570/1130/1690/2240/2800、
3x=670/1340/2010/2670/3340、4x=890/1780/2670/3550/4440（步数/epoch ≈ 120/240/450/560/670/890，
比例 1:2:3.7:4.7:5.6:7.4 与 rows 比例 1:2:4:5:6:8 基本一致）。

### ⚠️ 2.5x 的 train 停滞（待 v3/s43 确认）

- `nglab2_5x_input_fv_v2`（train=1+2+64，3200 步）train loss 在 steps ~1200–2700
  （epoch 3-5）卡在 ~3.7–3.9 不动，直到 epoch 6（steps 2800–3200、lr→0.05）才掉到
  3.31；val（fixed shards 4..）同步回升 3.70→4.01。同规模的 3x/4x 同 epoch 处
  train 已到 2.0–2.3。
- 排除项：val 重叠（v1 同停滞）；table 爆炸（trigram RMS 0.134 @3200 vs 3x 0.135
  @3800 vs 4x 0.144 @5000，曲线平滑）。
- **最可能原因 = LR 调度混杂**：`get_lr_multiplier` 以 `progress=(step+1)/max_steps`
  计算，2.5x/3x/4x 是延长 run（3200/3800/5000），LR 升温/衰减都被拉长 →
  step 2000 时 lr_mult = 0.598/0.742/0.927，而 ≤2x 的 run 在 step 2000 已衰减到
  0.05。2.5x 的 warmdown 从 1120 开始，停滞区正好落在它的中高 LR 段；当 LR 衰减
  到 <0.3 后 train 才开始快速下降。因此 v2 的「final gap @~6 epoch」三组之间
  不可直接比，`gap@2000` 对 2.5x/3x/4x 也测在不同 LR 工作点上。
- 次要因素：2.5x 的 epoch = 1+2+shard3 前半（shard 64），与 3x 共享前 2 个 shard，
  数据构成差异待验证。

### 验证批（10:16 启动；v3 已于 11:35 完成，s43 运行中）

- `nglab2_5x_input_fv_v3` / `nglab3x_input_fv_v3` / `nglab4x_input_fv_v3`：
  同 v2 的 train/val 配置但 **max_steps=2000**（与主 sweep 完全相同的 LR 调度），
  得到公平的 gap@2000 —— **结果：+0.04 / +0.03 / −0.04**（train 3.49/3.29/3.30，
  val 3.53/3.32/3.26，epoch 4/3/3）≈ 0，与 v2 延长 run 的 ≈0 一致 → **主剂量曲线
  对 LR 调度稳健**（2.5x/3x/4x 在 2000 步内 gap 确实≈0，不是 LR 假象）。
  注：同 LR 调度下 2.5x train@2000 = 3.49（v2 是 3.80）——延长 run 的 LR 拉伸确实
  拖慢了 2.5x 的 train 下降，但 val 同步下降，gap 仍≈0。
- `nglab2_5x_input_fv_s43`：2.5x @3200、seed 43，检验停滞是否 seed/数据相关
  —— **结果（12:20 完成）：与 seed42 v2 几乎逐点重合**，epoch 2-5 train 同样停滞
  ~3.8–3.9（s43: 4.39/3.94/3.80/3.95 vs s42: 4.41/3.91/3.78/3.92），epoch 6 才掉到
  3.33，final gap **+0.74 @3200**（s42 +0.69）→ **停滞是「3200 步 LR 拉伸 × 2.5x 数据」
  的确定性现象，跨 seed 复现**；不是 val 重叠、不是 table 爆炸、不是 seed 噪声。
  结合 v3（同 LR 调度下 train@2000 = 3.49、无停滞），可归因于延长 run 的 warmdown
  拉伸（2.5x 从 step 1120 开始衰减、停滞区正好落在中高 LR 段），而 2000 步预算内
  （主剂量曲线）2.5x 无异常。
- 脚本：`code/cluster/run_verify_v3.sh`（⚠️ 该脚本未随迁移带入，仅在 ophis-gpu 远端存在；nohup，日志 `verify_v3.log`）。

### 图（已用 v2 数据重跑，10:11 同步回本地）
`docs/figs/epoch_scale/dose_response_gap2000.png`（gap@2000 vs shard size，log-x）、
`gap_vs_epochs.png`（gap vs 已过 epoch 数，横轴按 epoch 缩放）、`sweep_train_val_gap.png`。
脚本：`docs/plot_scripts/gen_shard_sweep_figs.py`。

### 结果（12 点全部完成，2026-08-07 凌晨→早上；2.5x/3x/4x 为 v2 重跑）

| size | run | gap@2000 | epoch@2000 | final gap（步数 / epoch）|
|---|---|---|---|---|
| 0.25x | `nglab0_25x_input_fv` | **+12.99** | 36 | +12.99（2000 / 36）|
| 0.5x | `nglab0_5x_input_fv` | **+4.95** | 18 | +4.95（2000 / 18）|
| 0.75x | `nglab0_75x_input_fv` | **+2.12** | 12 | +2.12（2000 / 12）|
| 1x | `nglab1x_v10_input` | **+1.96** | 9 | +1.96（2000 / 9）|
| 1.5x | `nglab1_5x_input_fv` | **+0.87** | 6 | +0.87（2000 / 6）|
| 2x | `nglab2x_input_v10_fv` | **+0.50** | 5 | +0.50（2000 / 5）|
| 2.5x | `nglab2_5x_input_fv_v2` | −0.03 | 4 | +0.69（3200 / 6）⚠️ |
| 3x | `nglab3x_input_fv_v2` | +0.11 | 3 | +1.94（3800 / 6）|
| 4x | `nglab4x_input_fv_v2` | −0.03 | 2-3 | +1.55（5000 / 6.2）|
| 5x | `nglab5x_input_fv` | −0.05 | 2 | −0.05（2000 / 2）|
| 6x | `nglab6x_input_fv` | −0.11 | 2 | −0.11（2000 / 2）|
| 8x | `nglab8x_input_fv` | +0.03 | 2 | +0.03（2000 / 2）|

**结论（用户假设成立）**：epoch shard 越大，gap@2000 单调变小——
0.25x→8x：+13 → +5 → +2 → +1.96 → +0.87 → +0.50 → ~0（≥2.5x 在 2000 步内
gap≈0，≥5x 只走 ~2 个 epoch、gap 尚未形成）。0.25–2x 段近似幂律
（log-log 斜率约 −1.5~−2）。横轴双尺度：`gap_vs_epochs.png` 按「已过 epoch 数」
缩放后可见 3x/4x 在 epoch 5-6 的 gap 反而高于 1.5x/2x → 大 shard 只是**延迟**了
gap 的出现（步数维度），并未消除（epoch 维度），支持「重播轮数 × 数据大小共同决定」。

**观测到的 epoch 边界**（fixed-val + freq-bin eval 每 10 步额外消耗 4 个 train batch，
故步数/epoch < 纯 rows/72）：0.5x≈120、1x≈240、2x≈450、2.5x=570/1130/1690/2240/2800、
3x=670/1340/2010/2670/3340、4x=890/1780/2670/3550/4440（步数/epoch ≈ 120/240/450/560/670/890，
比例 1:2:3.7:4.7:5.6:7.4 与 rows 比例 1:2:4:5:6:8 基本一致）。

### ⚠️ 2.5x 的 train 停滞（待 v3/s43 确认）

- `nglab2_5x_input_fv_v2`（train=1+2+64，3200 步）train loss 在 steps ~1200–2700
  （epoch 3-5）卡在 ~3.7–3.9 不动，直到 epoch 6（steps 2800–3200、lr→0.05）才掉到
  3.31；val（fixed shards 4..）同步回升 3.70→4.01。同规模的 3x/4x 同 epoch 处
  train 已到 2.0–2.3。
- 排除项：val 重叠（v1 同停滞）；table 爆炸（trigram RMS 0.134 @3200 vs 3x 0.135
  @3800 vs 4x 0.144 @5000，曲线平滑）。
- **最可能原因 = LR 调度混杂**：`get_lr_multiplier` 以 `progress=(step+1)/max_steps`
  计算，2.5x/3x/4x 是延长 run（3200/3800/5000），LR 升温/衰减都被拉长 →
  step 2000 时 lr_mult = 0.598/0.742/0.927，而 ≤2x 的 run 在 step 2000 已衰减到
  0.05。2.5x 的 warmdown 从 1120 开始，停滞区正好落在它的中高 LR 段；当 LR 衰减
  到 <0.3 后 train 才开始快速下降。因此 v2 的「final gap @~6 epoch」三组之间
  不可直接比，`gap@2000` 对 2.5x/3x/4x 也测在不同 LR 工作点上。
- 次要因素：2.5x 的 epoch = 1+2+shard3 前半（shard 64），与 3x 共享前 2 个 shard，
  数据构成差异待验证。

### 验证批（10:16 启动；v3 已于 11:35 完成，s43 运行中）

- `nglab2_5x_input_fv_v3` / `nglab3x_input_fv_v3` / `nglab4x_input_fv_v3`：
  同 v2 的 train/val 配置但 **max_steps=2000**（与主 sweep 完全相同的 LR 调度），
  得到公平的 gap@2000 —— **结果：+0.04 / +0.03 / −0.04**（train 3.49/3.29/3.30，
  val 3.53/3.32/3.26，epoch 4/3/3）≈ 0，与 v2 延长 run 的 ≈0 一致 → **主剂量曲线
  对 LR 调度稳健**（2.5x/3x/4x 在 2000 步内 gap 确实≈0，不是 LR 假象）。
  注：同 LR 调度下 2.5x train@2000 = 3.49（v2 是 3.80）——延长 run 的 LR 拉伸确实
  拖慢了 2.5x 的 train 下降，但 val 同步下降，gap 仍≈0。
- `nglab2_5x_input_fv_s43`：2.5x @3200、seed 43，检验停滞是否 seed/数据相关
  —— **结果（12:20 完成）：与 seed42 v2 几乎逐点重合**，epoch 2-5 train 同样停滞
  ~3.8–3.9（s43: 4.39/3.94/3.80/3.95 vs s42: 4.41/3.91/3.78/3.92），epoch 6 才掉到
  3.33，final gap **+0.74 @3200**（s42 +0.69）→ **停滞是「3200 步 LR 拉伸 × 2.5x 数据」
  的确定性现象，跨 seed 复现**；不是 val 重叠、不是 table 爆炸、不是 seed 噪声。
  结合 v3（同 LR 调度下 train@2000 = 3.49、无停滞），可归因于延长 run 的 warmdown
  拉伸（2.5x 从 step 1120 开始衰减、停滞区正好落在中高 LR 段），而 2000 步预算内
  （主剂量曲线）2.5x 无异常。
- 脚本：`code/cluster/run_verify_v3.sh`（⚠️ 该脚本未随迁移带入，仅在 ophis-gpu 远端存在；nohup，日志 `verify_v3.log`）。

### 图（已用 v2 数据重跑，10:11 同步回本地）
## 11. toy-model 台阶清晰度溯源（2026-08-07，郭绍阳 + 2 workers）

> 背景：主实验（1x/2x epoch）gap 曲线「不够清晰」（无 toy 那种每 epoch 台阶）；
> beta2 扫描（0.999/0.9999/0.99999/0.99/0.98）影响很小。toy5/t5 的 table beta
> 与主实验同为 `NGRAM_TABLE_BETAS=0.0,0.999`，但其台阶状极清晰——差异主因是
> **epoch 长度**（toy5 low: 2000 步 ≈ 29 epoch，~70 步/epoch；主实验 1x ~225、
> 2x ~450 步/epoch）。假设：台阶清晰度 ∝ 重放频率（epoch 越短越清晰），beta 影响次要。
> 两个 worker 并行验证：(A) toy 侧扫 `NGRAM_TABLE_BETAS` 看台阶是否随 beta 变化；
> (B) 真实模型侧扫「0.25x/0.5x epoch × beta2」网格，量化台阶清晰度。

### Setting（worker A：toy 侧 beta 扫描）
| 项 | 值 |
|---|---|
| 脚本 | 已迁入 `tasks/` 的 toy 数据生成脚本与 launcher（历史 toy 工作区）|
| 变体 | `NGRAM_TABLE_BETAS` = 0.0,0.999（基准）/ 0.0,0.99 / 0.0,0.9999 / 0.9,0.999 / 0.9,0.9999 |
| steps | 2000（low cache，~29 epoch）|
| 输出 | 每 epoch `headline_gap`（台阶）、`seen_gap` |

### Setting（worker B：真实模型短 epoch × beta2）
| 项 | 值 |
|---|---|
| 脚本 | `ngram-gap-lab/code/train.py` + `code/cluster/run_epoch_scale_v10.sh` 风格 |
| 变体 | epoch 0.25x/0.5x × beta2 {0.999, 0.99}（=4 run，2000 步，input 注入，seed42）|
| 对照 | §7 `nglab0_5x_input_fv`（0.5x·beta2=0.999 已有）|
| 输出 | train_log.jsonl（每 10 步 gap）+ table_norm.jsonl |

### 结果（Worker A：toy 侧 beta 扫描，已完成 2026-08-07 16:10，360-2）

- 5 个变体（0.0,0.999 基准 / 0.0,0.99 / 0.0,0.9999 / 0.9,0.999 / 0.9,0.9999），
  seed 42、low cache、2000 步（~29 epoch），全部 rc=0。
- 分析脚本：集群 `toy_analyze.py`（已同步本地新版 `tasks/` 分析脚本 +
  `toy_model.py` 相对路径 patch），exact-context headline_gap per-epoch。
- **结论：β 不改变 toy 的台阶形状** —— 5 条 per-epoch 曲线几乎重合，最终 gap
  7.43–7.89（基准 7.89），200/400/800 步处差异 ≤0.3 nats，无系统性顺序
  （0.9999 反而略低）。台阶清晰度由重放频率（epoch 长度）决定，与 table β 无关。

| betas | gap@200 | gap@400 | gap@800 | gap@1200 | gap@1600 | gap@2000 |
|---|---|---|---|---|---|---|
| 0.0,0.999（基准）| 0.407 | 1.965 | 6.123 | 7.950 | 7.884 | 7.894 |
| 0.0,0.99 | 0.424 | 2.391 | 5.149 | 7.659 | 7.849 | 7.737 |
| 0.0,0.9999 | 0.406 | 1.400 | 4.822 | 7.310 | 7.199 | 7.434 |
| 0.9,0.999 | 0.422 | 2.621 | 5.366 | 7.755 | 7.949 | 7.631 |
| 0.9,0.9999 | 0.436 | 2.251 | 5.287 | 7.538 | 7.893 | 7.705 |

> 注：绝对值与历史 ophis-gpu 值（t5_on_low final 6.79）不同，因新 `toy_analyze.py`
> 用 exact-context counts 重新量化 r，同一脚本内 5 个变体可比。step 级曲线显示
> toy 的 gap 在 epoch 内平滑增长（每 epoch 段内 drift +0.2~+1.3），台阶来自
> per-epoch 采样点（200/400/800...），与主实验的差异是 epoch 密度而非 β。

### 结果（Worker B：真实模型短 epoch × beta2，已完成 2026-08-07 17:35，360-2 GPU3/4）

- 2 个新 run（0.25x·b2=0.99、0.5x·b2=0.99）对照 §10 的 b2=0.999 参考（2000 步、
  input 注入、seed42、fixed-val），全部 rc=0，md5 核对通过。
- 数据：`data/runs/nglab025x_b2_099` / `nglab05x_b2_099`（已同步本地）。

| arm | b2 | gap@2000 | train@2000 | val@2000 | epochs | 台阶清晰度比（boundary/within）|
|---|---:|---:|---:|---:|---:|---:|
| 0.25x | 0.999（§10）| +12.991 | 0.376 | 13.367 | 36 | 7.57 |
| 0.25x | 0.99（新）| **+13.577** | 0.280 | 13.857 | 36 | 6.38 |
| 0.5x | 0.999（§10）| +4.952 | 1.776 | 6.728 | 18 | 4.12 |
| 0.5x | 0.99（新）| **+5.017** | 1.789 | 6.806 | 18 | 5.64 |

**结论（Worker B）**

1. **β2=0.99 在短 epoch 下仍只有微弱正效应**：0.25x +0.59（+4.5%）、0.5x +0.07（+1.4%），
   与 §9d 的 1x·RMS4x 结论（β2=0.99 +0.40，+8.5%）同方向但量级更小。
2. **β2 不改变台阶清晰度**：清晰度比在 0.25x 下 7.57→6.38、0.5x 下 4.12→5.64，
   方向不一致且幅度小（±1.5 以内）；而 epoch 长度本身的效应大得多
   （0.25x 清晰度比 6.4~7.6 vs 0.5x 4.1~5.6 vs 1x 5.5 vs 2x 6.6 混杂）。
   曲线仍以 epoch 内平滑上升为主（within/step 0.008~0.012），没有出现 toy 那种
   每 epoch 边界跳变——真实模型的台阶感弱，本质是「每 epoch 段内的连续过拟合」，
   与 β 无关。
3. **两 worker 合并结论：台阶清晰度 ∝ 重放频率（epoch 密度），table β（1 阶/2 阶）
   不改变台阶形态**。想让曲线更「台阶化」，只能缩短 epoch（0.25x 已是 36 epoch，
   gap@2000 最大 +13.6）；想让台阶变模糊，则拉长 epoch（1x/2x 已模糊）。

### 产物
- `docs/figs/theory/figs_v11_toy_beta_scan_per_epoch.svg` / `_step_level.svg`
  （同框 5 变体，per-epoch + step 级）
- `docs/figs/short_epoch_b2/short_epoch_b2_gap_v11.{svg,png}`（4 条 gap-step 曲线同框 +
  per-epoch mean gap 台阶视图）、`docs/figs/short_epoch_b2/staircase_shape_comparison.{svg,png}`
  （toy vs 真实模型归一化台阶形状）
- 脚本：`docs/plot_scripts/gen_short_epoch_b2_figs.py`；launcher：
  `code/cluster/run_epoch_short_b2.sh`（Worker B）、`tasks/` 中的 beta-scan launcher（Worker A）



`docs/figs/epoch_scale/dose_response_gap2000.png`（gap@2000 vs shard size，log-x +
幂律拟合）、`gap_vs_epochs.png`（gap vs 已过 epoch 数）、`sweep_train_val_gap.png`
（12 条全曲线）。脚本：`docs/plot_scripts/gen_shard_sweep_figs.py`（SWEEP 映射指向
v2；本地 `data/runs/<run_id>/train_log.jsonl` 已补齐 12 个 run）。



## 12. epoch 对齐批（同 epoch 数 × 同 LR-per-epoch 轨迹，2026-08-07 进行中）

> ⚠️ **勘误（2026-08-24）**：本节的**原版设计**是用 `--lr_schedule_epochs 6` 把 LR
> 锚定到 epoch（下表 Setting 的原始记录）。但**实际落盘的 `_fixed` 批**（
> `data/runs_fixed/nglab*_e6_fixed`）**并未使用 LR 锚定**——全部 `lr_schedule_epochs=0`、
> warmdown 0.65（与 §14 控制臂 / plan-4 统一口径一致）。且 `_fixed` 批实际只跑了
> **5 个 epoch**（非 6），steps = 5 × 每-epoch 步数（如 1x=1685=5×337）。
> 后续以 `_fixed` 批为准；下方表格保留原版 6.1-epoch 规划值作为历史记录。

> 背景：§10 的 step 对齐 sweep（gap@2000）显示「shard 越大 gap 越小」，
> 但大 shard 在 2000 步内只走了更少的 epoch（8x 仅 ~2 epoch），
> 「少重播」与「大 shard」混杂。用户提出：**对齐 epoch 数量**再看。
> 本批：所有 shard 大小都训到 ~6 个 epoch，且用 `--lr_schedule_epochs 6`
> 把 LR 锚定到 epoch（所有 run 共享同一条 LR-vs-epoch 轨迹，
> 排除 §10 ⚠️ 里发现的 LR 拉伸混杂）。

### Setting

| 项 | 值 |
|---|---|
| 注入点 / seed | input / 42（与 §10 主 sweep 完全一致）|
| LR | `--lr_schedule_epochs 6`（progress = epoch/6，warmdown 0.65；epoch 6→7 边界 lr=0.05）|
| steps | 每 run 跑到 ~6.1 epoch（epoch 7 开始后 ~60-130 步）|
| val / freq eval | v10，fixed-val |
| 集群 | 0.25x–1.5x：360-2（14:26–15:47）；2x–3x：360-1（16:17–17:12，360-2 首跑 OOM 后重跑）；4x–8x：ophis-gpu（13:50 启动，进行中）|

| size | run | train shards | val shards | steps（目标 ~6.1 ep）|
|---|---|---|---|---|
| 0.25x | `nglab0_25x_e6` | 62 | 2..10,6542 | 420 |
| 0.5x | `nglab0_5x_e6` | 60 | 2..10,6542 | 780 |
| 0.75x | `nglab0_75x_e6` | 63 | 2..10,6542 | 1080 |
| 1x | `nglab1x_e6` | 1 | 2..10,6542 | 1440 |
| 1.5x | `nglab1_5x_e6` | 1,61 | 3..10,6542 | 2100 |
| 2x | `nglab2x_e6` | 1,2 | 3..10,6542 | 2800 |
| 2.5x | `nglab2_5x_e6` | 1,2,64 | 4..10,6542 | 3500 |
| 3x | `nglab3x_e6` | 1,2,3 | 4..10,6542 | 4200 |
| 4x | `nglab4x_e6` | 1..4 | 5..10,6542 | 5490 |
| 5x | `nglab5x_e6` | 1..5 | 6..10,6542 | 6900 |
| 6x | `nglab6x_e6` | 1..6 | 7..10,6542 | 8260 |
| 8x | `nglab8x_e6` | 1..8 | 9,10,6542 | 10960 |

### 结果（0.25x–3x 已完成；4x–8x 进行中）

gap 在「6 个完整 pass 后」（epoch 7 首个 eval，lr=0.05）与「pass 6 内 mean/peak」：

| size | gap@6pass (bnd) | pass6 mean | pass6 peak | 到达步数 | gap@2000（§10 对照）|
|---|---:|---:|---:|---:|---:|
| 0.25x | **+1.094** | +1.083 | +1.206 | 350 | +12.991 |
| 0.5x | **+1.419** | +1.566 | +1.616 | 680 | +4.952 |
| 0.75x | **+0.845** | +0.732 | +0.841 | 1020 | +2.123 |
| 1x | **+1.911** | +1.918 | +2.147 | 1360 | +1.961 |
| 1.5x | **+0.914** | +0.848 | +0.977 | 2030 | +0.870 |
| 2x | **+0.800** | +0.711 | +0.894 | 2690 | +0.502 |
| 2.5x | **+0.925** | +0.557 | +0.927 | 3360 | −0.027 |
| 3x | **+2.141** | +1.712 | +2.117 | 4010 | +0.113 |
| 4x | 进行中（step 4580/5490，epoch 6，gap +1.10 @17:16）| | | | −0.031 |
| 5x | 进行中（step 4350/6900，epoch 4，gap −0.04）| | | | −0.047 |
| 6x | 进行中（step 4370/8260，epoch 4，gap −0.12）| | | | −0.110 |
| 8x | 进行中（step 4340/10960，epoch 3，gap −0.02）| | | | +0.027 |

**初步结论（待 4x–8x 补全）**：对齐 epoch 数后，§10 的「shard 越大 gap 越小」
**单调关系消失**——0.25x→3x 的 gap@6pass 在 +0.80~+2.14 之间非单调波动
（1x=+1.91、3x=+2.14 偏高，0.75x/2x 偏低 ~0.8），不再随 shard 大小单调下降。
step 对齐下的单调递减主要来自**大 shard 看到的重播轮数更少**（8x 在 2000 步内
只有 ~2 epoch），而非 shard 大小本身；在「同重播轮数 + 同 LR-per-epoch」下，
每个 epoch 的重放 gap 大致相当（0.8–2.1），与数据量 0.25x–3x 无系统关系。
> 注意：0.25x 在 6 pass 时 gap 只有 +1.1（vs 2000 步 36 pass 时的 +13.0），
> 说明小 shard 的巨额 gap 是「重播次数」累积出来的，而非单次重播更强。

图：`docs/figs/epoch_scale/gap_vs_shard_size_epoch_aligned.png`
（epoch 对齐 vs step 对齐双曲线）、`gap_vs_epoch_curves.png`（gap vs epoch 数，
各 shard 轨迹）、`epoch_aligned_train_val_gap.png`。
脚本：`docs/plot_scripts/gen_epoch_aligned_figs.py`。


---

## 13. toy 严格 Zipf 分布 · per-bucket gap 双对数（2026-08-07，planned）

> 背景：真实语料近似 Zipf，per-bin gap–frequency 双对数拟合已较好（bigram R²≈0.96、
> trigram R²≈0.81，见 `docs/figs/theory/fig_zipf_gap_analysis.png`）；toy 当前的频次分布是
> **anti-Zipf 设计**（N_r∝1/r，每桶 token 数相等，a≈−0.93，R²=0.99）。
> 用户提出：把 toy 的 ngram 分布筛选成**严格 Zipf**（N_r∝1/r²，经典 rank 指数 1），
> 再看 gap–ngram 双对数线性是否变好。
> 理论预判（原历史理论笔记，⚠️ 该文件已不存在；相关推导见 `docs/notes/theory/`）：
> per-bucket gap g(r) 由训练动力学+val 协议决定，**与总体分布 N_r 可分离**——
> 严格 Zipf 只改权重/累计曲线，不改 g(r) 形状。本批跑 3 个 seed 做经验验证。

### Setting（与 t5_low 完全一致，仅频次分布不同）

| 项 | 值 |
|---|---|
| 分布 | `mode=zipf`：N_r = round(C/r²)，r=1..199，Σ=32768 keys，整数精确计数（counts_exact）|
| 协议 | coincidental r<16 / shared r≥16（同 low）|
| 训练 | 2000 步 · input 注入 · seed 42/43/44 · β=(0.0,0.999) · RMSProp(table)+AdamW(backbone) |
| 数据 | `toy5_data_gen.py --mode zipf` + `toy_prep.py --vocab 2048` → cache `t5_zipf` |
| 评估 | exact-context per-r gap（r=1,2,4,8,16,...）· val 每 10 步 |
| 集群 | 待定（360-2 当前不可达；ophis-gpu 需先同步 toy5 代码+建 cache）|

### 预期结果

1. per-bucket g(r) 与 t5_low 重合（可分离性验证）：双对数 R² 仍 ~0.2–0.4，拐点在 r=16。
2. 严格 Zipf 加权后的累计/每 token 贡献曲线 → 干净幂律（重加权分析已示：
   token 加权贡献 slope≈−1.09，R²≈0.97）。
3. 高频桶（r≥32）样本少（N_32≈20, N_64≈5）→ per-bucket 统计在高频端变噪，
   与理论「纯 Zipf 下高 r 桶统计崩掉」一致。

### 结果（已完成 2026-08-07，360-2 GPU1/2/5）

| run | final gap@2000 | rho_logr | log-log slope | log-log R² | per-r gap (1,2,4,16,64) |
|---|---:|---:|---:|---:|---|
| `t5z_zipf_s42` | +7.012 | −0.879 | −0.154 | 0.784 | 8.56 / 9.40 / 8.93 / 5.27 / 5.32 |
| `t5z_zipf_s43` | +7.956 | −0.861 | −0.131 | 0.765 | 8.97 / 9.91 / 9.73 / 6.12 / 6.00 |
| `t5z_zipf_s44` | +7.558 | −0.851 | −0.123 | 0.754 | 8.24 / 9.24 / 9.04 / 5.88 / 5.67 |
| `t5b_beta_000_999_low`（anti-Zipf 对照）| +7.894 | −0.364 | −0.099（全 6 点）/ −0.112（去 r=8 离群）| 0.235 / 0.821 | 8.80 / 9.54 / 9.05 /(8:14.30)/ 6.43 / 6.15 |

**关键观察 / 结论**

1. **严格 Zipf 构建成功**：`mode=zipf` 生成 N_r≈C/r²（r=1:19921, r=2:4980, r=4:1245,
   r=8:311, r=16:78，比值 4:1），Σ=32768 keys，`counts_exact=True`；3 seeds（42/43/44）
   全部 rc=0，cache `t5_zipf` 在 360-2 构建（548k tokens/epoch，2000 步 ≈ 34 epoch）。
2. **per-bucket g(r) 与分布可分离（经验验证成立）**：严格 Zipf 的 per-r gap
   （8.59/9.52/9.23/5.76/5.66）与 anti-Zipf 的 t5b（8.80/9.54/9.05/6.43/6.15）
   几乎重合（r=1..4 差 <0.3，r=16/64 差 ~0.5–0.7），同协议下换分布 g(r) 不变。
3. **双对数线性没有变好**：zipf R²≈0.75–0.78 vs t5b 去 r=8 离群后 R²≈0.82，
   斜率都只有 ≈ −0.12~−0.15（不是干净幂律 −1）；t5b 全点 R²=0.24 的“坏”主要是
   r=8 离群（14.30，疑似 low-cache probe 噪声），不是分布问题。
4. 旧 kink 设计（t5_on_low）的陡斜率 −1.9 / R²=0.80 来自 r≥16 shared key gap→0 的
   硬拐点，与 Zipf 无关。
5. 与真实模型对照：真实语料（近似 Zipf）per-bin 双对数 R²=0.81–0.96，已较好；
   toy 的偏差来自协议（r≥16 拐点 + 低频段 g 微升），分布不是原因。

### 产物

- 代码：已迁入 `tasks/` 的 toy 数据生成脚本（`mode=zipf`）；launcher：
  `tasks/` 中的 zipf launcher。
- 数据：历史 `t5z_zipf_s{42,43,44}` run 的 metadata 已纳入本实验记录。
- 图：`docs/figs/theory/fig_zipf_experiment.{png,svg}`（左：per-bucket g(r) 重合；
  右：N_r 分布 −1 vs −2）；`docs/figs/theory/fig_zipf_gap_analysis.{png,svg}`（重加权分析）。
- 脚本：`docs/plot_scripts/gen_zipf_experiment_figs.py`、`analyze_zipf_gap.py`。

## 14. 干净 vanilla 复现（2026-08-23，input 主臂 + nogram 对照）

### Setting
极简 SSOT（agents.md §1）标准配置，无任何偏离：

| 项 | 值 |
|---|---|
| backbone | vanilla nanoGPT 8L·6H·768D，vocab 8192，seq 2048 |
| n-gram | bigram+trigram，`input` 注入，table 1M（默认未动） |
| 优化器 | table RMSProp(0.0,0.99)，backbone AdamW(0.8,0.95) lr 0.004 wd 0.1 |
| 数据 | shard 1 train（24264 rows ≈ 337 steps/epoch），shard 2 val，fixed 顺序 |
| 步数 / 评测 | 1000 步（≈3 epoch），seed 42，val 每 10 步 fixed batches（v10 口径） |

### 目的
在 `code/train.py` 干净复现上重跑标准极简设置，确认 gap 现象可复现
（历史参考：`nglab1x_v10_input` 1.931@2000，`nglab1x_v10_nogram` 0.231@2000）。

### 结果（已完成 2026-08-23）

| run | gap@999 | gap@1000 | train@1000 | val@1000 | 说明 |
|---|---:|---:|---:|---:|---|
| `vanilla_input_1000_seed42` | +0.803 | **+0.858** | 3.608 | 4.466 | input 注入主臂，epoch 2 起 fork |
| `vanilla_nogram_1000_seed42` | +0.022 | **+0.038** | 5.305 | 5.342 | 无 n-gram 对照，全程无 fork |

**关键观察 / 结论**

1. **gap 现象在干净 vanilla 上成功复现**：input 臂 gap 从 epoch 2（step 337 边界）起单调 fork，
   step 430 时 +0.07 → step 1000 时 +0.86；nogram 对照全程 ±0.04 内波动。
2. **gap 的来源确认为 n-gram 表**：input 臂 train 压到 3.61（表记住训练 token 的 over-encoding），
   val 只降到 4.47；nogram 臂 train/val 同步停在 5.3 附近（纯 backbone 泛化）。
3. 与历史口径一致（`nglab1x_v10_input` 1.931@2000 / `nglab1x_v10_nogram` 0.231@2000），
   1000 步的 fork 幅度约为 2000 步的一半，趋势吻合。

### 产物
- `data/runs_fixed/vanilla_input_1000_seed42_fixed/`（train_log.jsonl + summary.json）
- `data/runs_fixed/vanilla_nogram_1000_seed42_fixed/`
- 训练代码 `code/train.py`（未改动），数据生成 `code/prepare_data.py`（shard 1/2 现生成）

## 15. P1/P2 因果干预 · 极简 setting 重跑（2026-08-24）

### 目的
复现 `agents.md` §6.3「废弃结论，保留问题」队列的四条因果结论（原 current-shell 数字），
在极简 setting（vanilla nanoGPT + input 注入 + table 1M + RMSProp 无动量）下重跑。
旧结论（DEPRECATED SETTING，见 `docs/_archive/docs/p12-causal-results.md`）：
table 回滚 −89% / readout 屏蔽 −89% / 冻结 table −49% / 冻结 backbone −54%。

### Setting

| 项 | 值 |
|---|---|
| backbone / n-gram | 同 §14（8L·6H·768D，bigram+trigram input 注入，table 1M） |
| 优化器 / 数据 / 步数 | 同 §14（RMSProp+AdamW，shard1 train / shard2 val，1000 步，seed42，v10 fixed-val） |
| 干预触发点 | epoch 边界（`--intervention_epoch`，0-indexed：1 = e1→e2 边界 ~step337，2 = e2→e3 边界 ~step674） |
| 控制臂 | 复用 `vanilla_input_1000_seed42`（+0.858 @1000），不重跑 |

### 干预臂矩阵

| 臂 | 干预 | 触发 | 复现旧结论 |
|---|---|---|---|
| `nglab1x_input_reset_e2` | 全 table 行回滚 init | e2 边界 | p1_reset_all_e2（−89%）|
| `nglab1x_input_reset_e1` | 全 table 行回滚 init | e1 边界 | p1_reset_all_e1（−13%，对照）|
| `nglab1x_input_mask_e1` | 屏蔽 bigram/trigram readout | e1 边界 | p2_readout_mask_e1（−89%）|
| `nglab1x_input_freeze_table_e1` | 冻结 table（保留 e1 内容）| e1 边界 | p2_freeze_table_e1（−49%）|
| `nglab1x_input_freeze_backbone_e1` | 冻结 backbone（仅 table 更新）| e1 边界 | p2_table_gate_only_e1（−54%）|

### 状态（已完成 2026-08-24）

| 臂 | 状态 | final gap@1000 | train@1000 | val@1000 | vs 控制 |
|---|---|---|---:|---:|---:|---|
| 控制 `vanilla_input_1000_seed42` | ✅ done | +0.858 | 3.608 | 4.466 | — |
| `nglab1x_input_reset_e2` | ✅ done | **+0.054** | 4.014 | 4.068 | **−94%** |
| `nglab1x_input_reset_e1` | ✅ done | +0.351 | 4.004 | 4.355 | −59% |
| `nglab1x_input_mask_e1` | ✅ done | **+0.058** | 4.269 | 4.327 | **−93%** |
| `nglab1x_input_freeze_table_e1` | ✅ done | +0.601 | 3.851 | 4.453 | −30% |
| `nglab1x_input_freeze_backbone_e1` | ✅ done | +0.780 | 4.159 | 4.939 | −9% |

**关键观察 / 结论**

1. **两个 −89% 级关键干预在极简 setting 上复现**：
   - e2 边界全 table 回滚（`reset_e2`）：gap 0.858 → 0.054（**−94%**，旧 −89%）。
     e1/e2 两 epoch 累积的行内容是 e3 大 gap 的必要条件。
   - e1 边界屏蔽 readout（`mask_e1`）：gap 0.858 → 0.058（**−93%**，旧 −89%）。
     n-gram readout 通道是 gap 的必要传导口。
2. **回滚时机剂量**：e1 回滚（−59%）比 e2 回滚（−94%）弱——e1 擦掉后 e2 还能重写，
   到 e3 时行历史部分恢复；e2 擦掉后只剩 e3 一个 epoch 重写，恢复不了。与旧的
   「e1 −13% / e2 −89%」方向一致（本批 e1 干预更大，因极简 setting 的 e1 行写入更强）。
3. **table write vs backbone 各贡献一半的旧结论未完全复现**：本批 freeze_table −30%、
   freeze_backbone −9%，backbone 冻结影响远小于旧的 −54%。即极简 setting 下 gap 更依赖
   table 持续写入 + backbone 放大，backbone 本身的训练动态贡献较小（旧 current-shell 有
   gate/reader 等额外可训练放大器，占一半）。
4. 与 §14 控制臂串起来：**gap 的产生与传导完全依赖 n-gram 表**（回滚/屏蔽 → 塌缩到
   nogram 对照量级 0.04~0.06），主干结论稳定跨 backbone 架构。

### 产物
- 干预实现：`code/train.py`（新增 `--intervention` / `--intervention_epoch` / `--table_mult`）
- launcher：`code/cluster/run_causal_minimal.sh`
- 集群数据：`data/runs_fixed/nglab1x_input_{reset_e2,reset_e1,mask_e1,freeze_table_e1,freeze_backbone_e1}_fixed/`

## 16. bf16 精度验证 + 提速（2026-08-24）

### 目的
确认把全 fp32 前向切到 bf16（`torch.autocast`，权重/优化器仍 fp32）不会改变 gap 现象，
同时量化提速幅度，作为后续实验的默认计算精度。

### 测速（H200，batch 72×2048，28.8B 全模型，单卡空闲）

| 配置 | train step | 相对 fp32 |
|---|---:|---:|
| fp32 | ~2.76 s | 1.0x |
| bf16（autocast） | ~0.48–0.53 s | **~5.3x** |
| bf16 + torch.compile | ~0.50 s（GPU 共租下） | ~5.5x |

- fp8 不可行：`torch.autocast(dtype=float8_e4m3fn)` 在 `nn.Linear` addmm 上不支持，
  需专门 `_scaled_mm` 工程，未采用。
- `--dtype {fp32,bf16,fp8}` + `--compile` 开关已加入 `code/train.py`；
  标准 launcher `run_causal_minimal.sh` 默认 `bf16` + `--compile`（可通过
  `NGLAB_DTYPE` / `NGLAB_COMPILE` 覆盖），并为每臂隔离 `TORCHINDUCTOR_CACHE_DIR`。

### 同超参精度对照（关键）

| 项 | fp32 `vanilla_input_1000_seed42` | bf16 `..._bf16_samehp` |
|---|---:|---:|
| 表优化器 | RMSProp β₂=0.999, lr_scale=1.0 | 同左（完全一致） |
| train@1000 | 3.608 | 3.580 |
| val@1000 | 4.466 | 4.405 |
| **gap@1000** | **+0.858** | **+0.825** |
| gap 曲线 | 见下 | 逐点重合（10/340/670/1000: +0.009/+0.021/+0.361/+0.858 ↔ +0.008/+0.049/+0.374/+0.825）|

**结论：bf16 在相同超参下逐点复现 fp32 曲线（loss 差 <0.1，final gap 0.858 vs 0.825），
且提速 ~5.3x。后续正式实验默认 bf16。**

> 注意：先前 `vanilla_input_1000_seed42_bf16`（gap +1.661）用了不同超参
> （β₂=0.99, lr_scale=2.0），非精度差异，勿与其对比。

### 产物
- 图：`docs/figs/fig_fp32_vs_bf16_samehp.png`
- 代码：`code/train.py`（`--dtype` / `--compile`）、`code/cluster/run_causal_minimal.sh`（默认 bf16+compile）
- 集群数据：`data/runs/vanilla_input_1000_seed42_bf16_samehp/`（+0.825）

## 17. S1 三轴 scaling 验证（2026-08-24，plan-5）

### 目的
在唯一极简 setting 下正式验证三条 scaling 曲线：epoch 长度 L、exact context
frequency f、table size（1M 逻辑地址只向下）。计划：
`docs/plans/plan-5-s1-three-axis-handoff.md`。

### 登记（planned → running → done；seed 42 首轮）

| run_id 前缀 | 轴 | setting | 状态 |
|---|---|---|---|
| `basic_*` | 基础 QC 锚点（7 run，seed 42） | 25 步 cadence + bf16（历史 run 含 compile） | 🟡 历史 QC；不属于当前标准证据 |
| `bb_safety_L1_nogram_5000` | backbone safety（L1 no-ngram 5000 步） | **旧 cadence**（50 步 + fp32 无 compile） | ✅ done（360-2 GPU7，**final fixed gap +16.66 @5000**） |
| `ep_{L}_{arm}_fs` | epoch · fixed-step | L1-L4 × 4 arms × 1000 步，step-anchored LR，10 步 cadence，历史 bf16+compile | 🟡 历史 16/16；当前 no-compile 待重跑 |
| `ep_{L}_{arm}_fe` | epoch · fixed-epoch | 6 完整 epoch（L1=252/L2=504/L3=1008/L4=2022 步），epoch-anchored LR，历史 bf16+compile | 🟡 历史 16/16；当前 no-compile 待重跑 |
| `tbl_{TM}_{arm}` | table · L4 | 23 measured sizes；原始 21 个 dense run 每 10 步，另 48 个 sparse 加密 run 只在最终步监测；1000 步，历史 bf16+compile | 🟡 历史 69/69；当前 no-compile 待重跑 |
| `freq_{arm}_{fs/fe}` | frequency 轴专用 | L4 + 1M table × 4 arms，exact-freq 跟随 validation 步点 | 🟡 历史 8/8；当前 no-compile 待重跑 |

### 关键口径决策（用户 2026-08-24 拍板）

1. **L4 = 337 batches/epoch** = 完整 shard 1（24,264 chunks / 72）。L1/L2/L3 为
   嵌套前缀 42/84/168。此前 plan/launcher 用 L4=336（42 的整数倍），已废弃。
2. **普通网格不计算 exact-frequency / freq-bin 诊断**（不传 `--freq_index`），
   只算在线 train/val loss + online gap。fixed train probe 仅作为显式诊断；
   频率轴单独跑一小批
   run（L4 + 1M table × 4 module 臂，exact-freq 跟随 validation 步点）。
3. **no-ngram baseline 重跑当前标准**（10 步 cadence + bf16 不 compile，每个 L
   一个）。`bb_safety_L1_nogram_5000`（50 步 + fp32）只能作长训 backbone gap
   量级参考，不能作为正式 grid 的 no-ngram baseline。
4. 当前标准完整曲线的 validation/table norm/frequency 每 10 步；只需曲线
   可用每 50 步；只需末端结果使用 `--val_steps 1000`，frequency 观测严格
   跟随这些 validation 步点；table 加密取点使用 sparse 模式，只在最终 step
   1000 监测。

### seed 42 正式网格回填（2026-08-24）

 - 结果目录：`data/runs_scaling/<run_id>_fixed/`，共 **109/109** 个正式 run；
- 所有历史 run 的 `summary.json` 均满足当时 contract：bf16、`torch_compile=true`、
  RMSProp `(0.0,0.99)`、`table_lr_scale=2.0`；dense run 为 10 步
  validation/probe cadence，48 个 sparse table run 只在最终 step 1000 触发；
- 109 个正式 fixed train probe SHA256 全部为 `38d1254a827759d6`；该 probe
  现只作为 exposure-contaminated 诊断，主 gap 不再读取它；
- JSON/JSONL 产物无 NaN、坏行或缺失；table run 均有
  `table_occupancy.json`；
- 三轴图和摘要位于 `docs/appendices/s1_scaling_three_axis/figs/`；
- frequency 轴的探索性两因素拟合和逐项排除 manifest 位于
  `figs/fit_manifest.json`；
- **尚未完成**：seed 43/44 复现、跨 seed uncertainty、frequency 的
  epoch-dependent fit，因此不把 seed 42 结果写成已确认的 scaling 定律。

### 17.2 三 seed 复现与 H1–H4 检验（2026-08-25 回填）

历史 compile 波次已完成 seed 43/44 三 seed 复现；这些结果仅作探索性数学审计：

| 批次 | run_id 后缀 | run 数 | launcher | 状态 |
|---|---|---:|---|---|
| epoch fs/fe | `_s43` / `_s44` | 32×2 | `run_scaling_epoch_full.sh`（`SEED=43/44`，dense 10 步） | ✅ done |
| table 加密 | `_s43` / `_s44` | 36×2（12 mult × 3 module） | `run_scaling_table_full.sh`（sparse 仅最终步） | ✅ done |
| frequency fs/fe | `_s43` / `_s44` | 8×2 | `run_scaling_frequency_axis.sh`（exact-freq 每 10 步） | ✅ done |

- 历史波次合计 **261 个 `_fixed`**（s42:109，s43/s44 各 76），通过当时
  contract / NaN / probe-hash QC，但不满足当前 no-compile 标准；
- 修复了 `train.py` 中 exact-freq 与 `--fixed_train_probe 0` 的解耦问题
  （有 `freq_index` 时始终建 `exact_freq_log`，train 侧用独立诊断迭代器抓
  4 个固定 batch，不消费训练流），历史 frequency 轴生成了
  `exact_freq_loss.jsonl`；其中 seed 42 及部分复现 run 的 exact-frequency
  cadence 为 100 步，部分后续复现 run 为 10 步，不能把它们统一写成当前
  10 步标准；
- 代码已 md5 核对同步到 360-2（train.py `5beca0cd…`）；集群单测 14/14 通过。

**H1–H4 判定**（全部 online gap 主口径，详见附录报告 §7）：

| 猜想 | 判定 |
|---|---|
| H1 两因素频率律 | β seed-stable（四组 cv 4–13%）；A/c/γ identifiability-limited（cv 33–141%），不报告绝对值 |
| H2 epoch 对齐律 | ΔG 方向 seed-stable（24/24 同号为正）；幅度 fixed-step seed-sensitive（L1_both cv>50%）、fixed-epoch 稳（L4_trigram 5.90/5.72/5.56，cv≈2%） |
| H3 table saturation | 有限窗口内总体上升；未解析稳定饱和平台，也不能写成全区间幂律；扩展 table 轴需在 no-compile 标准下重跑 |
| H4 模块交互 | I 显著非零且 seed-sensitive（fs 下变号，fe L4 强负；table mult≥48 剧烈变号），**不允许 bigram+trigram 合并单公式**，both 仅作对照 |

产物：`figs/epoch_final_gap.csv`（96 行）、`epoch_deltaG_fs_multiseed.png`、
`table_summary.csv`（141 行）、`fit_manifest.json`（12 个三 seed 拟合）。

### bb_safety 最终结果（2026-08-24 回填）

`bb_safety_L1_nogram_5000`（L1=42 b/ep，5000 步，no-ngram，seed 42）已完成：

| 量 | 值 |
|---|---:|
| 最终 fixed train loss | 0.0065 |
| 最终 fixed val loss | 16.666 |
| **最终 fixed gap** | **+16.66** |

- **含义**：长训（5000 步）no-ngram backbone 自身就会产生巨大 gap（train 接近
  0 而 val 16.7）—— 1000 步时 gap ≈ 0 不代表 5000 步仍为 0。这与
  `bb_safety` 的早先快照趋势一致（4000 步 +13.24）。
- **口径警告**：该 run 为旧 cadence（50 步）+ fp32 无 compile，仅作量级参考。
  正式 full grid 的 no-ngram baseline 使用 10 步 cadence + bf16、不 compile；
  S1 主 gap 统一读取 online train/val，fixed probe 只作诊断。
- **影响**：no-ngram 对照必须在每个 L、每个对齐下重跑，不能假设 backbone
  gap 恒为零；`ΔG = G_module − G_no-ngram` 的修正口径因此仍然必要。

### 说明
- 数据源：`data/runs_scaling/<run_id>_fixed/`（新 namespace）。
- 代码：`tasks/s1_scaling_three_axis/`（train.py / ngram_freq.py /
  table_occupancy.py / launchers / analysis）。
- 每个 run 的 summary.json 含 `table_betas=[0.0,0.99]`、`table_lr_scale`、
  `compute_dtype`、`torch_compile`、`fixed_train_probe_sha256`、`epoch_batches`、
  `exact_freq_eval_interval`。S1 正式 run 的
  `fixed_train_probe_batches=4`；scaling 分析直接读取 `train_log.jsonl` 的
  online gap。

### 17.1 口径修订（2026-08-24）

- **主 gap**：`train_log.jsonl` 的 `val_loss − train_loss`，其中 train loss
  是当前在线训练 batch，val 是同一步的 fixed validation。
- **fixed train probe**：只用于 exposure/训练进度诊断，不作为 scaling gap、
  epoch-1 gap 或模块比较的证据；`uniform` 采样也不能替代 online loss。
- 已同步：`agents.md` §1.6、S1 README、epoch/table/frequency 分析脚本，以及
  标准 scaling launcher 的 `--fixed_train_probe 0` 和 no-compile contract。

## 18. 训练提速工程：freq-bin eval 瓶颈 + `--val_steps`（2026-08-24）

### 目的
用户报告"以前 7 分钟 1000 步，现在一次实验常要 20+ 分钟"。逐项量化
fp32→bf16、torch.compile、freq-bin eval 三类因素的耗时贡献，并引入
`--val_steps` 允许只测特定步的 gap，作为 §1.4 纪律的补充手段。

### 决定性对照（H200，batch 72×2048，28.8B 全模型，单卡空闲）

| 配置 | train step | 说明 |
|---|---:|---|
| fp32（无 freq） | ~1.91 s | 旧 baseline |
| bf16（无 freq） | **0.32–0.37 s** | ~5.4x，bf16 提速真实有效 |
| bf16 + torch.compile | **1.20 s** | **负优化，慢 3.5x**（见下） |
| bf16 + freq eval（每 10 步） | ~2.09 s | freq eval 是最大瓶颈 |
| bf16 + `--val_steps`（30 步内 2 次完整 eval） | ~0.58 s | 只测末端时很快 |

结论：bf16 本身加速 ~5.4x 有效；当前"跑一次 20+ 分钟"的主因是
**freq-bin eval**（每 10 步一次，每次 ~13s = 8 次完整 forward + 全量 CE，
logits 72×2048×8192 ≈ 12 亿元素），不是 bf16 没生效。

### torch.compile 负优化结论
- bf16 + compile 反变慢 3.5x（0.34s → 1.2s/步）。
- 疑似原因：`NanoGPT` 的 `bigram_ves` / `trigram_ves` 字典 + 索引逻辑导致大量
  graph break，inductor 生成低效代码且 32 个 compile worker 占用资源。
- **决定：默认不 `torch.compile`**。已从所有 launcher 移除 `--compile`，
  agents.md §1.4 更新为"bf16 不 compile"。

### `--val_steps` 功能
- 新增 CLI：`--val_steps "1000"` —— 只在指定步做 val + freq eval；若显式启用
  fixed train probe，则 probe 也跟随该步点，
  训练全程不打断（val 步点由 `NGRAM5_PROBE_STEPS` 语义自动对齐）。
- summary.json 记录 `val_steps` 字段；freq eval 跟随 val_steps 对齐。
- smoke 验证：30 步 val_steps=10,30 → 0.58s/步，freq 自动对齐。
- 用途：只关心末端 gap 的实验（如因果干预、bf16 对照）用 `--val_steps 1000`
  可大幅省时；关心曲线的实验仍用默认 interval 10。
- agents.md §1.4 纪律已更新：允许按实验需要调整 eval 步点。

### freq 相关优化
- `hit_count_tensor` 从 `np.vectorize` + dict 逐元素查改成 `np.searchsorted`
  （预排序 keys）：速度 0.381s → 0.235s，正确性验证 bigram/trigram max diff = 0。
- 注意：该优化只解决 lookup 部分；freq eval 慢的主因是 8 次 forward + 大 CE，
  真正杠杆是降 freq 频率或 `--val_steps`。

### 产物
- 代码：`code/train.py`、`tasks/s1_scaling_three_axis/code/train.py`（`--val_steps`）、
  `code/ngram_freq.py`（searchsorted）
- 图：`docs/figs/fig_bf16_vs_fp32_accel.png`（如有）
- 纪律：`agents.md` §1.4（bf16 不 compile；允许调整 eval 步点）

---

## 19. 自然语言 5gram（order=5）· 极简 setting 重跑（2026-08-24）

### 背景
历史上 `ngram5_freq_gap` 包实际只跑过 trigram（`--order 3`），order=5 的自然语言
5gram 从未用极简 setting 执行过。OPHIS 旧库只有 order=5 的 smoke（`run_contract_20260806-115854.json`：
`"order": 5`，但 `loader_selection.smoke: true`，train_docs=200，25600 tokens）。本次
用 `code/make_ngram_blocks.py` + `ngram5_freq_gap/trainer.py` 在极简 setting 下重跑。

### 数据（`data/ngram5_minimal_order5/`，新生成）
| 项 | 值 |
|---|---|
| 来源 | `data/tokenized/shard_00001.bin`（train shard 1，全局语料，49.7M tokens）|
| train shards | 1 |
| val shards | 2,3,4,5,6,7,8,9,10,6542（与 train 不重叠）|
| order | 5（5-gram context）|
| block_len | 7（`[c0..c4, next, SEP]`）|
| train_blocks | 49,716,931 |
| val_blocks | 489,350,326 |
| distinct 5-gram contexts | **43,039,820**（train epoch 全量）|
| 生成器 | `code/make_ngram_blocks.py`（滑窗，每事件一份）|

### Setting（全部对齐 agents.md §1 极简基线）
- 模型：vanilla nanoGPT 8L·6H·768D，vocab 8192，seq 2048，learned abs，LayerNorm，tied
- 表：1M hash table（524,288 行 × 2 hash），RMSProp，`NGRAM_VE_BETAS=(0.0, 0.99)`（trainer 硬编码）
- 骨干：AdamW `(0.8, 0.95)`，lr 0.004，weight_decay 0.1
- table LR scale = **2.0**（表实际 lr 0.008），与主线极简 setting 一致
- 注入：input（wte），`NANOGPT_NGRAM_INJECTION_POSITION=input`
- batch 72 × 2048 = 147,456 tokens；2000 步；seed 42；bf16，**不 compile**（§18 结论）
- val：fixed batches，interval 10；freq probe：exact_context，edges `0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001`
- `NGRAM5_TRACE_ALL_BATCHES=0`（避免 OPHIS 1.7GB trace 事故）

### 臂
| run_id | 臂 | 说明 |
|---|---|---|
| `ngram5_order5_trigram_fixed` | +trigram 注入 | 5gram context 通过 trigram 表注入（input 位），主臂 |
| `ngram5_order5_puretransformer_fixed` | 纯 transformer | 无 n-gram 表（negative control）|
| `ngram5_order5_trigram_s43_fixed` | +trigram 注入 · seed 43 | 与 N1 完全相同 setting，仅 seed=43，跨 seed 复现 |

### 状态
- seed 42 的四个 run 与 seed 43 主臂复现均已完成；最终值以各 run 的
  `training_loss.jsonl` / `validation_loss.jsonl` 为准。
- seed 43 主臂使用与 seed 42 相同的数据、fixed probe 和评测口径，仅训练 seed 改为 43。

### 结果（2026-08-24 回填，全部 2000 步；数值按最终 JSONL 校正）

| run_id | 臂 | 表 LR | train@2000 | val@2000 | global gap |
|---|---|---|---|---|---|
| `ngram5_order5_trigram_fixed` | +trigram 注入 | ×2 (0.008) | 0.7165 | 0.7098 | **−0.0067** |
| `ngram5_order5_trigram_lr1x_fixed` | +trigram · LR×1 | ×1 (0.004) | 0.7736 | 0.7751 | +0.0015 |
| `ngram5_order5_trigram_lr4x_fixed` | +trigram · LR×4 | ×4 (0.016) | 0.7118 | 0.7026 | **−0.0092** |
| `ngram5_order5_puretransformer_fixed` | 无表对照 | — | 0.8311 | 0.8364 | +0.0054 |
| `ngram5_order5_trigram_s43_fixed` | +trigram 注入 · seed 43 | ×2 (0.008) | 0.6946 | 0.6856 | **−0.0090** |

主臂两 seed 汇总：global gap 均值 **−0.0078**，样本标准差 **0.0016**
（seed 42/43；仅两个 seed，不作更强的 uncertainty 声明）。

**关键发现（自然语言 5gram vs 合成 markov）**：
1. **表有效降低 train loss**：LR×2/×4（0.70）< LR×1（0.77）< 无表（0.83）。表在学 5gram context。
2. **全局 gap 极小（−0.0092 到 +0.0054）**：43M distinct 5gram contexts 挤在 1M 行表，collision 可能稀释 coincidental gap。与合成 markov（gap 可达 2+）完全不同；其中主臂的 −0.0067（seed 42）与 −0.0090（seed 43）方向一致，但目前仍只有两个 seed。
3. **表 LR 消融**：×2 与 ×4 的 train/val loss 较低（0.7165/0.7098 与 0.7118/0.7026），×1 稍差（0.7736/0.7751）；LR 消融目前只有 seed 42。
4. **per-bucket gap 峰在中频段**（trigram 主臂，step 2000）：
   - `[21,51)`: **+1.00**
   - `[101,201)`: +0.55
   - `[501,1001)`: **+1.82**
   - `[1,50)`: ~0 甚至负（表学不过来，43M 挤 1M）
   - `>5000`: ~0（样本足够，val 稳定）
   - **暂观察到 gap 峰在中高频而非低频**，但这些桶的 token fraction 很小、可配对频次类较少；seed 43 的主臂图形相近，但仍需 seed 44 和更多配对频次统计后再判断是否是自然语言长尾分布下的稳健现象。

### 待办
- [x] 回填两臂 final gap @2000
- [x] 学习率消融（table LR scale ×1 / ×2 / ×4）
- [x] 频次分解图（gap-vs-frequency，seed 42/43 探索性版本）
- [ ] bigram 注入臂（是否需要）
- [x] seed 43 主臂复现（与 seed 42 同一 data/probe hash，gap −0.0090）
- [ ] seed 44 复现；LR 消融的多 seed 复现

## 20. bigram 大表 + 免碰撞（perfect-map）极限臂（2026-08-25）⚠️ **历史 4 层框架**

> **2026-08-25 补充**：本节所有 run 均基于**旧 4 层求和 + 2-hash 拼接架构**。
> 用户同日拍板改为 **clean 单表**重扫（`docs/notes/method/clean-table-rework.md`），
> 本节结果（mult=128/256 离群、perfect 2.17 倍等）属于历史框架，引用时须标注
> `[HISTORICAL 4-LAYER FRAMEWORK]`。

### 目的（superposition/localization 相图，郭绍阳提议）

检验 gap Δ 与 hash table 大小 K 的定量关系。粗糙模型 Δ ~ min(N, K) 预言
K<<N 时 log-log 斜率 1；实测 23 点网格斜率仅 0.49（bigram），且现有网格
K/N 最大 0.30（bigram N=3.54M distinct contexts），从未触及 K~N 的
jamming 区。本实验把 bigram 推到 jamming 点并加零碰撞极限锚点。

### 偏离极简基线的登记（P1 要求）

- `tbl_128/256_bigram_fixed`：仅改 `table_mult`（128/256），其余全对齐 §1。
- `tbl_perfect_bigram_l1_fixed`：bigram hash → **预计算 packed-context→row
  静态映射（零碰撞，N+1 行含 UNK）**，且 bigram 表**仅开第 1 个 ngram 层**
  （fp32 显存约束：4 层 × 3.54M × 768 × 12B > 单卡 H200）。OOV（train 未见）
  context → 共享 UNK 行，val OOV 率由 map 构建脚本记账。
- `tbl_64_bigram_l1_fixed`：mult=64 + `--bigram_single_layer`，作为 perfect
  臂的同层数对照（二者 Δ 只在单层口径内直接可比；与 4 层主网格的折算另记）。
- train.py 变更：`--bigram_perfect_map` / `--bigram_single_layer` /
  `--save_final_model`（自 tasks 副本合并回归）+ exact_freq probe=0 兜底修复。
  全部新参数默认关闭，**不影响任何已有 run 的口径**。

### 登记（planned）

| run_id | mult | K/N_bi | 监测 | 角色 |
|---|---|---|---|---|
| `tbl_128_bigram_fixed` | 128 | 0.59 | sparse（末端） | 过渡段加密 |
| `tbl_256_bigram_fixed` | 256 | 1.19 | sparse（末端） | jamming 点 |
| `tbl_64_bigram_l1_fixed` | 64 单层 | 0.30 | freq=50 曲线 | 单层对照 |
| `tbl_perfect_bigram_l1_fixed` | 零碰撞单层 | ∞ | freq=50 曲线 | Δ∞ 锚点 + forking 上限 |

数据/优化器/步数全部对齐正式网格（shard 1，val 2-10+6542，epoch 337，
β=(0,0.99)，lr 0.004×2，bf16，1000 步，seed 42）。launcher:
`tasks/s1_scaling_three_axis/launchers/run_bigram_large_perfect.sh`；
map 构建: `code/tools/make_bigram_perfect_map.py`。

### 结果（2026-08-25 回填，seed 42，1000 步，online final gap）

| run_id | K/N_bi | collision | singleton | final gap |
|---|---|---|---|---|
| `tbl_128_bigram_fixed` | 0.59 | 0.714 | 0.060 | **+0.5773** |
| `tbl_256_bigram_fixed` | 1.19 | 0.518 | 0.190 | **+1.2042** |
| `tbl_64_bigram_l1_fixed`（单层碰撞对照） | 0.30 | 0.852 | 0.005 | **+0.2604** |
| `tbl_perfect_bigram_l1_fixed`（单层零碰撞） | ∞ | 0 | 1.0 | **+0.5651** |

参照：4 层 mult=64（正式网格）= +0.9985。perfect map：distinct = 3,538,293
（与 table_occupancy 完全一致），val OOV token 率 **4.30%**（shards 2,3）。

**发现**：
1. **单层裁决：零碰撞 gap 是碰撞版的 2.17 倍**（0.565 vs 0.260，同层数同预算，
   唯一差异是碰撞）→ 碰撞本身抑制 gap，直接支持「碰撞削弱表记忆」机制。
   层数折算 0.999/0.260 ≈ 3.8（4 层近似线性累加）。
2. **4 层 64→128→256 非单调**（0.999 → 0.577 → 1.204）：mult=128 显著离群。
   单 seed 无法区分两种解释：(a) 用户猜测的 jamming 区临界涨落；(b) hash 实现
   伪影（乘法 hash mod 2^k，R=2^19/2^20/2^21 的低位结构差异；occupancy 早已
   显示 hash 远非均匀：mult=64 singleton 0.47% vs Poisson 预言 ~11.5%）。
   **需 seed 43/44 仲裁**。
3. **forking 两种定义分离**（freq=50，epoch 边界 337/674）：
   - 边界瞬时 train 跳变：碰撞版更大（@337 −0.112 vs perfect −0.041）——
     碰撞行在 epoch 边界反复争抢重写，震荡大；
   - epoch 级 gap 增速：perfect 更陡且持续（epoch 2 起每 epoch 增量 ~+0.2 vs
     对照 ~+0.13）——零碰撞行无干扰，记忆净积累快。
   - 「K>N 时 forking 剧烈」的预言按定义 (a) 方向相反、按定义 (b) 方向成立；
     剧烈的是争抢震荡，不是净记忆积累。
4. **row-level（32/16/8/4/2 五档，fixed probe）**：同档内 row gap 对行的
   distinct-context 负载基本不敏感（曲线平），档间整体平移（token 加权均值
   0.05 → 0.28，log 线性）→ 与 winner-take-all 记忆一致：碰撞行里主导高频
   context 吃掉大部分梯度，负载本身边际效应弱；容量（K/N）才是主变量。

产物：`figs/fig_gap_vs_KN.png`（相图）、`figs/fig_l1_forking.png`（forking 对比）、
`figs/fig_row_level_multi.png`（row-level 三面板）；
`tasks/s1_scaling_three_axis/analysis/plot_gap_vs_KN.py` / `plot_l1_forking.py`。

### 后续（待拍板）
- [ ] mult=128/256 的 seed 43/44（仲裁 128 离群：临界涨落 vs hash 伪影）
- [ ] 可选：mult=128 换一组 primes 重跑（直接检验 hash mod 2^20 结构假说）
- [ ] 可选：4 层 perfect（bf16 表存储，~87GB）验证 4× 单层折算 ≈ 2.26

### 追加：非 2 幂加密取点（dense-fill，2026-08-25 下午）

用户要求脱离 2/4/8/16 采样，验证 gap vs log K 的连续性。补 9 点
（bigram-only，sparse 末端，同 §20 口径）：

| mult | 44 | 52 | 60 | 80 | 96 | 112 | 160 | 192 | 224 |
|---|---|---|---|---|---|---|---|---|---|
| gap | 0.414 | 0.624 | **1.122** | 0.494 | 0.581 | **1.181** | 0.586 | 0.602 | 0.998 |

**发现 5（修正发现 2）**：mult=128 不是孤点——加密后 **K/N ≈ 0.15–1.2 整个区间
都是锯齿**，峰谷交替（…0.41→1.12→0.49→1.18→0.58→0.60→1.00→1.20），振幅 ±0.3–0.4
与趋势本身同量级。关键证据：mult=60 的峰（1.122）与 mult=112（1.181）的 R 都不是
2 的幂（491520 / 917504）→ **峰不绑定 2 的幂 mod 结构，hash mod 2^k 伪影解释减弱，
jamming 临界涨落解释增强**（但仍需 seed 43/44 排除「每 mult 一个 hash 实例」的
实例噪声）。包络仍随 log K 上升（0.09 → 0.6–1.2），log K 解释趋势、不解释涨落。

产物：`figs/fig_gap_vs_KN.png`（34 点折线版）；
launcher `tasks/s1_scaling_three_axis/launchers/run_bigram_dense_fill.sh`。

### 后续（待拍板）
- [ ] jamming 区若干点的 seed 43/44（仲裁锯齿：临界涨落 vs 单 hash 实例噪声）
- [ ] 可选：4 层 perfect（bf16 表存储，~87GB）验证 4× 单层折算 ≈ 2.26

## 21. v3 波次：freq-bin train 侧改为当前 batch（online，零额外 forward）（2026-08-25）

### 目的

v2 波次 freq-bin 的 train 侧是「每次评估从独立诊断迭代器新取 4 个 train batch」
的窗口（agents.md 新口径 §1.6 之前的历史做法）。用户 2026-08-25 拍板：
**前 4 个 batch 是错误做法**，train 侧应直接看当前训练 batch 的 per-token loss
（与 online train_loss 同一 batch、同一 forward），且省掉 4 batch 的重复 forward。

### 口径变更（P2 登记：影响口径，新起 run_id）

- `code/train.py`：训练循环最后一个 micro-batch 的 forward 改为
  `return_token_losses=True`，缓存 `(inp, ptl)`；freq-bin eval 的 train 侧用
  `accumulate_freq_bins(...)` 直接复用该 per-token loss，不再 `next(freq_train_iter)`。
- 因此 v3 的 `freq_bin_loss.jsonl.train` 与 `train_log.jsonl.train_loss` 完全同 batch
  同 forward（train_loss = 该 batch 非 pad 均值；per-bin 加权均值应等于它，可用作校验）。
- val 侧、exact_freq、fixed probe 全部不变（仍为 fixed / 诊断口径）。
- launcher 复用 `run_rerun_v2.sh`（bf16、不 compile、β₂=0.99、×2），run_id 后缀 `_v3`。
- `exact_freq_loss.jsonl` 的 train 参考仍取 4 个固定 batch（它是 exact-f 的固定参考，
  属诊断口径，不进主图）；如用户后续要求也改当前 batch，另起 run_id。

### 登记（planned → running）

| run_id | 内容 | steps | 机器 |
|---|---|---|---|
| `nglab_smoke_v3` | 冒烟 100 步（回填时跳过） | 100 | 360-2 GPU0 |
| `nglab1x_input_v3` … 全量 15 run（injpos 4 + dose 11） | 主线重刷 | 2000 | 三机队列 |

### 校验（smoke 必过）

- [ ] `freq_bin_loss.jsonl` 每 eval 的 train token_count = 147456（单 batch）
- [ ] per-bin 加权均值 ≈ 同 step `train_log.train_loss`（相对差 < 1e-3）

## 22. clean 单表 bigram R 网格（新 SSOT 框架首扫，2026-08-25）

### 目的

按 `docs/notes/method/clean-table-rework.md`（用户 2026-08-25 拍板的新 SSOT）
重扫 table-size 轴：`--bigram_clean_table R`（单 `nn.Embedding(R, 768)`、单层、
单 hash、R 任意）。§20 的 34 点网格为 [HISTORICAL 4-LAYER FRAMEWORK]，本节
是新框架的第一张相图，与旧框架并列对照。

### 实现（train.py）

- `--bigram_clean_table R`：bigram 分支强制单层 `{ngram_layers[0]}` + `bigram_K=1`
  （单全宽 embedding，`torch.cat` 单张量恒等，lookup 路径零侵入）。
- 与 `--bigram_perfect_map` 组合 = 零碰撞锚点（K=1 单表 + map 行号，
  R = n_distinct+1，不再需要 single_layer 开关）。
- 默认 0（关闭），旧 4 层路径与已有 run 口径不变；四路径本地冒烟通过。
- `table_occupancy.py --bigram_clean_table R --trigram_clean_table R`：clean
  模式 occupancy（每个 branch 均为 layer-1 primes 的第一组 hash、R 行）。
- run_id namespace `ctbl_*`，产物仍入 `data/runs_scaling/`（与旧框架并列）。
- launcher `tasks/s1_scaling_three_axis/launchers/run_clean_table_grid.sh`
  （wave 调度；rolling-slot 曾把 786K/65K 发到被 perfect/1M 占用的卡上 OOM，
  已修为按波次 wait）。

### 结果（seed 42，1000 步，online final gap，N=3,538,293）

| R | K/N | gap | | R | K/N | gap |
|---|---|---|---|---|---|---|
| 64K | 0.019 | +0.097 | | 1.5M | 0.444 | +0.306 |
| 128K | 0.037 | +0.146 | | 2M | 0.593 | +0.305 |
| 256K | 0.074 | +0.159 | | 2.5M | 0.741 | +0.370 |
| 384K | 0.111 | +0.190 | | 3M | 0.889 | +0.393 |
| 512K | 0.148 | +0.218 | | 4M | 1.185 | +0.466 |
| 768K | 0.222 | +0.281 | | **perfect** | **1.0 零碰撞** | **+0.561** |
| 1M | 0.296 | +0.265 | | | | |

**发现**：
1. **clean 单表的 gap-R 曲线光滑近似单调**（仅 768K→1M 微降 0.016、
   1.5M≈2M 平台），与旧 4 层框架 ±0.3–0.4 的剧烈锯齿形成决定性对比。
   **旧框架的"jamming 锯齿"主要是 4 层 × 2-hash 拼接的架构干涉伪影
   （8 组 hash 相互作用），不是 jamming 物理**——§20 发现 5 的临界涨落
   解释在新框架下被否证。这是 clean 重做的第一个实质科学收益。
2. **碰撞抑制 gap 跨框架复现**：clean perfect（0.561）vs clean R=1M 碰撞
   （0.265）= **2.12 倍**（旧框架单层对为 2.17 倍）；且 R=4M（K/N=1.19）
   的 hash 点（0.466，collision=0.325）仍显著低于 perfect——**R>N 时 hash
   伪碰撞继续压制 gap，零碰撞的价值不随容量增大消失**。
3. **旧 4 层 gap 大部分是参数量/求和结构的功劳**：同 R=512K 附近，旧 4 层
   mult=64 gap=0.999 vs clean 0.218（4.6 倍）；旧单层 mult=64（K=2 拼接）
   0.260 vs clean 0.218——2-hash 拼接贡献 ~19%，4 层求和贡献 ~3.8 倍。
   回答 SSOT §4 问题 3：主要是参数量（4× 行数）+ 求和平均的方差缩减，
   不是"表记忆容量"本身的增益。
4. **forking（clean，同架构唯一差异=碰撞）**：epoch 边界 train 跳变
   perfect 更大（@337 −0.136 vs −0.115；@674 −0.085 vs −0.070），gap 增速
   也更快更持续（epoch 2 起每 epoch +0.15–0.2 vs +0.05–0.1）——clean 框架下
   零碰撞同时赢得瞬时跳变与净积累，旧框架"碰撞版瞬时跳变更大"的干涉
   模式消失，进一步支持锯齿=架构伪影。
5. min(N,K) 模型（图中虚线，归一化到 perfect）在 K/N<0.3 段仍高估斜率
   （实测 log 斜率 ~0.35–0.5）：频率加权修正依然必要。

产物：`figs/fig_clean_gap_vs_KN.png`、`figs/fig_clean_forking.png`；
`tasks/s1_scaling_three_axis/analysis/plot_clean_figures.py`。

### §22b · wave-2 加密网格 + trigram 首扫（2026-08-25）

用户拍板：table size 自由加密（小 R 区重点）、双对数视图、forking 只画
小表/大表/零碰撞三条。launcher `run_clean_table_dense2.sh`（25 run，360-2
GPU 1-7，wave 调度）。bigram 累计 29 点（新增 16K/32K/96K/160K/192K/320K/
448K/640K/896K/1.25M/1.75M/2.25M/3.5M/5M/6M + 64K/4M 的 freq=50 `_curve`
补跑），trigram 首扫 6 点（R=64K…8M，K/N 0.0034–0.44；trigram N=18,989,467，
无零碰撞锚点——R=N 需 19M 行放不下单卡）。

**发现**：
1. **大 R 区（K/N>1）涨落回归**：5M 回落到 0.407（低于 4M 的 0.466），
   **6M 跳到 0.751，超过 perfect 锚点 0.561**。"零碰撞是所有 hash 表的
   上界"在 K/N>1.5 区不再成立（该论断在 K/N≤1.19 区全部成立）。
   单 seed 无法区分 hash 实例噪声与真实过饱和行为——**待 seed 43/44 仲裁**。
2. **低 R 饱和**：16K/32K 均 ~0.08，与 64K 0.097 接近——小表端 gap 由
   backbone 自身过拟合地板主导，表贡献趋零。
3. **trigram 远陡于 bigram**：同 K/N 下 gap 3–5 倍；log-log 低 K 段斜率
   tri ≈ 0.67 vs bi ≈ 0.33。min(N,K) 模型预言低 K 斜率 1，两者均不满足
   （bigram 0.33–0.49、trigram 0.67），频率加权 × 采样律修正仍必要。
4. **双对数视图**（`fig_clean_gap_vs_KN_loglog.png`）：bigram 主体近似
   幂律但大 R 断点清晰；trigram 全段更接近直线幂律。
5. **forking 三曲线**（64K / 4M / perfect）：64K gap 轨迹平（<0.1）但
   瞬时跳变幅度与其他曲线相当（@337 −0.107）；4M 与 perfect 前 2 epoch
   重合，epoch 3 perfect 拉开——零碰撞优势主要在 replay 后期的积累阶段。

产物：`figs/fig_clean_gap_vs_KN.png`（semilog 29+6 点）、
`figs/fig_clean_gap_vs_KN_loglog.png`、`figs/fig_clean_forking.png`；
launcher `run_clean_table_dense2.sh`。

### 后续（待拍板）
- [ ] clean 网格 seed 43/44（**仲裁 6M>perfect 是否 hash 实例噪声**；同时
  确认光滑性与 perfect 倍率的跨 seed 稳健性）
- [ ] clean 版 trigram 加密 + both module（SSOT §3.2 要求三 module）
- [ ] clean 版 row-level（--save_final_model + probe，复用 §20 管线）
- [ ] jamming 区更密取点（R 1M-3M 间插 6-8 点）刻画 1.5M≈2M 平台结构
- [x] trigram clean occupancy 模式已实现；18 点 v5 双表产物的 occupancy
  回填待执行，见 §24b。

---

## §21 · V4 波次（uniform LR 基线 · 2026-08-25）

**背景**：v2/v3 全程含 warmdown（`get_lr_multiplier`，前 35% 步线性升、后 65% 步线性降）。
对可解释性实验，lr 的时间结构会与「n-gram 记忆 vs 泛化」动力学混淆，违反 P1 极简原则。
**用户 2026-08-25 拍板**：改为 **uniform LR（constant schedule）**，作为 v4 新基线。
生产 cosine 流行是为 maximize performance，不是为 mechanism isolation——所以选 uniform 而非 cosine。

**代码改动**（commit `0da5ad0`）：
- `code/train.py`：新增 `--lr_schedule warmdown|constant`（默认 warmdown 保持兼容）；
  `constant` 时 `get_lr_multiplier` 恒返 1.0。
- `code/cluster/run_rerun_v4.sh`：新 v4 launcher，显式传 `--lr_schedule constant`。

**口径影响**：lr schedule 全程变化 → 影响所有已有 run 的 gap 曲线形状。
按 P2 必须新起 run_id，故开 **v4 波次**（后缀 `_v4_fixed`）。v2/v3 降级为「含 warmdown 的历史口径」。

**当前队列**（360-2，GPU3-6 并行，2000 步，seed 42）：
- `nglab1x_input_v4` / `nglab1x_y_v4` / `nglab1x_v_v4` / `nglab1x_nogram_v4`

| run_id | 日期 | 实验 | 状态 | gap 关键值 | 详情 |
|---|---|---|---|---|---|
| `nglab1x_input_v4` | 2026-08-25 | 注入点消融 · input · uniform LR | ⛔ superseded | 未形成权威产物；被 v5-refresh 取代 | §21 |
| `nglab1x_y_v4` | 2026-08-25 | 注入点消融 · y · uniform LR | ⛔ superseded | 未形成权威产物；被 v5-refresh 取代 | §21 |
| `nglab1x_v_v4` | 2026-08-25 | 注入点消融 · v · uniform LR | ⛔ superseded | 未形成权威产物；被 v5-refresh 取代 | §21 |
| `nglab1x_nogram_v4` | 2026-08-25 | 注入点消融 · nogram 对照 · uniform LR | ⛔ superseded | 未形成权威产物；被 v5-refresh 取代 | §21 |

---

## §21a · Clean-table backbone LR 快速扫描（warmup 后恒定 · 2026-08-25）

**科学问题**：在当前 clean 单表、online train loss、固定 validation 的极简契约下，
将 backbone base LR 从 `0.004` 降至 vanilla nanoGPT 量级的 `0.0006` 或
`0.0004`，是否改变 input 注入的训练/验证分叉形态？这是一组筛选实验，不替代
多 seed 主结论。

**唯一变量**：`--lr`。三个臂均保留 `--table_lr_scale 2.0`，因此 table 的实际
LR 分别为 `0.008`、`0.0012`、`0.0008`；不能把比较解释为“固定 table LR”的实验。
其余坐标严格一致：vanilla nanoGPT 8L/6H/768D、input 注入、clean bigram/trigram
单表 `R=2^20`、RMSProp `(0.0,0.99)`、bf16、不 compile、fixed replay、seed 42、
train shard `1`、非重叠 validation shards `2,3,4,5,6,7,8,9,10,6542`、每 10 step
fixed val + current-batch frequency bins，`warmup_constant`（100 steps）。

**可证伪预期**：若 `0.004` 的异常形态主要由绝对步长过大造成，低 LR 臂应在相同
1000 steps 内呈现更稳定的 train/val 曲线；若只是学习更慢，则低 LR 臂会整体滞后，
而非仅消除异常。验收条件：每臂存在 `summary.json`、100 个 logged `train_log.jsonl`
点、100 个 `freq_bin_loss.jsonl` 点，且 summary 记录 clean-table R、bf16、未 compile、
`warmup_constant` 与本臂 LR。

| run_id | 日期 | 实验 | 状态 | gap 关键值 | 详情 |
|---|---|---|---|---|---|
| `lrscan_input_lr0p004_wc` | 2026-08-25 | clean-table input · backbone LR 0.004 | ✅ done | +0.060@1000 | §21a |
| `lrscan_input_lr0p0006_wc` | 2026-08-25 | clean-table input · backbone LR 0.0006 | ✅ done | +1.534@1000 | §21a |
| `lrscan_input_lr0p0004_wc` | 2026-08-25 | clean-table input · backbone LR 0.0004 | ✅ done | +1.187@1000 | §21a |
| `lrscan_y_lr0p0006_wc` | 2026-08-25 | clean-table y · backbone LR 0.0006 | ✅ done | +1.008@1000 | §21a |
| `lrscan_v_lr0p0006_wc` | 2026-08-25 | clean-table v · backbone LR 0.0006 | ✅ done | +0.209@1000 | §21a |
| `lrscan_nogram_lr0p0006_wc` | 2026-08-25 | no-gram 对照 · backbone LR 0.0006 | ✅ done | +0.025@1000 | §21a |

**执行位置与命令**：360-1（GPU0/1/2），仓库
`/data/home/guoshaoyang/ngram-gap-lab`，解释器 `python3`；命令唯一地由
`code/train.py` 与下列显式 flags 确定：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python3 -u code/train.py \
  --run_id <run_id>_fixed --out_dir data/runs_fixed --data_dir data/tokenized \
  --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 --seed 42 --steps 1000 \
  --dtype bf16 --injection_position input --enable_unigram 0 --enable_bigram 1 \
  --enable_trigram 1 --bigram_clean_table 1048576 --trigram_clean_table 1048576 \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size 72 --total_batch_size 147456 --lr <0.004|0.0006|0.0004> \
  --lr_schedule warmup_constant --warmup_steps 100 --table_optimizer rmsprop \
  --table_betas 0.0,0.99 --table_lr_scale 2.0 --val_interval 10 --val_batches 4 \
  --freq_index data/freq_index.npz --freq_eval_interval 10 --freq_eval_batches 4 \
  --exact_freq_eval_interval 10 --table_norm_interval 10 --fixed_train_probe 0
```

**回填结果（seed 42，step 1000）**：`0.004` 的 online train / fixed val 为
`6.174 / 6.234`，gap `+0.060`；`0.0006` 为 `2.899 / 4.433`，gap
`+1.534`；`0.0004` 为 `3.121 / 4.309`，gap `+1.187`。三臂各有 100 个
`train_log.jsonl`、`freq_bin_loss.jsonl`、`table_norm.jsonl` 记录；summary
已核验为 clean `R_bigram=R_trigram=1,048,576`、`bf16`、`torch_compile=false`、
`lr_schedule=warmup_constant`。现象图：
`docs/figs/main/fig_lrscan_clean_input.png`，其原始数据仅来自这三条 run 的
`train_log.jsonl`。该单 seed 快速筛选支持优先推进 `6e-4`；是否把它升级为
SSOT 仍需完整注入点臂与多 seed 确认。

**注入点快速对照（已登记）**：在同一 `0.0006` 基线补 `y`、`v` 与 no-gram
负对照，各 1000 steps。相对 `lrscan_input_lr0p0006_wc`，每条只改 injection
coordinate（no-gram 则关闭 bigram/trigram，作为既定负对照）；其余完整命令、数据、
clean R、评估节奏与验收条件完全相同。**回填（seed 42，step 1000）**：input
`+1.534`、y `+1.008`、v `+0.209`、no-gram `+0.025`。因此 input 的强 gap
不是低 LR 下纯 backbone 的共同现象；y 也有分叉但低于 input，v 更弱。该结果只作为
1000-step 单 seed gate，完整 2000-step 四臂仍由 §25 的 `nglab1x_*_v5` 重刷确认。

---

## §24 · V5 优化器与学习率定标（clean-table 主线 gate，2026-08-25）

**固定 v5 基线**：vanilla nanoGPT 8L/6H/768D；input 注入（optimizer
对照唯一例外见下）；bigram + trigram clean 单表，均为 `R=1,048,576`；
fixed replay train shard `1`、非重叠 val `2,3,4,5,6,7,8,9,10,6542`；
seed 42、bf16、无 compile；backbone AdamW `(0.8,0.95)`、wd `0.1`、
`lr=0.0006`；`warmup_constant --warmup_steps 100`；online train loss 与
fixed validation loss 的同 logged-step gap。table 主基线为 RMSProp 无动量
`(0.0,0.99)`、`table_lr_scale=2.0`（实际 `0.0012`）。

**目的与判定**：先锁定 table 优化器的健康区，而不是挑选最大 gap。一个臂若出现
NaN/Inf、末端 train loss 不低于 no-gram 对照、或 table RMS 失控，判为不健康，不得
升级为主线。末端筛选臂显式使用 `--val_steps 1000`，只在 step 1000 计算固定 val、
online gap 与频率统计；关键曲线臂每 10 step 记录完整轨迹。最终主线必须同时满足：
训练稳定、input/nogram 分离、表 scale=`2` 的局部稳健性、β₂=`0.99` 的局部稳健性，
并由 seed 43/44 复现。

| run_id | 变量（其余严格为 v5 基线） | 预算 / 测量 | 状态 | 详情 |
|---|---|---|---|---|
| `optv5_rms_b099_s0p5` | table scale **0.5** | 1000，末端 | ✅ done | gap +0.490 |
| `optv5_rms_b099_s1p0` | table scale **1.0** | 1000，末端 | ✅ done | gap +0.861 |
| `lrscan_input_lr0p0006_wc` | table scale 2.0、β₂ 0.99 | 1000，完整曲线 | ✅ done | v5 参照，gap +1.534 |
| `optv5_rms_b099_s3p0` | table scale **3.0** | 1000，末端 | ✅ done | gap +1.904 |
| `optv5_rms_b099_s4p0` | table scale **4.0** | 1000，末端 | ✅ done | gap +2.083 |
| `optv5_rms_b095_s2p0` | RMSProp β₂ **0.95** | 1000，末端 | ✅ done | gap +1.239 |
| `optv5_rms_b098_s2p0` | RMSProp β₂ **0.98** | 1000，末端 | ✅ done | gap +1.444 |
| `optv5_rms_b0995_s2p0` | RMSProp β₂ **0.995** | 1000，末端 | ✅ done | gap +1.607 |
| `optv5_rms_b0999_s2p0` | RMSProp β₂ **0.999** | 1000，末端 | ✅ done | gap +1.629 |
| `optv5_adamw_b099_s2p0` | table optimizer **AdamW `(0,0.99)`** | 1000，末端 | ✅ done | gap +1.503 |
| `optv5_sgd_m0_s2p0` | table optimizer **SGD momentum 0** | 1000，末端 | ✅ done | gap +0.051 |
| `optv5_rms_b098_s1p0` | β₂ **0.98**、scale **1.0** | 1000，末端 | ✅ done | gap +0.768 |
| `optv5_rms_b098_s3p0` | β₂ **0.98**、scale **3.0** | 1000，末端 | ✅ done | gap +1.752 |
| `optv5_rms_b0995_s1p0` | β₂ **0.995**、scale **1.0** | 1000，末端 | ✅ done | gap +0.966 |
| `optv5_rms_b0995_s3p0` | β₂ **0.995**、scale **3.0** | 1000，末端 | ✅ done | gap +1.943 |
| `optv5_rms_b098_s2p0_curve` | β₂ **0.98** | 1000，freq=10 曲线 | ✅ done | gap +1.403 |
| `optv5_rms_b099_s1p0_curve` | scale **1.0** | 1000，freq=10 曲线 | ✅ done | gap +0.856 |
| `optv5_rms_b099_s3p0_curve` | scale **3.0** | 1000，freq=10 曲线 | ✅ done | gap +1.895 |
| `optv5_rms_b099_s2p0_s43_2000` | seed **43** | 2000，freq=10 曲线 | ✅ done | 多 seed gate，gap +5.681 |
| `optv5_rms_b099_s2p0_s44_2000` | seed **44** | 2000，freq=10 曲线 | ✅ done | 多 seed gate，gap +5.458 |
| `optv5_rms_b0995_s2p0_s43_2000` | β₂ **0.995**、seed 43 | 2000，freq=10 曲线 | ✅ done | 高 β₂ gate，gap +5.891 |
| `optv5_rms_b0995_s2p0_s44_2000` | β₂ **0.995**、seed 44 | 2000，freq=10 曲线 | ✅ done | 高 β₂ gate，gap +5.681 |

**运行后唯一可接受的选择规则**：不以本批 single-seed 的最大 gap 决定主线；
若 `scale=2, β₂=0.99` 位于健康的局部平坦区，保留它作为 v5 默认。只有它被本批
证伪（数值不稳或被相邻点严格支配）时，才在同一 clean-table setting 下选择相邻
健康点并以新的 run_id 登记主线。

**multi-seed gate 回填（2000 steps）**：β₂ `.99` 的 seed 43/44 final
`(train,val,gap)` 分别为 `(0.878,6.560,5.681)` / `(0.967,6.425,5.458)`；
β₂ `.995` 为 `(0.877,6.768,5.891)` / `(0.901,6.581,5.681)`。`.995` 的 gap
增量伴随两 seed 都更高的 validation loss（`+0.208/+0.156`），不作为更好的
机制 setting。按预注册健康规则，v5 固定 **RMSProp `(0,0.99)`、scale `2.0`**；
`.995` 保留为优化器消融结果。

---

## §24b · V5 证据刷新：完整曲线 optimizer、causal 与剂量频率（2026-08-26）

### 固定契约

本节新增 run 不覆盖 §24/§25 的 completed run。全部使用当前极简基线：
vanilla nanoGPT 8L/6H/768D，input 注入，bigram + trigram clean 单表且
`R=2^20`，backbone AdamW `(0.8,0.95)`、wd `0.1`、`lr=0.0006`，
table RMSProp 无动量 `(0.0,0.99)`、scale `2.0`（只在 optimizer 消融臂中
改变登记变量），`warmup_constant(100)`，bf16、不 compile。评估固定为
`val_interval=10`、`freq_eval_interval=10`、`exact_freq_eval_interval=10`、
`table_norm_interval=10`。gap 只能由同一 logged step 的 fixed validation loss
减当前训练 batch online loss 得到。

| family | run_id 模式 | 数量 | steps | 唯一变量 | 状态 |
|---|---|---:|---:|---|---|
| optimizer full curves | `optv5c_*` | 11 | 1000 | table scale / β₂ / table optimizer | ✅ done (2026-08-26) |
| causal refresh | `causalv5c_*` | 6 | 1000 | epoch 边界干预 | ✅ done (2026-08-26) |
| M2 frequency refresh | `nglab1x_{input,y,v,nogram}_v5_freq10` | 4 | 2000 | injection position | ✅ done (2026-08-26) |
| dose frequency refresh | `nglab{0_25x..8x}_input_v5_freq10` | 11 | 2000 | non-1x train-shard dose | ✅ done (2026-08-26) |
| table occupancy backfill | `ctbl_v5_both_{R}` | 18 | no retraining | historical clean-table diagnostics | 🗄️ superseded by formal `s1v5_128_tbl_{bi2,tri2}_R*` load-proxy evidence |

### Preflight 记录（2026-08-26）

- source commit：`b976c71`（`feat(v5): add current-batch evidence refresh`）；该 commit
  已定向同步至 `ophis-gpu`、`360-1`、`360-2`。三机的
  `train.py`、`ngram_freq.py`、`run_v5_clean.sh`、主 manifest、optimizer sweep
  与 table grid 的 MD5 已逐文件核对一致。
- 三机均通过 `bash -n` 与 Python compile；可用解释器分别为 ophis torch
  `2.9.1+cu128`、360 torch `2.13.0+cu130`。
- `360-1` 缺 5x / 6x / 8x 的专属 `freq_index_train*.npz`，故不接受
  `dose_freq10` 的整批调度；所有 11 个剂量索引已在 ophis 与 360-2 存在。
- `nglab_smoke_v3_fixed` / `nglab_smoke_v2_fixed` 为非主线 smoke，不能回填为
  实验结果。v5-refresh 的产物目录必须 create-only；已有 partial 目录应先报告，
  不得覆盖。
- 2026-08-26 的 100-step 端到端 smoke 通过（当前 batch `freq_bin_loss`、
  exact-frequency、table RMS 与 hash-reseed event 均写出），但其中
  `nglab1x_input_v5_freq10_fixed`（360-1）和
  `optv5c_rms_b099_s2p0_fixed`（360-2）误使用了正式 ID，均明确作废且保留为
  smoke。正式完整 run 改为 `nglab1x_input_v5_freq10_r1` 与
  `optv5c_rms_b099_s2p0_r1`，不覆盖、不删除 smoke 目录。
- 360-2 上误以 `run_v5_table_grid.sh` 对无 summary 的目录启动了 5 条 table-size
  重训；4 条已有正常 summary 的重复结果与 5 条未完成 partial 都不作为权威证据，
  队列已停止且产物保留。为杜绝此类重训，18 条权威 summary 位于 360-1 时只允许运行
  `code/cluster/backfill_v5_table_occupancy.sh`：它拒绝缺 summary 的目录、绝不调用
  `train.py`，只产生 occupancy JSON。
- 剂量 refresh 在 360-2 初始调度时短暂重复启动了仍由 ophis-gpu 运行的
  `nglab0_25x_input_v5_freq10` 与 `nglab0_5x_input_v5_freq10`。二者均在产生
  `summary.json` 前收到 `SIGTERM`，对应 360-2 目录保留为非权威 partial；
  这两个剂量只读取 ophis-gpu 的权威完整产物。360-2 队列随后自动释放两张卡并继续
  其余 9 个互不重复的剂量 run。
- M2 current-batch frequency refresh 已在 360-1 完成。四臂在 step 2000 都具有
  `summary.json`、`train_log.jsonl`、`freq_bin_loss.jsonl`、`exact_freq_loss.jsonl`
  与 `table_norm.jsonl`；final gap（input/y/v/nogram）为
  `5.754749/3.465313/2.011368/0.248242`。绘图只读取这四条 final freq-bin
  记录，图中 train 侧是该 logged step 的当前训练 batch pre-update per-token loss，
  val 侧是 fixed validation batches；bigram/trigram 图已生成并嵌入 registry。
- Optimizer full curves 已在 360-2 完成并回传小型证据。11 条正式
  `optv5c_*` run 全部到达 step 1000，每条均有 100 条 `train_log.jsonl`、
  `freq_bin_loss.jsonl`、`exact_freq_loss.jsonl` 与 `table_norm.jsonl` 记录；
  `summary.json` 的 train、val 与 gap 均为有限值。正式图
  `fig_v5_optimizer_full_curves.png` 与
  `fig_v5_optimizer_frequency.png` 已只从这些完整产物生成并嵌入 registry。

所有新训练 run 的 owner 为 local v5-refresh queue，seed 42，结果目录为
`data/runs_fixed/<run_id>_fixed/`；训练/验证 shard 均由下表和 launcher
explicitly 固定。每个训练 run 的验收条件均为：`summary.json` 完整、
`train_log.jsonl` 到达目标 step、`freq_bin_loss.jsonl`/`exact_freq_loss.jsonl`/
`table_norm.jsonl` 每 10 steps 覆盖、无 NaN/Inf；scalar gap 只读取
same-step `fixed val − current-batch online train`。

| run_id | train → val shards | steps | 单一变化 | 预期产物 / 状态 |
|---|---|---:|---|---|
| `optv5c_rms_b099_s0p5` | `1` → `2,3,4,5,6,7,8,9,10,6542` | 1000 | table LR scale 0.5 | 10-step curves / ✅ done, gap 0.465 |
| `optv5c_rms_b099_s1p0` | 同上 | 1000 | table LR scale 1.0 | 10-step curves / ✅ done, gap 0.858 |
| `optv5c_rms_b099_s2p0_r1` | 同上 | 1000 | table LR scale 2.0；`r1` 避开 100-step smoke | 10-step curves / ✅ done, gap 1.551 |
| `optv5c_rms_b099_s3p0` | 同上 | 1000 | table LR scale 3.0 | 10-step curves / ✅ done, gap 1.886 |
| `optv5c_rms_b099_s4p0` | 同上 | 1000 | table LR scale 4.0 | 10-step curves / ✅ done, gap 2.072 |
| `optv5c_rms_b095_s2p0` | 同上 | 1000 | RMSProp β₂=.95 | 10-step curves / ✅ done, gap 1.239 |
| `optv5c_rms_b098_s2p0` | 同上 | 1000 | RMSProp β₂=.98 | 10-step curves / ✅ done, gap 1.460 |
| `optv5c_rms_b0995_s2p0` | 同上 | 1000 | RMSProp β₂=.995 | 10-step curves / ✅ done, gap 1.594 |
| `optv5c_rms_b0999_s2p0` | 同上 | 1000 | RMSProp β₂=.999 | 10-step curves / ✅ done, gap 1.669 |
| `optv5c_adamw_b099_s2p0` | 同上 | 1000 | table AdamW `(0,.99)` | 10-step curves / ✅ done, gap 1.502 |
| `optv5c_sgd_m0_s2p0` | 同上 | 1000 | table SGD momentum 0 | 10-step curves / ✅ done, gap 0.078 |
| `causalv5c_none` | `1` → `2,3,4,5,6,7,8,9,10,6542` | 1000 | no intervention | 10-step curves / ✅ done, gap 1.544 |
| `causalv5c_freeze_table_e1` | 同上 | 1000 | freeze table at epoch 2 | event step 338 / ✅ done, gap 1.133 |
| `causalv5c_freeze_backbone_e1` | 同上 | 1000 | freeze backbone at epoch 2 | event step 338 / ✅ done, gap 0.821 |
| `causalv5c_hash_reseed_e1` | 同上 | 1000 | reseed context→row hash at epoch 2 | state-preserved event step 338 / ✅ done, gap 0.637 |
| `causalv5c_mask_low_f200_e1` | 同上 | 1000 | mask `f<200`（旧语义 `f≤200`）at epoch 2 | index provenance / ✅ done（旧语义）, gap 0.066 |
| `causalv5c_mask_high_f200_e1` | 同上 | 1000 | mask `f≥200`（旧语义 `f>200`）at epoch 2 | index provenance / ✅ done（旧语义）, gap 1.529 |
| `nglab0_25x_input_v5_freq10` | `62` → `2,3,4,5,6,7,8,9,10,6542` | 2000 | 0.25x dose | 10-step curves + matching index / ✅ done, gap 11.536 |
| `nglab0_5x_input_v5_freq10` | `60` → `2,3,4,5,6,7,8,9,10,6542` | 2000 | 0.5x dose | 10-step curves + matching index / ✅ done, gap 9.234 |
| `nglab0_75x_input_v5_freq10` | `63` → `2,3,4,5,6,7,8,9,10,6542` | 2000 | 0.75x dose | 10-step curves + matching index / ✅ done, gap 7.792 |
| `nglab1_5x_input_v5_freq10` | `1,61` → `3,4,5,6,7,8,9,10,6542` | 2000 | 1.5x dose | 10-step curves + matching index / ✅ done, gap 2.517 |
| `nglab2x_input_v5_freq10` | `1,2` → `3,4,5,6,7,8,9,10,6542` | 2000 | 2x dose | 10-step curves + matching index / ✅ done, gap 1.192 |
| `nglab2_5x_input_v5_freq10` | `1,2,64` → `4,5,6,7,8,9,10,6542` | 2000 | 2.5x dose | 10-step curves + matching index / ✅ done, gap 0.843 |
| `nglab3x_input_v5_freq10` | `1,2,3` → `4,5,6,7,8,9,10,6542` | 2000 | 3x dose | 10-step curves + matching index / ✅ done, gap 0.268 |
| `nglab4x_input_v5_freq10` | `1,2,3,4` → `5,6,7,8,9,10,6542` | 2000 | 4x dose | 10-step curves + matching index / ✅ done, gap 0.205 |
| `nglab5x_input_v5_freq10` | `1,2,3,4,5` → `6,7,8,9,10,6542` | 2000 | 5x dose | 10-step curves + matching index / ✅ done, gap 0.084 |
| `nglab6x_input_v5_freq10` | `1,2,3,4,5,6` → `7,8,9,10,6542` | 2000 | 6x dose | 10-step curves + matching index / ✅ done, gap -0.088 |
| `nglab8x_input_v5_freq10` | `1,2,3,4,5,6,7,8` → `9,10,6542` | 2000 | 8x dose | 10-step curves + matching index / ✅ done, gap -0.075 |
| `nglab1x_input_v5_freq10_r1` | `1` → `2,3,4,5,6,7,8,9,10,6542` | 2000 | M2 input current-batch frequency；`r1` 避开 100-step smoke | 10-step curves + matching index / ✅ done, gap 5.755 |
| `nglab1x_y_v5_freq10` | 同上 | 2000 | M2 y current-batch frequency | 10-step curves + matching index / ✅ done, gap 3.465 |
| `nglab1x_v_v5_freq10` | 同上 | 2000 | M2 v current-batch frequency | 10-step curves + matching index / ✅ done, gap 2.011 |
| `nglab1x_nogram_v5_freq10` | 同上 | 2000 | M2 no-gram current-batch frequency | 10-step curves + matching index / ✅ done, gap 0.248 |

### Optimizer full curves（11 臂）

所有 arm 完整记录 online train、fixed val、online gap、current-batch
frequency bins、exact frequency 与 table RMS，不能用 §24 的 sparse endpoint
取代曲线。

| 组 | run_id | 变化 |
|---|---|---|
| scale | `optv5c_rms_b099_s{0p5,1p0,2p0,3p0,4p0}` | table LR scale |
| β₂ | `optv5c_rms_b{095,098,0995,0999}_s2p0` | RMSProp β₂；`.99/s2` 与 scale 组中心点共用 |
| optimizer | `optv5c_{adamw_b099,sgd_m0}_s2p0` | AdamW `(0,.99)` / SGD momentum 0 |

`optv5_rms_*` 是早期 completed precursor：多数只在末端评估，仅
`optv5_rms_b098_s2p0_curve`、`optv5_rms_b099_s1p0_curve`、
`optv5_rms_b099_s3p0_curve` 有完整曲线。它们可用于质量审计，不能替代新 11 臂
的正式比较。

**回填结果（2026-08-26，seed 42，step 1000）**：scale
`0.5/1.0/2.0/3.0/4.0` 的 final gap 依次为
`0.464630/0.858302/1.550698/1.886087/2.072434`；β₂
`.95/.98/.995/.999`（均 scale 2，`.99` 复用中心臂）依次为
`1.239104/1.459937/1.594211/1.669014`；table AdamW `(0,.99)` 为
`1.502218`，table SGD momentum 0 为 `0.077974`。这是一批完整曲线和健康性
证据，不能仅凭最大 gap 更改 v5 的预注册中心点；scale 2、β₂ .99 仍作为主线，
其他臂仅承担消融比较。

### Causal refresh（6 臂）

干预统一在 `intervention_epoch=1`（epoch 2 开始）触发。所有事件必须写入
`summary.json.intervention.events`：
step、epoch、干预类型、hash identity 前后、频率阈值和索引 SHA256。

| run_id | 干预语义 |
|---|---|
| `causalv5c_none` | 无边界干预 control |
| `causalv5c_freeze_table_e1` / `causalv5c_freeze_backbone_e1` | 停止表写入 / backbone 更新 |
| `causalv5c_hash_reseed_e1` | 只替换 context→row hash identity；保留表权重与 RMSProp state |
| `causalv5c_mask_low_f200_e1` / `causalv5c_mask_high_f200_e1` | 按 train-shard static frequency index 屏蔽互补集合 `f<200` / `f≥200` 的 n-gram residual（**2026-08-29 晚语义修正：边界从 `f>t` 改为 `f≥t`；下方回填数值为旧 `f≤200`/`f>200` 语义，需按新语义重刷**） |

频率 mask 的静态 index 是测量与 intervention 的共同 provenance，但不可消费
训练迭代器；训练仍只从主训练流取 batch。low/high mask 使用同一阈值且必须在单元测试中
逐位置互补。

**回填结果（2026-08-26，seed 42，step 1000）**：六臂均具备 `summary.json`
与 100 条 step-10 `train_log.jsonl` / `freq_bin_loss.jsonl` /
`exact_freq_loss.jsonl` / `table_norm.jsonl`，所有终值有限。final gap 依次为：
control `1.543546`；freeze-table/backbone `1.133129/0.821469`；hash-reseed
`0.637072`；mask-low/high `0.065961/1.528893`。事件记录确认 e1 干预在
step 338；hash-reseed 改变 hash identity 且保留表参数和
optimizer state。正式图 `fig_v5_causal_losses.png` 与
`fig_v5_causal_frequency_effect.png` 只读取此批完整证据；后者排除 `novel`
桶，因为它没有 train loss，不能定义 gap。

### Dose frequency refresh（12 臂）

为每个 dose 单独选取 matching `freq_index_train*.npz`；缺任一专属索引，launcher
必须失败，不能回落到泛用 `freq_index.npz`。完成后产出 bigram/trigram 两面板的
step-2000 raw frequency-bin gap heatmap；novel bucket 不定义 gap，不进入热图。

**最终回填（2026-08-27，seed 42）**：`0.25×/0.5×/0.75×/1.5×/2×/2.5×/3×`
的 final gap 依次为 `11.535647/9.234025/7.792285/2.517155/1.192452/0.843443/0.267917`；
4×/5×/6×/8× 补齐为 `0.204953/0.084222/-0.088167/-0.075474`。11 个非 1x
run 均有 `summary.json` 和各 200 条 step-10 `train_log.jsonl` /
`freq_bin_loss.jsonl` / `exact_freq_loss.jsonl` / `table_norm.jsonl`，终值均有限；
含 M2 的 1× run 后，正式 frequency-refresh 共 12/12。5× 到 6× 之间发生 sign
change，因此图和拟合不把全区间写成单一幂律。

### Clean double-table occupancy 回填（历史，不作当前证据）

旧的 `ctbl_v5_both_{R}` occupancy 回填和
`fig_v5_s1_table_load_collision.png` 保留为历史现场，不再作为当前 S1 clean
结论来源。正式的 `s1v5_128_tbl_bi2_R*` / `s1v5_128_tbl_tri2_R*` run 没有记录
occupied rows，因此当前图只报告共享频率索引给出的 `K/R` load proxy：
`K_bi=3,541,098`、`K_tri=19,027,841`；collision rate 不从 `K/R` 反推。

---

## §24c · 高 table-LR 的 β₂ 不敏感性 gate（2026-08-26）

**动机**：用户要求选择可作定量分析的 table setting，核心门槛是 gap 不能强依赖
RMSProp 的 β₂。§24b 的有效完整 scale=2 中心臂是
`optv5c_rms_b099_s2p0_r1`（不要误用同名 100-step smoke）；β₂ 从 `.95` 到
`.999` 的 gap 为 `1.239→1.669`，跨度 `0.430`（均值的 28.6%），故**scale=2
不能被称为 β₂ 不敏感**。它的 table-LR 单轴曲线在 scale `2→3→4` 的 gap
增量为 `+0.335→+0.186`，只显示边际递减，尚未证明饱和。

**科学问题 / 可证伪比较**：在更高的固定 table LR（scale=8，实际 table LR
`0.0048`）下，RMSProp β₂ 的差异是否降到可忽略量，同时训练和 fixed validation
仍健康？若三个 β₂ 臂的末端 gap 相对跨度仍超过 10%，或出现非有限数/明显失稳，
则否定「scale=8 是 β₂ 不敏感的定量 setting」；不得仅以较大 gap 选它为主线。

**本 family 内唯一变量**：`--table_betas` 的第二项，`.95/.99/.995`。高 LR
`--table_lr_scale 8.0` 是本次 gate 的固定基准坐标，而非对 SSOT scale=2 的静默
替换；backbone LR 仍严格为 `0.0006`，不扫描 backbone LR。

**固定完整契约**：vanilla nanoGPT 8L/6H/768D；input 注入；bigram+trigram
clean 单表且各 `R=1,048,576`；backbone AdamW `(0.8,.95)`、wd `.1`、LR
`.0006`；table RMSProp、β₁=0；warmup_constant 100 steps；fixed replay
train shard `1`、non-overlap val shards `2,3,4,5,6,7,8,9,10,6542`；seed 42；
1000 steps；bf16、无 compile；val/frequency/exact-frequency/table-RMS 均每 10
steps；gap=同 logged step 的 fixed val−current-batch online train。

| run_id | GPU / cluster | 只改变量 | 结果目录 | 状态 |
|---|---|---|---|---|
| `optv5d_rms_b095_s8p0` | 360-2 GPU0 | β₂=.95 | `data/runs_fixed/optv5d_rms_b095_s8p0_fixed/` | ✅ done, gap 2.228379 |
| `optv5d_rms_b099_s8p0` | 360-2 GPU1 | β₂=.99 | `data/runs_fixed/optv5d_rms_b099_s8p0_fixed/` | ✅ done, gap 2.432147 |
| `optv5d_rms_b0995_s8p0` | 360-2 GPU2 | β₂=.995 | `data/runs_fixed/optv5d_rms_b0995_s8p0_fixed/` | ✅ done, gap 2.428866 |

**精确执行命令**（远端仓库 `/data/home/guoshaoyang/ngram-gap-lab`，解释器
`/usr/bin/python3`；运行前已核对本机与 360-2 的 `train.py`、`ngram_freq.py`、
`run_v5_clean.sh` MD5 一致）：

```bash
NGLAB_PY=/usr/bin/python3 bash code/cluster/run_v5_clean.sh <GPU> <run_id> 1 \
  2,3,4,5,6,7,8,9,10,6542 1000 --table_lr_scale 8.0 --table_betas 0.0,<beta2>
```

**验收与后续 gate**：每臂必须有完整 `summary.json`、100 个 step-10
`train_log.jsonl` / `freq_bin_loss.jsonl` / `exact_freq_loss.jsonl` /
`table_norm.jsonl` 点，达到 step 1000 且无 NaN/Inf。令
`spread=(max(gap)-min(gap))/gap(beta2=.99)`；只有 `spread≤0.10`、三个 train/val
曲线均有限且无明显末端爆炸时，才称为候选 β₂-insensitive 区。通过后才登记
scale=16 的中心臂及同样的 β₂ gate；不通过则停止把 LR 上探作为定量主 setting，
保留 scale=2 仅作 optimizer-dependent 现象设置。

**回填与后续登记（2026-08-27）**：scale=8 三臂各有 100 条上述四类
step-10 日志且 final train / val / gap 均有限。`β₂=.95/.99/.995` 的 final gap
为 `2.228379/2.432147/2.428866`，相对 `.99` 的跨度为 `8.3781%`，通过预注册的
10% gate。故在不改变 SSOT 的前提下完成同一 β₂ 三臂的 scale=16 验收：

| run_id | cluster / GPU | 唯一变量 | 状态 |
|---|---|---|---|
| `optv5e_rms_b095_s16p0` | 360-1 GPU1 | β₂=.95，table scale 16 | ✅ done；2.732527 / 5.222180 / +2.489653 |
| `optv5e_rms_b099_s16p0` | 360-1 GPU2 | β₂=.99，table scale 16 | ✅ done；2.704175 / 5.303527 / +2.599353 |
| `optv5e_rms_b0995_s16p0` | 360-1 GPU3 | β₂=.995，table scale 16 | ⚠️ failed @800：CUDA peer-memory hardware error |
| `optv5e_rms_b0995_s16p0_r1` | 360-1 GPU1 | β₂=.995，table scale 16 | ✅ done；2.697428 / 5.296299 / +2.598872 |

它们与本节固定完整契约完全相同，仅把 table scale 固定为 `16.0`
（实际 table LR `0.0096`），并仍以 `spread≤0.10` 与无 NaN/Inf 为 gate；
即使通过，也只说明该范围的 β₂ 敏感性，不能凭 single-seed gap 替代 v5 的
scale=2 主线。

**失败处理（2026-08-26；已结案）**：`optv5e_rms_b0995_s16p0` 在 step 800 的 fixed-val
计算报 `CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error`，
此前日志数值均有限，故这是 GPU3 硬件/驱动错误而非实验数值结论。保留 partial 目录
作运行溯源，不回填结果；随后以新的 `optv5e_rms_b0995_s16p0_r1` 在 GPU1 全量重跑，
代码 MD5 与 1x frequency-index SHA256 已再次核对。此前只有 r1 完整验收后才计算
scale=16 的 β₂ spread；现 scale=16 的三臂均已通过完整产物验收；其 final gap
为 `2.489653/2.599353/2.598872`（β₂=.95/.99/.995），相对 `.99` 的 spread
为 `4.2203%`，继续通过 10% gate。三臂均为 1000 steps、100 条 step-10
`train_log.jsonl` / `freq_bin_loss.jsonl` / `exact_freq_loss.jsonl` /
`table_norm.jsonl`，无 NaN/Inf。

**来源说明**：scale=16 的三条完整 run 在 360-1 上执行时使用
`train.py=f6ab90831ffd24364e3db2c47c83f913`；该版本仅比当前
`c4729b30e6f3e842b3321dc701b55bbb` 少了尚未启用的窄表宽度参数，默认
`d=768` 的 clean-table forward、优化器与测量路径一致。该差异不隐瞒，当前
`optv5f` 2000-step retry 与 S1 正式批使用当前 source；每个 family 的 source
身份以本节及 §32/§33 的记录为准。

---

## §25 · V5 主线全量重刷队列（2026-08-25）

**已锁定 setting**：optimizer gate 的第一批 endpoint 结果确认，RMSProp
`(0.0,0.99)`、table scale `2.0` 是保守主线中心点。scale `3/4` 虽给出更大
single-seed gap，却同时把 final validation loss 从 scale 2 的 `4.433` 提升到
`4.669/4.805`；无动量 SGD final gap 仅 `0.051`，AdamW `(0,.99)` 为 `1.503`，
均不构成替代 RMSProp 的理由。β₂ `0.98/.99/.995/.999` 的 1000-step gap 为
`1.444/1.534/1.607/1.629`，局部变化远小于 scale 扫描。v5 因此固定
`lr=6e-4`、RMSProp `(0,.99)`、scale `2`、clean `R_bigram=R_trigram=2^20`、
100-step `warmup_constant`、bf16、无 compile。

所有 run 均显式使用 current-batch online train loss 与 fixed validation；完整曲线
为 val/freq=10，长计算表只取末端的 run 使用 `--val_steps <final step>`。run 由
`code/cluster/run_v5_clean.sh` 单独创建目录，批处理只通过
`code/cluster/run_v5_main_manifest.sh` 调度；存在 `summary.json` 自动跳过，存在
partial 目录则拒绝覆盖。

| 家族 | run_id 模式 | 数量 | 步数 / 变量 | 状态 |
|---|---|---:|---|---|
| 注入点 | `nglab1x_{input,y,v,nogram}_v5` | 4 | 2000；唯一变量为注入坐标 | ✅ 4/4 完成：input 5.741、y 3.640、v 2.014、nogram 0.245 |
| 注入点复现 | `nglab1x_{input,y,v,nogram}_v5_s{43,44}` | 8 | 2000；唯一变量为随机 seed | ✅ 8/8 完成；s43 input/y/v/nogram=5.811/3.277/2.881/0.253，s44=5.515/3.439/2.723/0.253 |
| dose fixed-step | `nglab{0_25x..8x}_input_v5` | 11 | 2000；唯一变量为训练 shard 剂量 | ✅ 11/11 done；gap 从 0.25x 的 11.589 降至 8x 的 −0.077 |
| epoch-aligned | `nglab{0_25x..4x}_e5_v5` | 9 | 5 epoch，420–6700；唯一变量为剂量、epoch 数恒定 | ✅ 9/9 done；gap 从 0.75x 的 4.511 至 4x 的 2.089 |
| causal | `nglab1x_{freeze,hash_reseed,mask_low/high}_*_v5` | 6 | 1000；唯一变量为登记的 intervention | ✅ 当前机制登记改由 v5-refresh 的 6 臂与 mask_high 阈值扫描承担 |
| fixed probe | `nglab{1,2}x_input_rho_v5` | 2 | 2000；唯一新增诊断为 fixed probe | ✅ 2/2 done；1x gap 5.583、2x gap 1.249（以 ophis 权威 run 为准） |
| backbone safety | `nglab1x_nogram_long_v5` | 1 | 8000；无 n-gram 的长训练保险对照 | ✅ 8000/8000 done；gap 1.102 |
| table size | `ctbl_v5_both_{R}` | 18 | 1000 末端；唯一变量为 bigram/trigram 同步的 clean R | ✅ 18/18 done（权威目录在 360-1 `runs_scaling`） |
| S1 epoch-prefix | `s1v5_L{1..4}_{both,nogram}_fs` | 8 | 1000；唯一变量为 `epoch_batches=42/84/168/337` 与既定 no-gram control | ✅ 8/8 done；both gap 为 10.683/7.726/5.251/1.530，nogram 为 3.002/0.793/0.133/0.035 |
| S1 frequency | `s1v5_freq_{bigram,trigram,both,nogram}` | 4 | 1000；L4 `epoch_batches=337`，唯一变量为既定 module arm | ✅ 4/4 done；bigram/trigram/both/nogram=0.586/1.099/1.529/0.031（`runs_scaling`） |

**table-size 采样**：当前正式阵列拆成两条单变量轴；两条轴都同时开启
bigram / trigram。bigram-R 轴固定 trigram `R=2^20`，trigram-R 轴固定
bigram `R=2^20`。每条轴使用相同的 18 个近 log-uniform 点
`R=16K, 22K, 30K, 41K, 56K, 76K, 104K, 142K, 194K, 265K, 362K, 494K, 675K,
922K, 1.259M, 1.719M, 2.0M, 2.347M`，保持主线双 n-gram 结构。
`2.347M` 是在 H200 上为 RMSProp state 及完整 batch 留出的保守上限；先前误启动的
single-branch 目录是无效诊断，不纳入任何数据源或图表。

---

## §21b · clean 单表 v4 加密网格（uniform LR 重刷，2026-08-25）

**背景**：§22/§22b 的 clean 单表网格（`ctbl_*`）跑在 **warmdown** 口径
（launcher 缺 `--lr_schedule constant`，末端 `lr_m 0.05`）。用户 2026-08-25
拍板：重刷全部 v4 实验，clean 网格统一到 **v4 uniform LR**，bigram 与 trigram
**都加密取点**。双对数线性非常好（bigram 斜率 ~0.33、trigram ~0.67），趋势可信，
重刷是统一口径而非纠错。6M>perfect 的大 R 涨落**不跑 seed 43/44 仲裁**，作为
观察保留。

**口径**：与 v4 基线一致——`--lr_schedule constant`、`--table_betas 0.0,0.99`、
`--table_lr_scale 2.0`、`--dtype bf16`（不 compile）、1000 步、seed 42、sparse
末端（`--val_steps 1000`，只取 final gap）。run_id namespace `ctbl_v4_*`，
旧 warmdown `ctbl_*` 不覆盖（P2）。

**网格**：
- bigram **24 点对数均匀**（R 16K→5.42M，K/N 0.0045→1.53）+ `ctbl_v4_perfect_bigram`
  零碰撞锚点（复用 `bigram_perfect_map_s1.npz`，val OOV 4.30%）
- trigram **14 点对数均匀**（R 64K→8.39M，K/N 0.0035→0.44；19M 行放不下，
  无零碰撞锚点）
- 共 39 run；launcher `run_clean_table_v4_grid.sh`，wave 调度
  （360-2 GPU0/1/2/7 + 360-1 GPU1-7，11 卡并行）

**登记**（run 表，2026-08-25 启动，backfill 后回填）：

| run_id 前缀 | 点数 | 说明 |
|---|---|---|
| `ctbl_v4_{16K..5.42M}_bigram` | 24 | bigram 对数均匀网格 |
| `ctbl_v4_perfect_bigram` | 1 | 零碰撞锚点 |
| `ctbl_v4_{64K..7M}_trigram` | 14 | trigram 对数均匀网格（**8M 降为 7M**，见下） |

> ⚠️ **trigram 8M 降级为 7M**（2026-08-25 现场发现）：R=8M 的 fp32 表
> 状态 = 8M×768×12B ≈ 78GB，峰值 149GB > H200 单卡 143GB，初始化即 OOM
> （`expandable_segments` 亦失败；旧 warmdown `ctbl_8388608_trigram` 的 OOM
> warning 后成功是运气，不具复现性）。降为 **R=7M**（fp32 状态 64.5GB，
> K/N = 0.369），仍覆盖大 R 区，趋势完整。此改动不涉及模型架构，仅表容量
> 上界受显存约束，P1 不受影响。

**产出**：v4 权威相图（semilog + 双对数），与 §22b warmdown 版并列对照。
详细 gap 数值在 run 完成后回填到本表下方。

---

## §26 · 5-gram condition sample285 trunk 对照（2026-08-26）

**背景**：主报告第 7 节原先展示的 sample285 页面实际是 `order=3`
trigram controlled data。该页面没有可追溯的生成命令、`run_id` 或原始
run 目录，因此旧数字降级为历史存档，不能作为本节的 5-gram 证据。
本节保留原科学问题，但要求使用真正的 `order=5` context。

### 登记

| run_id | 状态 | owner | target | 变量 |
|---|---|---|---|---|
| `ngram5_order5_sample285_v5_transformer_s42` | planned | Codex | 待用户确认空闲 H200；未完成 `nvidia-smi` 前不得启动 | Transformer trunk；其余坐标固定 |
| `ngram5_order5_sample285_v5_mlp_s42` | stalled | Codex | 不适用 | `code/train.py` 当前没有 position-wise MLP trunk，必须先实现并单独 smoke；不得以历史 HTML 数字回填 |

### 固定 setting 与数据契约

- 数据条件：真实文本 `order=5`（5-gram context），`fivegram_alpha*`
  controlled blocks；完整 upstream train/val 不重叠，固定 train/val batch
  hash；sample285 的目标是每个 epoch 285 个 device steps。
- 模型：vanilla nanoGPT，8L·6H·768D，vocab 8192，sequence length 2048，
  learned absolute position，LayerNorm，tied embedding，LLLL full attention。
- 注入：input / wte；bigram + trigram value tables；unigram/fourgram 关闭。
  数据 context order 与注入表 order 是两个独立坐标，本实验是
  **5-gram condition + bigram/trigram injection**，不是 trigram condition。
- 表：clean single table，`R_bigram=R_trigram=1,048,576`，每个 branch
  一张表、一个 hash、单层；不能回落到 `table_mult` 历史架构。
- 优化：backbone AdamW betas `(0.8,0.95)`、weight decay 0.1、LR `0.0006`；
  table RMSProp 无动量，betas `(0.0,0.99)`，LR scale 2.0；固定
  `warmup_constant`，100 steps warmup，之后不 warmdown；bf16，不 compile。
- 测量：`VAL_LOSS_INTERVAL_STEPS=10`，fixed validation batches；主 gap 为
  同一 logged step 的 fixed val loss 减当前 online train loss；`novel` 不进
  gap；frequency 使用完整 upstream train epoch 的 collision-free exact
  order-5 context count。

### 可执行命令与验收

launcher：`ngram5_freq_gap/cluster/run_on_cluster.sh`，必须显式设置
`NGRAM5_RUN_ID=ngram5_order5_sample285_v5_transformer_s42`，并在目标机完成
代码 hash 核对、GPU 空闲核对和数据生成 smoke。训练前需确认数据 metadata
为 `order=5`、`block_len=7`，以及 `exact_ngram_counts.npz` 使用
`contexts` 矩阵而不是溢出的 packed int64 key。验收产物为
`data/runs_fixed/ngram5_order5_sample285_v5_transformer_s42_fixed/`，至少含
run contract、summary、training/validation JSONL、fixed batch hashes 和
exact-frequency probe 输出；MLP 臂只有在实现后另起 run_id。

---

## §23 · L6 残差—响应精确模型（2026-08-25）

**问题**：−1 是否必然出现？loss gap 是否只由方差决定？

| run_id | 只改变的变量 | 数据/重复 | 状态 | 产物 |
|---|---|---|---|---|
| `l6_counttable_freq_exact_v1` | 真概率 `p={0.50,0.20,0.05}` 与样本数 `f` | 二项分布精确枚举；无随机 seed | ✅ done | `tasks/l6_residual_response/results/l6_counttable_freq_exact_v1/` |
| `l6_response_moments_exact_v1` | 学习响应 `u(δ)={δ,sign(δ),δ³}` | Rademacher 均值精确枚举；无随机 seed | ✅ done | `tasks/l6_residual_response/results/l6_response_moments_exact_v1/` |

**结果**：count table 在 `f≥512` 的三条斜率为 −1.000001 / −1.000003 /
−1.000062，但有限 f 的局部斜率明显更浅（`p=.2`、4→8 为 −0.658）。同一
Rademacher 残差经 linear/sign/cubic 响应的拟合斜率分别为 **−1.0000 / −0.4995 /
−1.9986**；sign 臂在 `f=4096` 的精确值 0.012466，与 `√(2/(πf))` 一致。

两臂都不使用 nanoGPT、GPU 或自然语料；它们只检验数学命题。主图：
`docs/figs/theory/fig_l6_residual_response.{png,svg}`。

### §23a · `l6_counttable_freq_exact_v1`

- **状态 / owner / 日期**：`done` / Codex / 2026-08-25。
- **科学问题**：resolved finite-support count table 的 expected gap 在什么范围才趋近
  `f^-1`，三阶与四阶矩在有限样本时是否可忽略？
- **预期比较 / endpoint**：比较有限 `f` 的 exact/local slope 与大样本 `-1` guide；
  endpoint 为完整枚举 3 个 `p` × 11 个 `f`。
- **唯一实验变量**：真概率 `p={0.50,0.20,0.05}` 与每个 context 的样本数
  `f=4,8,...,4096`；估计器固定为 Jeffreys smoothing `alpha=0.5`。
- **数据 / seed / compute**：对每个二项计数逐项精确求和，无 Monte Carlo，
  `seed=null`；本地 CPU，不占 GPU、不涉及集群。
- **命令**：`.venv/bin/python tasks/l6_residual_response/run_exact.py --experiment counttable`。
- **安全复核**：追加 `--output-root <新的空目录>`，与已登记目录递归 `diff`；不覆盖
  已完成的同名 run。
- **结果目录**：`tasks/l6_residual_response/results/l6_counttable_freq_exact_v1/`。
- **验收标准**：`config.json` 与上述 contract 一致；`metrics.csv` 为 3×11=33 行、
  无 NaN；`summary.json.status=done`；大样本拟合接近 `-1`，同时保留逐相邻 `f`
  的 local slope 以暴露有限样本偏离；输出目录 create-only，禁止覆盖同名 run。
- **产物**：`config.json`（输入 contract）、`metrics.csv`（exact gap、二/三/四阶项、
  local slope）、`summary.json`（拟合区间与斜率）。

### §23b · `l6_response_moments_exact_v1`

- **状态 / owner / 日期**：`done` / Codex / 2026-08-25。
- **科学问题**：在完全相同的 `f^-1/2` 采样残差下，仅改变 learned response，
  是否会选择不同矩并产生不同 gap 指数？
- **预期比较 / endpoint**：比较三种响应的 exact curve 与各自理论矩；endpoint 为
  完整枚举 3 个 response × 10 个 `f`。
- **唯一实验变量**：`u(delta)={delta, sign(delta), delta^3}`；残差始终是 `f` 个
  Rademacher 样本的均值，`f=8,16,...,4096`。
- **数据 / seed / compute**：Rademacher/binomial 精确枚举，`seed=null`；本地 CPU，
  不占 GPU、不涉及集群。
- **命令**：`.venv/bin/python tasks/l6_residual_response/run_exact.py --experiment responses`。
- **安全复核**：追加 `--output-root <新的空目录>`，与已登记目录递归 `diff`；不覆盖
  已完成的同名 run。
- **结果目录**：`tasks/l6_residual_response/results/l6_response_moments_exact_v1/`。
- **验收标准**：`metrics.csv` 为 3×10=30 行、无 NaN；linear 臂逐点等于 `1/f`；
  cubic 臂逐点等于 `3/f^2-2/f^3`；sign 臂渐近接近 `sqrt(2/(pi f))`；
  `summary.json.status=done`；输出目录 create-only。
- **产物**：`config.json`、`metrics.csv`（exact gap、理论参考、local slope）、
  `summary.json`（拟合区间与三种响应的斜率）。

---

## §27 · X1 优化器三臂 × seed 复现（2026-08-26 登记）

**科学问题**：配对双差分口径下，table optimizer 的选择（RMSProp vs AdamW(0,.99)
vs SGD(m=0)）是否稳定地改变 n-gram 模块的 gap 贡献？seed 42 的单点结论
（rms 1.551 / adamw 1.502 / sgd 0.078）能否在 seed 43/44 复现？

**唯一变量**：table optimizer（含其 betas）；其余全部为 v5 极简契约坐标。
三臂统一 `--table_lr_scale 2.0`、betas `(0.0,0.99)`（adamw/sgd 臂 β₁=0 即无动量，
与 optv5c seed-42 波次逐旗标一致）。

### 登记总表

| run_id | 状态 | owner | 变量 |
|---|---|---|---|
| `optv5c_rms_b099_s2p0_r1_s43` | **done** | Codex | table RMSProp (0,0.99)，seed 43；gap 1.538525 |
| `optv5c_rms_b099_s2p0_r1_s44` | **done** | Codex | table RMSProp (0,0.99)，seed 44；gap 1.502156 |
| `optv5c_adamw_b099_s2p0_s43` | **done** | Codex | table AdamW (0,0.99)，seed 43；gap 1.528183 |
| `optv5c_adamw_b099_s2p0_s44` | **done** | Codex | table AdamW (0,0.99)，seed 44；gap 1.532573 |
| `optv5c_sgd_m0_s2p0_s43` | **done** | Codex | table SGD m=0，seed 43；gap 0.028385 |
| `optv5c_sgd_m0_s2p0_s44` | **done** | Codex | table SGD m=0，seed 44；gap 0.054851 |

### 复用判定（不重跑）

- **seed 42 三臂复用 `optv5c_*` 波次**（§24b，2026-08-26 done）：rms
  `s2p0_r1` gap 1.551、adamw `b099_s2p0` gap 1.502、sgd `m0_s2p0` gap 0.078。
  已核对 `optv5c_rms_b099_s2p0_r1_fixed/summary.json` 与本节规格逐项一致
  （scale 2.0 统一消除了旧波次「仅 RMSProp 带 scale 2.0」的混杂）。
- **nogram 对照复用 §25**：`nglab1x_nogram_v5_s42/s43/s44` gap
  0.245/0.253/0.253；nogram 与 optimizer 无交互，无需按臂重跑对照。

### 固定 setting

与 §24b optv5c 完全一致：train shards `1`、val shards
`2,3,4,5,6,7,8,9,10,6542`、1000 steps、bf16 无 compile、clean 单表
`R_bigram=R_trigram=2^20`、backbone lr `6e-4` warmup_constant 100 步、
device_batch 72 / total 147456、val/freq/exact/table_norm interval 全部 10。

### 命令与验收

launcher 直接派生自 `code/cluster/run_v5_optimizer_sweep.sh` 的旗标组合，
经 `run_v5_clean.sh` 执行（注意其硬编码 `--seed 42`，seed 43/44 必须显式覆盖）：

```bash
# rmsprop 臂示例（adamw/sgd 臂替换 --table_optimizer 与 run_id）
NGLAB_PY=.venv/bin/python bash code/cluster/run_v5_clean.sh <gpu> \
  optv5c_rms_b099_s2p0_r1_s43 1 2,3,4,5,6,7,8,9,10,6542 1000 \
  --seed 43 --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0
```

验收标准：每条 run 到达 step 1000、`summary.json` + 四类 step-10 JSONL
非空；回填字段为 final gap（同一 logged step fixed val − online train）、
三臂 × 三 seed 的均值与离散度；判定 SGD 低 gap 现象是否跨 seed 稳定。
全部 6 条新增 run 已到 step 1000，四类日志各 100 条且无 NaN/Inf。

### 回填（done, 2026-08-27）

seed 42 复用 §24b，seed 43/44 使用本节新增 run。final gap 如下：

| table optimizer | seed 42 | seed 43 | seed 44 | mean ± sample SD |
|---|---:|---:|---:|---:|
| RMSProp `(0,.99)` | 1.550698 | 1.538525 | 1.502156 | **1.530459 ± 0.025256** |
| AdamW `(0,.99)` | 1.502218 | 1.528183 | 1.532573 | **1.520992 ± 0.016406** |
| SGD `m=0` | 0.077974 | 0.028385 | 0.054851 | **0.053736 ± 0.024813** |

三 seed 均保持 RMSProp/AdamW 的约 `1.5` gap 与 SGD 的近零 gap；
optimizer 选择效应不是单 seed 偶然现象。上述数值均为 step-1000、
seed-specific、同一步 fixed-val − current-batch online train loss。

---

## §28 · X2 clean 表行宽扫描 d ∈ {768,192,48,12}（2026-08-26 登记）

**科学问题**：clean 单表行宽 `d=n_embd=768` 是极简 setting 的绑定项。把表行宽
降到远小于 backbone 宽度（冻结零填充投影升维到 768 后注入），gap 是否保持？
即检验「per-context 私有自由度」是否是 gap 的必要容量条件。

**唯一变量**：clean 表行宽 `d`（bigram/trigram 同步）；投影矩阵固定随机、
不训练，d=768 时与基线逐位一致。依赖 `train.py` 新增 `--bigram_table_dim` /
`--trigram_table_dim` 参数（实现中），零填充投影不影响 RNG 流与优化器分组。

### 登记总表

| run_id | 状态 | owner | 变量 |
|---|---|---|---|
| `ctbl_dim768_input_v5` | reused | — | d=768 ≡ v5 基线 `nglab1x_input_v5`（gap 5.741，§25） |
| `ctbl_dim192_input_v5` | **done** | Codex | d=192；gap 0.803015 |
| `ctbl_dim48_input_v5` | **done** | Codex | d=48；gap 0.363563 |
| `ctbl_dim12_input_v5` | **done** | Codex | d=12；gap 0.157422 |

### 固定 setting 与命令

其余坐标 = v5 主线（同 §27 固定 setting；seed 42、1000 steps、input 注入）：

```bash
NGLAB_PY=.venv/bin/python bash code/cluster/run_v5_clean.sh <gpu> \
  ctbl_dim192_input_v5 1 2,3,4,5,6,7,8,9,10,6542 1000 \
  --bigram_table_dim 192 --trigram_table_dim 192
```

验收标准：d=768 对照点直接引用基线；新增三档到达 step 1000，
`summary.json`、`train_log.jsonl`、`freq_bin_loss.jsonl`、
`exact_freq_loss.jsonl`、`table_norm.jsonl` 各有 100 条记录且无 NaN/Inf。
create-only，禁止覆盖同名目录。

### 回填（done, 2026-08-27）

| table row width `d` | final train | final fixed val | final gap |
|---:|---:|---:|---:|
| 768（v5 baseline `nglab1x_input_v5`） | 0.893233 | 6.634589 | **5.741356** |
| 192 | 3.203357 | 4.006372 | **0.803015** |
| 48 | 3.443224 | 3.806787 | **0.363563** |
| 12 | 3.548037 | 3.705458 | **0.157422** |

在该单 seed、step-1000 快速实验中，减小 clean-table row width 使 gap
从 `5.741356` 降到接近 no-gram 的 `0.245`；这是容量效应证据，不把
它升级为多 seed 的普适定律。X2 的三档新增 run 均为
`RMSProp (0,.99)`、table LR scale `2.0`、`warmup_constant(100)`、
bf16/no-compile、当前 batch online train 与 fixed validation。

---

## §29 · X3 语料侧 r̄(f) 支撑宽度统计（2026-08-26 登记）

**科学问题**：exact context frequency f 对应的输出支撑宽度 r̄(f) 在自然语料中
如何随 f 变化？这决定采样律解析区条件 `f·P(y)≫1` 在多大频率窗口内成立，
是解读 −1 斜率适用范围的语料侧前置量。

**唯一变量**：无训练变量；纯语料统计。输入为 collision-free exact count
索引 `data/freq_index.npz`（已验证：bigram keys/counts 各 3,541,098 条、
trigram 各 19,027,841 条）+ train shard 1 原始 token 流。

### 登记

| run_id | 状态 | owner | compute |
|---|---|---|---|
| `corpus_rbar_freq_v1` | **done**（2026-08-26 17:05 CST） | Codex | 零 GPU；ophis-gpu 远端 CPU（本地缺 `shard_00001.bin`，远端已确认存在） |

### 命令与验收

脚本 `code/tools/rbar_support_stats.py`（只读 freq_index + shard，
不触碰训练入口）；在 ophis-gpu 上以 `.venv/bin/python` 运行，产物拷回本地
`data/runs_fixed/corpus_rbar_freq_v1_fixed/`。

验收标准：输出 per-frequency-bin 的 r̄(f)、support 分布分位数与
`f·P(y)` 解析区覆盖率；全部数字可由 freq_index.npz + shard_00001.bin
精确复算；CPU-only，不占 GPU 卡位，不影响并行推进的 X1/X2 训练波次。

### 回填（done, 2026-08-26）

- **口径核对通过**：chunk 语义重算 bigram 3,538,293 contexts / trigram
  18,989,467 contexts；与 raw-concat 的 `freq_index.npz` 交叉核对仅
  bigram 20,027 / trigram 19,542 个 context 有差（最大 |diff| 22564/1577），
  属预期内的跨 chunk 对污染，训练侧按 chunk 语义为准。
- **关键结果（决定解析区边界）**：
  - bigram `r̄(f) ~ f^0.54`（log-log 后半窗斜率 0.544）；
    trigram `r̄(f) ~ f^0.32`（斜率 0.323）。
  - Good-Turing 缺失质量 `mgt = s1/f` 整体 `~f^-0.3`（后半窗
    bigram −0.80 / trigram −0.58）；即使 `f≈5e3–2e5` 超高频 bin，
    mgt 仍在 `10^-2–10^-3` 量级，`mgt<0.01` 覆盖的 occurrence 占比
    bigram 仅约 0.06–0.07、trigram 约 0.06。
  - 含义：自然语料 exact-frequency 支撑宽度亚线性、缺失质量衰减远慢于
    `1/f`，`f·P(y)≫1` 的解析区条件在大频率窗口内**不成立**；
    `(K−1)/f` 的 −1 采样律只属于 L3 有限 support、近似 iid 的合成对照，
    不能外推到 S1 自然语料——与两因素框架中「未解析长尾压低斜率」一致。
- 全部数字可由 `freq_index.npz` + `shard_00001.bin`（md5
  `f0e978173187ec38b7f7f5c58987016a`）精确复算。

---

## §30 · V5 zero-warmup constant schedule 配对消融（2026-08-26）

**科学问题 / 可证伪比较**：当前 v5 的前 100 step warmup 是否实质性改变 clean-table
input 的 train、fixed-val 或 gap 曲线？将本臂与完整曲线对照
`optv5c_rms_b099_s2p0_r1` 逐个 logged step 比较。若 constant 臂出现 NaN/Inf、明显
train/val 失稳，或在 epoch 1 后的曲线系统性分离，则不能把 warmup 当作无关工程细节；
若两臂均健康且 step 100 后形态近似，则 zero-warmup 是可候选的更简洁 protocol。

**run / owner / 目标**：`optv5g_rms_b099_s2p0_constant`；Codex；360-2 GPU0；
seed 42；1000 optimizer steps。结果目录为
`/data/home/guoshaoyang/ngram-gap-lab/data/runs_fixed/optv5g_rms_b099_s2p0_constant_fixed/`。
已于 2026-08-26 17:19 CST 启动，现已完成并通过验收。

**唯一变化**：相对 §24b 的中心完整曲线
`optv5c_rms_b099_s2p0_r1`，只覆盖 `--lr_schedule constant`。因此 step 1 就使用
backbone LR `0.0006` 与 table LR `0.0012`；命令中 launcher 原有的
`--warmup_steps 100` 对 `constant` 不生效。不得同时改变 table scale、β₂、数据、
seed、评估 cadence 或模型代码。

**固定契约**：vanilla nanoGPT 8L/6H/768D；input 注入；bigram+trigram clean 单表，
各 `R=1,048,576`；backbone AdamW `(0.8,.95)`、weight decay `.1`、LR `.0006`；
table RMSProp 无动量 `(0,.99)`、scale `2.0`；fixed replay train shard `1`、
non-overlap val shards `2,3,4,5,6,7,8,9,10,6542`；bf16、无 compile；
val/freq/exact-frequency/table-RMS 均每 10 steps；gap = same-step fixed val −
current-batch online train。

**代码身份与命令**：使用 360-2 已与对照曲线一致的 source：
`train.py=f6ab90831ffd24364e3db2c47c83f913`、
`ngram_freq.py=e4f45f5be1317c33e6b3c39bc6cb4bc5`、
`run_v5_clean.sh=8c86d03f79cd42d0cd559259bc77224e`。本地未同步的 `train.py`
不参与此 run。

```bash
cd /data/home/guoshaoyang/ngram-gap-lab && \
NGLAB_PY=/usr/bin/python3 bash code/cluster/run_v5_clean.sh 0 \
  optv5g_rms_b099_s2p0_constant 1 2,3,4,5,6,7,8,9,10,6542 1000 \
  --lr_schedule constant
```

**验收**：`summary.json` 有限且到达 step 1000；`train_log.jsonl`、
`freq_bin_loss.jsonl`、`exact_freq_loss.jsonl`、`table_norm.jsonl` 各恰有 100 个
step-10 记录；记录的 config 为 `lr_schedule=constant`；无 NaN/Inf。完成后回填
step-1000 train、fixed val、gap，并和 `optv5c_rms_b099_s2p0_r1` 生成三联曲线比较。

**回填**：final train / fixed val / gap 为
`4.032243 / 4.566232 / +0.533989 @1000`；四类日志均有 100 条 step-10 记录，
且无 NaN/Inf。

---

## §31 · V5 warmup 起始倍率敏感性（2026-08-26）

**科学问题 / 可证伪比较**：在固定 100-step linear warmup 长度下，起始 LR multiplier
是否决定后续 loss 进度或 replay gap？已完成的 `.25` 臂
`optv5c_rms_b099_s2p0_r1` 与运行中的 zero-warmup/constant 臂（等效 `1.0`）
构成两个端点；本节只补中间的 `.1` 与 `.5`，不重复已有实验。

**本 family 的唯一变量**：`--warmup_start_lr_mult`，分别为 `.1`、`.25`、`.5`、
`1.0`。其中 `.25` 是当前 v5 对照；`1.0` 由 §30 的 `constant` 臂提供，逐步均为
完整 LR。其余任何 CLI flag、seed、模型和数据均不变。

| run_id | owner / target | 变化 | 状态 | 结果目录 |
|---|---|---|---|---|
| `optv5h_rms_b099_s2p0_warmstart0p1` | Codex / 360-1 GPU5 | start multiplier `.1` | ✅ done, train/val/gap `2.904605/4.409104/1.504498` | `data/runs_fixed/optv5h_rms_b099_s2p0_warmstart0p1_fixed/` |
| `optv5h_rms_b099_s2p0_warmstart0p5` | Codex / 360-1 GPU7 | start multiplier `.5` | ⚠️ failed during table initialization; CUDA unspecified launch failure, no JSONL result | `data/runs_fixed/optv5h_rms_b099_s2p0_warmstart0p5_fixed/` |
| `optv5h_rms_b099_s2p0_warmstart0p5_r1` | Codex / 360-1 GPU7 | start multiplier `.5`; independent retry after GPU health probe | ⚠️ failed during table initialization; large-allocation CUDA probe reproduced the launch failure | `data/runs_fixed/optv5h_rms_b099_s2p0_warmstart0p5_r1_fixed/` |
| `optv5h_rms_b099_s2p0_warmstart0p5_r2` | Codex / 360-1 GPU5 | start multiplier `.5`; queued only after `.1` summary exists, source MD5 and free-memory rechecked | ✅ done, train/val/gap `3.003625/4.445387/1.441762` | `data/runs_fixed/optv5h_rms_b099_s2p0_warmstart0p5_r2_fixed/` |

**固定完整契约**：与 §30 和 `optv5c_rms_b099_s2p0_r1` 完全一致：vanilla
nanoGPT 8L/6H/768D；input 注入；bigram+trigram clean table、各 `R=2^20`；
backbone AdamW `(0.8,.95)`、wd `.1`、LR `.0006`；table RMSProp `(0,.99)`、
scale `2`；fixed replay train shard `1`、non-overlap val `2,3,4,5,6,7,8,9,10,6542`；
seed 42、1000 steps、bf16、不 compile；val/freq/exact-frequency/table RMS 每 10
steps；gap = same-step fixed val − current online train。

**代码身份与启动命令**：两臂均使用 360-1 已和 `.25` 对照一致的 source：
`train.py=f6ab90831ffd24364e3db2c47c83f913`、
`ngram_freq.py=e4f45f5be1317c33e6b3c39bc6cb4bc5`、
`run_v5_clean.sh=8c86d03f79cd42d0cd559259bc77224e`。

```bash
cd /data/home/guoshaoyang/ngram-gap-lab
NGLAB_PY=/usr/bin/python3 bash code/cluster/run_v5_clean.sh 5 \
  optv5h_rms_b099_s2p0_warmstart0p1 1 2,3,4,5,6,7,8,9,10,6542 1000 \
  --warmup_start_lr_mult 0.1
NGLAB_PY=/usr/bin/python3 bash code/cluster/run_v5_clean.sh 7 \
  optv5h_rms_b099_s2p0_warmstart0p5 1 2,3,4,5,6,7,8,9,10,6542 1000 \
  --warmup_start_lr_mult 0.5
```

**验收**：每个新 run 具备有限的 `summary.json`、100 条 step-10
`train_log.jsonl` / `freq_bin_loss.jsonl` / `exact_freq_loss.jsonl` /
`table_norm.jsonl`；记录 `warmup_constant`、`warmup_steps=100` 与本臂的起始
multiplier。比较四臂的 train/val/gap 全曲线，尤其是 step 100 后能否重合、step 337
与 674 后的 gap 阶梯、以及 step-1000 train/val/gap；单 seed 只判定局部敏感性，
不升级成跨 seed 标准结论。

**回填（seed 42，1000 steps）**：四臂均通过完整产物验收：`summary.json` 有限，
以及 `train_log.jsonl` / `freq_bin_loss.jsonl` / `exact_freq_loss.jsonl` /
`table_norm.jsonl` 各 100 条 step-10 记录。终值为：

| warmup 起始倍率 | run | train | fixed val | gap |
|---:|---|---:|---:|---:|
| `.1` | `optv5h_rms_b099_s2p0_warmstart0p1` | 2.904605 | 4.409104 | 1.504498 |
| `.25` | `optv5c_rms_b099_s2p0_r1`（既有对照） | 2.882657 | 4.433355 | 1.550698 |
| `.5` | `optv5h_rms_b099_s2p0_warmstart0p5_r2` | 3.003625 | 4.445387 | 1.441762 |
| `1.0`（zero warmup） | `optv5g_rms_b099_s2p0_constant` | 4.032243 | 4.566232 | 0.533989 |

`.1/.25/.5` 的 1000-step gap 落在 `1.442–1.551`，而 zero-warmup 为 `0.534`。
曲线显示 moderate warmup 的 train/val 轨迹接近，zero-warmup 在 step 100 后仍持续
落后，并在 epoch 边界约 step 337、674 后形成显著更小的 gap 阶梯。因此本批支持：
warmup 本身是重要的启动时程控制；`.25` 不是这三个 moderate 起始倍率中的孤立尖点。
这是单 seed 局部稳健性证据，不替代跨 seed 验证。图源
`docs/plot_scripts/plot_v5_warmup_start_grid.py`，输出
`docs/figs/main/fig_v5_warmup_start_grid.png`。

---

## §32（§24d 归档编号）· V5 高 table-LR × β₂ 收敛批（optv5f，2026-08-26）

> 刷新计划中的 §24d 与本节是同一批次；为避免重复正文，本日志以 §32 作为
> 唯一详细登记位置，顶部总表和图注均可按这两个编号检索。

**科学问题 / 可证伪比较**：在固定 v5 极简基线下，把 table LR scale 推到
8×–1024×，观察 RMSProp β₂ = `.99` 与 `.999` 在训练 loss / fixed-val / gap 上的
收敛差异，以及 1000 步与 2000 步预算的区别。§24c 已在 scale=8 gate 通过，本节
是用户要求的「高 scale 也看彻底收敛 + 0.999 变体」。

**本 family 的唯一变量**：`--table_lr_scale`（8/16/32/64/128/256/512/1024）×
`--table_betas` 第二分量（`.99` / `.999`）× 预算（1000 / 2000 步）。其余任何
CLI flag、seed、模型和数据均不变。

**固定完整契约**（与 §24b/§30/§31 完全一致）：vanilla nanoGPT 8L/6H/768D；
input 注入；bigram+trigram clean table、各 `R=2^20`；backbone AdamW
`(0.8,.95)`、wd `.1`、LR `.0006`、`warmup_constant(100)`；table RMSProp
`(0,β₂)`、scale 见上；fixed replay train shard `1`、non-overlap val
`2,3,4,5,6,7,8,9,10,6542`；seed 42；bf16、不 compile；val / frequency /
exact-frequency / table RMS 每 10 步；gap = same-step fixed val − current
online train。输出目录 `data/runs_fixed/<run_id>_fixed/`。

**代码身份与启动命令**：全部使用 source revision
`7583ae3222ffb4bbfb13262295a6a828e1f08d3f`：
`train.py=c4729b30e6f3e842b3321dc701b55bbb`、
`ngram_freq.py=e4f45f5be1317c33e6b3c39bc6cb4bc5`、
`run_v5_clean.sh=8c86d03f79cd42d0cd559259bc77224e`。

```bash
# 示例（scale 32 × β₂=.999 · 2000 步；其余 arm 类推）
cd /data/home/guoshaoyang/ngram-gap-lab
NGLAB_PY=/usr/bin/python3 bash code/cluster/run_v5_clean.sh <gpu> \
  optv5f_rms_b0999_s32p0_2k 1 2,3,4,5,6,7,8,9,10,6542 2000 \
  --table_betas 0.0,0.999 --table_lr_scale 32.0
```

**状态与回填**（seed 42）：

| run_id | scale | β₂ | 预算 | 状态 | train / val / gap |
|---|---:|---:|---:|---|---|
| `optv5f_rms_b099_s8p0_2k` | 8 | .99 | 2000 | ✅ done | 1.055 / 7.076 / +6.020 @2000 |
| `optv5f_rms_b0999_s8p0_2k` | 8 | .999 | 2000 | ✅ done | 1.156 / 7.056 / +5.899 @2000 |
| `optv5f_rms_b099_s16p0_2k` | 16 | .99 | 2000 | ✅ done | 1.274 / 7.035 / +5.761 @2000 |
| `optv5f_rms_b0999_s16p0_2k` | 16 | .999 | 2000 | ✅ done | 1.376 / 6.951 / +5.575 @2000 |
| `optv5f_rms_b099_s32p0` | 32 | .99 | 1000 | ✅ done | 2.637 / 5.284 / +2.647 |
| `optv5f_rms_b099_s32p0_2k` | 32 | .99 | 2000 | ✅ done | 1.263 / 6.921 / +5.659 @2000 |
| `optv5f_rms_b0999_s32p0_2k` | 32 | .999 | 2000 | ⚠️ failed @1830 | CUDA peer-GPU/hardware error；partial gap +5.096 |
| `optv5f_rms_b0999_s32p0_2k_r1` | 32 | .999 | 2000 | ✅ done（360-2 GPU0） | 1.272 / 6.833 / +5.562 @2000；config 确认 β₂=.999、scale=32 |
| `optv5f_rms_b099_s64p0` | 64 | .99 | 1000 | ✅ done | 2.635 / 5.264 / +2.629 |
| `optv5f_rms_b0999_s64p0` | 64 | .999 | 1000 | ✅ done | 2.611 / 5.241 / +2.607 |
| `optv5f_rms_b099_s64p0_2k` | 64 | .99 | 2000 | ✅ done | 1.225 / 6.965 / +5.740 @2000 |
| `optv5f_rms_b0999_s64p0_2k` | 64 | .999 | 2000 | ✅ done | 1.329 / 6.755 / +5.425 @2000 |
| `optv5f_rms_b099_s128p0` | 128 | .99 | 1000 | ✅ done | 2.619 / 5.225 / +2.606 |
| `optv5f_rms_b0999_s128p0` | 128 | .999 | 1000 | ✅ done | 2.614 / 5.230 / +2.613 |
| `optv5f_rms_b099_s128p0_2k` | 128 | .99 | 2000 | ✅ done | 1.229 / 6.953 / +5.724 @2000 |
| `optv5f_rms_b0999_s128p0_2k` | 128 | .999 | 2000 | ✅ done | 1.359 / 6.758 / +5.399 @2000 |
| `optv5f_rms_b099_s256p0_2k` | 256 | .99 | 2000 | ✅ done | 1.301 / 6.844 / +5.543 @2000 |
| `optv5f_rms_b099_s512p0_2k` | 512 | .99 | 2000 | ✅ done | 1.415 / 6.889 / +5.474 @2000 |
| `optv5f_rms_b099_s1024p0_2k` | 1024 | .99 | 2000 | ✅ done | 1.462 / 6.797 / +5.335 @2000 |

**验收**：每个新 run 具备有限的 `summary.json`、step-10 `train_log.jsonl` /
`freq_bin_loss.jsonl` / `exact_freq_loss.jsonl` / `table_norm.jsonl`（1000 步
100 条、2000 步 200 条）；记录 `table_lr_scale` 与 `table_betas`；无 NaN/Inf。
完成后回填最终 train / fixed val / gap。

**回填结论（已完成臂）**：8×–1024× 的已完成 2000-step 合法结果已全部回填；在
β₂=.99 下 gap 为 `6.020 → 5.761 → 5.659 → 5.740 → 5.724`（scale
8/16/32/64/128），256/512/1024× 进一步为 `5.543 / 5.474 / 5.335`；
β₂=.999 下已完成的 8/16/64/128× 为 `5.899 / 5.575 / 5.425 / 5.399`。
因此在当前 2000-step 预算内，增大
table-LR 并未带来持续单调的 gap 收敛；32×·β₂=.999 的原始臂已在
`evaluate_exact_freq` 处因硬件错误中止，健康卡 retry 已以独立目录完成，
且 `summary.json.config` 确认 β₂=.999、scale=32。

**图源**：`docs/plot_scripts/plot_v5_optv5f_readable.py` 与
`docs/plot_scripts/plot_v5_beta099_gap_step1000.py`，当前保留
`fig_v5_beta099_gap_step1000_vs_table_lr.png`、
`fig_v5_optv5f_readable_overview.png` 和
`fig_v5_optv5f_readable_2000_facets.png`。图中点为原始 online
记录，线为 3 点视觉连接；2000 步分面同时标出 train、fixed val、gap
和 β₂ 差值，epoch boundary 用竖线标出。

**硬件备注**：360-1 GPU7 两次在表初始化时 CUDA unspecified launch failure，
且 `ecc.errors.uncorrected.volatile.total=1`（历史不可纠正 ECC），判定该卡不可靠
并弃用；`optv5f_rms_b0999_s64p0_2k` 已在 360-2 GPU1 完成。另有
`optv5f_rms_b0999_s32p0_2k` 在 360-1 运行到 step 1830 时，于
`evaluate_exact_freq` 触发 `CUDA error: Invalid access of peer GPU memory over nvlink
or a hardware error`，没有生成合法 `summary.json`，因此不计为完成；唯一 retry
`optv5f_rms_b0999_s32p0_2k_r1` 使用新输出目录，已在 360-2 GPU0 完成；
最终 train / val / gap 为 `1.271820 / 6.833356 / +5.561536` @2000。

---

## §33 · V5 三轴 scaling 快速批（table / frequency / epoch，2026-08-26）

**目的**：在等待 §32 的 256×/512×/1024× table-LR 收敛结果期间，先用已经
确认的 `table_lr_scale=128` 对三条 S1 scaling 轴做快速现象筛查。该批只用于
确定后续完整三轴实验的形状和优先级，不把单 seed 快速结果升级为最终 scaling
定律。

**固定完整契约**：vanilla nanoGPT 8L/6H/768D；input 注入；clean 单表
bigram+trigram（各 `R=2^20`，table-size 轴除外）；backbone AdamW
`(0.8,.95)`、wd `.1`、LR `.0006`、`warmup_constant(100)`；table RMSProp
无动量 `(0,.99)`、实际 table LR `0.0006×128=0.0768`；fixed replay train
shard `1`、non-overlap validation shards `2,3,4,5,6,7,8,9,10,6542`；
seed 42；bf16、不 compile；主 scalar val / frequency / exact-frequency /
table RMS 默认每 10 步，gap 为同一步 fixed val − 当前 batch online train。
所有 scaling 结果进入 `data/runs_scaling/<run_id>_fixed/`，不写入
`data/runs/`。

**代码身份**：本批启动前锁定 `git rev-parse HEAD`
`7583ae3222ffb4bbfb13262295a6a828e1f08d3f`；
`train.py=c4729b30e6f3e842b3321dc701b55bbb`、
`ngram_freq.py=e4f45f5be1317c33e6b3c39bc6cb4bc5`、
`run_v5_clean.sh=8c86d03f79cd42d0cd559259bc77224e`。
360-2 上已再次核对前三项 hash；本批未改训练代码。

### 33.1 Table-size 轴（bigram / trigram 分开）

> ⛔ **superseded**：本 §33 的 table-size（bi2/tri2）与 epoch-length（both）
> 双表结构被用户 2026-08-27 判定为设计缺陷（固定背景表稀释斜率），由
> **§34 单表重刷批**取代为现行标准。本节仅作历史现场保留，不再承担当前结论。

科学问题：在固定 `table_lr_scale=128` 下，分别测量 bigram 与 trigram clean
table 的 physical rows `R` 如何影响 1000-step online gap。两条轴严格分开，
但都保持主实验的双支路结构：bigram-R 轴只改变 bigram 的 `R`，trigram 保持
`2^20` 且继续开启；trigram-R 轴只改变 trigram 的 `R`，bigram 保持 `2^20`
且继续开启。这样每条轴只有一个 table-size 变量。
采用 18 个近似对数点：
`16000,22000,30000,41000,56000,76000,104000,142000,194000,265000,
362000,494000,675000,922000,1259000,1719000,2000000,2347000`。
bigram 轴使用 `--bigram_clean_table R --trigram_clean_table 1048576`，
trigram 轴使用 `--bigram_clean_table 1048576 --trigram_clean_table R`；
为保留约 1、2、3 个 L4 epoch 的备用对齐点，scalar val、frequency、
exact-frequency 与 gap 显式记录 step 337、674、1000；train 主流仍按当前
batch online 语义运行，table RMS 保留默认每 10 步诊断。

| run_id 模式 | 数量 | 状态 | 结果目录 |
|---|---:|---|---|
| `s1v5_128_tbl_bi2_R{16000..2347000}` | 18 | ✅ done；gap@1000 = 2.305–2.771；逐点保留 step 337/674/1000 | `data/runs_scaling/` |
| `s1v5_128_tbl_tri2_R{16000..2347000}` | 18 | ✅ done；gap@1000 = 1.044–3.264；逐点保留 step 337/674/1000 | `data/runs_scaling/` |

验收：`summary.json`、`train_log.jsonl` 中 step 337/674/1000、step-1000 `freq_bin_loss.jsonl`
与 `table_norm.jsonl`、clean-table 参数与 `R` 一致；无 NaN/Inf。table
occupancy 如需回填，必须记录 `K/R` 与实测 collision 分离。

回填结果：bigram-R 轴的 18 个 final gap 为 `2.305–2.771`，总体随 R 增大后
趋于平台并有轻微非单调波动；trigram-R 轴为 `1.044–3.264`，随 R 增大更接近
单调上升，动态范围明显大于 bigram 轴。两条轴均保持另一张 clean table 开启，
因此这些差异不能解释为关闭 n-gram 分支的模块消融。

### 33.2 Frequency-bin 轴（只保留主实验）

科学问题：主实验双支路在固定 `table_lr_scale=128` 下的 frequency-bin gap。
只保留 `s1v5_128_frequency_main` 一个 run：input、bigram+trigram、full
clean `R=2^20`、`epoch_batches=337`、1000 steps，绑定 `data/freq_index.npz`，
并显式只在 step 337、674、1000 记录 scalar/frequency/exact-frequency/table-RMS。
不再额外跑 bigram-only/trigram-only/no-gram frequency 轴；这些模块对照可从
已有合法结果使用，新的 frequency scaling 只服务于主实验。

验收：该 run 有 `summary.json`、step 337/674/1000 的 `train_log.jsonl`、
`freq_bin_loss.jsonl`、`exact_freq_loss.jsonl`、`table_norm.jsonl`；freq-bin
train 侧必须是对应评估时的当前训练 batch，不得使用额外诊断窗口；`novel`
只有 val loss，不定义 gap。

回填结果：`s1v5_128_frequency_main` 已完成 1000 steps，gap 为
`−0.0588 / 1.1876 / 2.7361`（step 337/674/1000）；频率图源为该 run 的
原始 frequency-bin 与 exact-frequency 日志。

### 33.3 Epoch-length 轴（按 L4 倍数，统一 3 epoch）

科学问题：在固定 **3 个完整 epoch** 而非固定 step 下，epoch 长度是否改变
gap 形状。以 L4（337 batches/epoch）为单位给出 12 个倍数点：
`0.125,0.1667,0.25,0.3333,0.5,0.6667,0.75,1.0,1.25,1.5,1.75,2.0×L4`，
对应 `epoch_batches={42,56,84,112,168,224,253,337,421,506,590,674}`。
目标 steps 是 `3×epoch_batches={126,168,252,336,504,672,759,1011,1263,
1518,1770,2022}`，并在 e1/e2/e3 边界保留观测值。

这条主轴只跑 both，避免把“epoch length”与 module 变量混合。另有 L4 的
10-epoch 长训 both/no-gram 对照：`s1v5_128_ep1xL4_10ep_{both,nogram}`，
专门观察 gap 随 epoch 是否继续展开、平台或反转。

验收：3-epoch 阵列每条最终 step 必须精确等于对应目标；长训最终 step=3370。
`summary.json` 记录 `epoch_batches`、实际 epoch、`table_lr_scale=128`；
train/val/gap 全部有限。比较 3-epoch 阵列时只在 epoch 1/2/3 对齐；长训
单独画 epoch-indexed gap trajectory，并将 no-gram 与 both 同图。

回填结果：12 个 3-epoch 点的 final gap 呈 U 形，从 `4.417`（0.125×L4）
下降至 `2.728`（1.0×L4）后回升至 `5.661`（2.0×L4）。L4 的 10-epoch
长训中，both 从 step 337 的 `−0.058` 增长到 step 3370 的 `8.917`；
no-gram 同期为 `−0.042` 到 `0.480`，因此长训曲线单独解释，不与 3-epoch
横向阵列混合。

### 33.4 调度与 stop rule

本批启动器为 `code/cluster/run_v5_s1_three_axis_queue.sh`，按
table-size bigram → table-size trigram → frequency main → epoch length 顺序
消费，最多 8 卡并行；输出目录为 `data/runs_scaling/<run_id>_fixed/`。
不覆盖已有目录，partial 目录停止并人工检查。360-1 GPU7 不使用；§32 三个
高 table-LR run 继续独立运行。step 337/674/1000 的保留由显式
`--val_steps` 完成，避免只留下终点。此前旧队列中已启动的
`s1v5_128_tbl_R*`、`s1v5_128_tbl_bi_R*`、`s1v5_128_freq_{bigram,trigram,both,nogram}`
和 `s1v5_128_L{1..4}_{both,nogram}_5ep` 属于过时的双表/单支路/模块四臂/
5-epoch 方案：未完成的 partial 目录保留但标为 superseded，不进入新阵列或
图表；新队列只读取 `bi2` / `tri2` 等新 run ID。新阵列已全部完成：
table-size 36 个、frequency main 1 个、epoch-length
14 个，合计 51 个正式 run，均有合法 `summary.json`，train 日志未发现
Traceback、CUDA error、NaN 或 Inf。旧 `s1v5_128_tbl_R*`、
`s1v5_128_tbl_bi_R*`、`s1v5_128_freq_{bigram,trigram,both,nogram}` 和
`s1v5_128_L{1..4}_{both,nogram}_5ep` partial 目录仍保留作历史现场，但不纳入
正式数据源或图表。

## §34 · V5 三轴 scaling 单表重刷批（用户 2026-08-27 拍板修正）

**背景与修正动因**：§33 的 table-size 与 epoch-length 两条轴均保持「双表
开启、只变一张表」结构（`bi2`/`tri2`、`ep*_3ep` both），用户审阅后明确
指出这是设计缺陷：要测的是**单个表自身大小**的影响，另一张 clean table
必须关闭（否则总 gap 被固定背景稀释，如 bigram 轴 raw slope 仅 0.041）。
本次按用户拍板重刷：table-size 两轴改为**单表**（只开被扫描分支，另一分支
关闭）；epoch-length 轴改为**只开 trigram**（用户指定优先 trigram）。

**固定完整契约**（与 §33 完全一致，仅 branch 开关不同）：vanilla nanoGPT
8L/6H/768D；input 注入；clean 单表；backbone AdamW `(0.8,.95)`、wd `.1`、
LR `.0006`、`warmup_constant(100)`；table RMSProp 无动量 `(0,.99)`、
实际 table LR `0.0006×128=0.0768`；fixed replay train shard `1`、
non-overlap validation shards `2,3,4,5,6,7,8,9,10,6542`；seed 42；
bf16、不 compile；val/frequency/exact-frequency/table RMS 每 10 步；
scalar/frequency/exact/table-RMS 显式记录 step 337/674/1000（epoch 轴为
e1/e2/e3 边界）；gap 为同一步 fixed val − 当前 batch online train。

**代码身份**：本批修改仅 launcher
`code/cluster/run_v5_s1_three_axis.sh`（新增 `table_size_bi1` /
`table_size_tri1` / `epoch_length_tri` group，双表旧 group 保留为 superseded
兼容），未改训练代码。`train.py=c4729b30e6f3e842b3321dc701b55bbb`、
`ngram_freq.py=e4f45f5be1317c33e6b3c39bc6cb4bc5`、
`run_v5_clean.sh=8c86d03f79cd42d0cd559259bc77224e`、
`run_v5_s1_three_axis.sh=f28997d83b6809c4c786cc87a3a0dfea`。三机已同步并
核对 hash；360-1 的 `train.py` 曾为旧 hash `f6ab9083…`，已按本地 commit
版重新同步为 `c4729b30…`。

### 34.1 Table-size 单表轴（bi1 / tri1）

科学问题：只开启被扫描的那张 clean table（另一张关闭），测量其 physical
rows `R` 对 1000-step online gap 的影响。这直接对应旧 `ctbl_*` 单表序列
的 v5 + 128× 重测，用于验证旧 0.33/0.67 斜率是否在新 setting 下复现。

- `table_size_bi1`：`--enable_bigram 1 --enable_trigram 0 --bigram_clean_table R --trigram_clean_table 0`，18 点
- `table_size_tri1`：`--enable_bigram 0 --enable_trigram 1 --bigram_clean_table 0 --trigram_clean_table R`，18 点
- R 点集同 §33：`16000,22000,…,2347000`；1000 steps；val_steps 337,674,1000。

| run_id 模式 | 数量 | 状态 | 结果目录 |
|---|---:|---|---|
| `s1v5_128_tbl_bi1_R{16000..2347000}` | 18 | ✅ done；gap@1000 = 0.143–1.152；loglog slope **0.429**（R²=.976） | `data/runs_scaling/` |
| `s1v5_128_tbl_tri1_R{16000..2347000}` | 18 | ✅ done；gap@1000 = 0.138–3.617；loglog slope **0.658**（R²=.995） | `data/runs_scaling/` |

### 34.2 Epoch-length 单表轴（trigram-only）

科学问题：固定 3 个完整 epoch 下，epoch 长度对 trigram-only 单表 gap 的
影响（用户指定优先只开 trigram）。12 个 L4 倍数点同 §33，外加 L4
10-epoch 长训 trigram-only 与 no-gram 对照。

| run_id 模式 | 数量 | 状态 | 结果目录 |
|---|---:|---|---|
| `s1v5_128_ep_tri_{mult}xL4_3ep` | 12 | ✅ done；U 形：1.0×L4 最低 2.469，0.125×=3.552、2.0×=5.582 | `data/runs_scaling/` |
| `s1v5_128_ep_tri_1xL4_10ep` | 1 | ✅ done；gap@3370 = 8.675 | `data/runs_scaling/` |
| `s1v5_128_ep_tri_1xL4_10ep_nogram` | 1 | ✅ done；gap@3370 = 0.455 | `data/runs_scaling/` |

### 34.3 调度与 stop rule

360-1 跑 `table_size_bi1`、360-2 跑 `table_size_tri1`（各 8 卡，run_id 不
重复）；完成后两机跑 `epoch_length_tri`。create-only 输出；partial 目录
停止并人工检查；不覆盖 §33 双表 run（保留作历史对照）。

---

## §35 · V5 标准 table LR 切到 128× 的全量标准实验重刷批（用户 2026-08-29 拍板）

**决策**：用户 2026-08-29 拍板「所有标准 setting 的 table LR = 128×」
（实际 `0.0006×128=0.0768`）。依据是 §31/§32 LR 扫描：step-1000 gap 随
table LR 单调升、在 128× 附近达峰（~2.73），256× 以上略回落。SSOT
`agents.md` §1.0/§1.1 已更新为 128×；`run_v5_clean.sh` 默认
`NGLAB_TABLE_LR_SCALE=128.0`（旧 2× 可用 env 覆盖复现）。

**范围**：凡「当前标准 setting」的 v5 实验线全部重刷为 128×，run_id 加
`_128x` 后缀（不覆盖旧 2× 证据）。S1 三轴批本就以 128× 运行，无需重刷。
旧 2× run 保留为历史证据并标记 superseded。

**固定完整契约**：vanilla nanoGPT 8L/6H/768D；input 注入（除非注明）；
clean 双表 R=2²⁰；backbone AdamW `(0.8,.95)`、wd `.1`、LR `.0006`、
`warmup_constant(100)`；table RMSProp 无动量 `(0,.99)`、**table LR
scale=128**；fixed replay；seed 42；bf16、不 compile；val/frequency/
exact-frequency/table RMS 每 10 步；gap = 同一步 fixed val − 当前 batch
online train。

**代码身份**：仅改 launcher（`run_v5_clean.sh` 默认 scale、新增
`run_v5_128x_rerun.sh`），未改训练代码。`train.py=c4729b30…`、
`ngram_freq.py=e4f45f5b…`。

### 35.1 M2 注入点消融（128×，2000 步）— ✅ 完成 2026-08-29

| run_id | 状态 | final train | final val | final gap |
|---|---|---|---|---|
| `nglab1x_input_v5_128x_freq10` | ✅ done | 1.259 | 6.930 | **5.672** |
| `nglab1x_y_v5_128x_freq10` | ✅ done | 1.203 | 6.451 | **5.248** |
| `nglab1x_v_v5_128x_freq10` | ✅ done | 0.385 | 8.033 | **7.648** |
| `nglab1x_nogram_v5_128x_freq10` | ✅ done | 3.121 | 3.348 | **0.227** |

对比 2×（§22 历史）：input 5.74→5.67（≈不变）；y 3.64→5.25、v 2.01→7.65
（高 LR 显著放大后端注入的 gap）；nogram 0.25→0.23（≈不变）。即
**128× 下注入点越靠后 gap 越大**，与 2× 时代「input 最大」的排序相反。

### 35.2 M5 剂量扫描（128×，2000 步，11 点）— ✅ 完成 2026-08-29

| run_id | final gap | | run_id | final gap |
|---|---|---|---|---|
| `nglab0_25x_input_v5_128x_freq10` | **10.895** | | `nglab3x_input_v5_128x_freq10` | **0.835** |
| `nglab0_5x_input_v5_128x_freq10` | **9.160** | | `nglab4x_input_v5_128x_freq10` | **0.640** |
| `nglab0_75x_input_v5_128x_freq10` | **7.207** | | `nglab5x_input_v5_128x_freq10` | **0.355** |
| `nglab1_5x_input_v5_128x_freq10` | **3.652** | | `nglab6x_input_v5_128x_freq10` | **−0.087** |
| `nglab2x_input_v5_128x_freq10` | **2.306** | | `nglab8x_input_v5_128x_freq10` | **−0.055** |
| `nglab2_5x_input_v5_128x_freq10` | **1.837** | | | |

gap 随剂量单调下降：小剂量（0.25x–0.75x）在 128× 下严重过拟合
（val 8–11），高剂量（6x/8x）gap 转负、逼近 nogram 对照。

### 35.3 Causal 干预（128×，1000 步，6 臂登记）— ⚠️ mask 两臂为旧 f>200 语义，待重刷

> 2026-08-29 更新：两种过强的早期干预按用户决定直接移除、不再登记。
> 机制证据只保留 `hash_reseed` 与互补 `mask_low/high`；freeze 只作写入路径参考。
> **2026-08-29 晚语义修正**：`mask_high` 边界改为 `f ≥ t`，`mask_low` 相应为
> `f < t`（含 novel）。下表 mask 两臂的数值是旧 `f>200` / `f≤200` 语义，需按
> 新语义重刷后更新；hash_reseed 与 freeze 两臂不受影响。

| run_id | final gap | 语义 |
|---|---|---|
| `causalv5c_none_128x` | **2.724** | 无干预基线 |
| `causalv5c_freeze_table_e1_128x` | **3.452** | e1 停止表更新 |
| `causalv5c_freeze_backbone_e1_128x` | **1.230** | e1 停止 backbone 更新 |
| `causalv5c_hash_reseed_e1_128x` | **1.354** | e1 仅重映射 context→row |
| `causalv5c_mask_low_f200_e1_128x` | 0.101（旧 f≤200） | 屏蔽 f<200 的 n-gram 输出（待重刷） |
| `causalv5c_mask_high_f200_e1_128x` | 2.808（旧 f>200） | 屏蔽 f≥200 的 n-gram 输出（待重刷） |

解读（量级参考）：low-freq 屏蔽几乎抹掉 gap（0.10），high-freq 屏蔽几乎不变
（2.81≈基线）→ **gap 主要由低频率 context 的表记忆贡献**。freeze_table 反而
升 gap（3.45），说明表仍在被 backbone 补偿。mask 两臂待按 `f≥200 / f<200`
语义重刷后正式登记。

### 35.4 X2 表行宽（128×，1000 步）— ✅ 完成 2026-08-29

| run_id | final gap | | run_id | final gap |
|---|---|---|---|---|
| `ctbl_dim12_input_v5_128x` | **0.180** | | `ctbl_dim192_input_v5_128x` | **1.458** |
| `ctbl_dim48_input_v5_128x` | **0.552** | | `ctbl_dim768_input_v5_128x` | **2.742** |

gap 随表行宽单调上升，768D（=全宽）达 2.74，接近 1000 步基线 2.72。

### 35.5 X1 表优化器（128×，1000 步）— ✅ 完成 2026-08-29

| run_id | final gap |
|---|---|
| `optv5c_rms_s128x` | **2.727** |
| `optv5c_adamw_s128x` | **2.731** |
| `optv5c_sgd_m0_s128x` | **0.047** |

RMSProp 与 AdamW（均 128×）几乎相同；SGD 无动量几乎不学（0.05）→
**128× 下优化器选择对 gap 不敏感（只要带自适应步长）**。

### 35.6 调度与 stop rule

ophis-gpu 6 张空闲卡（GPU 2/3/4/5/6/7）排队；360-1/360-2 待 VPN 恢复后
加入。create-only 输出；partial 目录停止并人工检查；不覆盖旧 2× run。

### 35.7 mask_high 阈值扫描（128×，1000 步，epoch 2 边界）— ⚠️ 旧 f>t 语义，待按 f≥t 重刷

设计依据（用户 2026-08-29）：causal 干预收敛为两个干净 setting —— `hash_reseed`
（只换 context→row 映射，保留表权重与 optimizer state）与 `mask_low/high_freq`
（按 train-shard static frequency index 屏蔽 residual）。两种早期过强干预已从
当前代码、登记和图表中删除。

**2026-08-29 晚语义修正**：`mask_high` 边界从 `f > thr` 改为 **`f ≥ thr`**（含边界）；
`mask_low` 相应为 `f < thr`（含 novel f=0）。本小节所有已完成 run 均为旧 `f > thr`
语义，需按新语义重刷后重新登记；下方数值仅作量级参考，不作为正式证据。

mask_high 阈值扫描：`--intervention mask_high_freq --intervention_epoch 1`，
threshold 从高到低取
`12800, 6400, 3200, 1600, 800, 400, 200(已有 f200 复用), 100, 50, 25, 10, 5, 2, 1`。
语义：边界后屏蔽 `f ≥ thr` 的 n-gram 输出；thr 越低屏蔽越多，扫描止于
`thr=1`，因为 novel（`f=0`）context 不属于 high 模式的 seen-context 集合。
目的：定位 gap 贡献的频率段临界点（低频贡献假说下，gap 应在 thr 降到低频区时才骤降）。

run_id：`causalv5m_mask_high_t{thr}_e1_128x`（f200 复用 `causalv5c_mask_high_f200_e1_128x`）。
全部 input 注入、R=2^20 双表、RMSProp(0,0.99)、128×、warmup_constant(100)、
bf16 no-compile、freq_index=本地 `freq_index.npz`（train shard 1）SHA256
`763a5548...7673d`。启动队列见 `code/cluster/run_v5_128x_rerun.sh` GROUP=maskhigh。

结果（旧 `f>thr` 语义，step 1000；点为 raw final gap，t=200 复用已完成 causal arm；
**待按新 `f≥thr` 语义重刷**）：

| threshold t | 12800 | 6400 | 3200 | 1600 | 800 | 400 | 200 | 100 | 50 | 25 | 10 | 5 | 2 | 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final gap | 2.755 | 2.767 | 2.783 | 2.754 | 2.762 | 2.802 | 2.808 | 2.762 | 2.686 | 2.600 | 2.334 | 2.167 | 1.929 | 1.757 |

图：`docs/figs/main/fig_v5_128x_mask_high_threshold_scan.png`；脚本：
`docs/plot_scripts/plot_v5_mask_high_threshold_scan.py`（图内语义已更新为 `f ≥ t`，
但当前曲线仍由旧 run 数据绘制，重刷后需重新生成）。阈值大于 100 时 gap
基本维持在 2.75–2.80，阈值降至 100 及以下后才开始下降。mask 是 context 级
屏蔽：high 模式只屏蔽训练集 seen context，**novel（f=0）context 永不屏蔽**，
故扫描止于 t=1。t=0 已改为明确屏蔽所有 context（包括 novel），需要重新运行；
旧 t=0 产物不再有效、不登记。全量屏蔽只消除 n-gram 贡献，gap 仍是 fixed
validation 与当前 train batch 的分布差，预期接近同预算 no-gram 对照而非严格为 0。
t=200 复用臂的配置完整记录在其
`summary.json.config` 中，其他扫描臂另有 `config.json`。

### 35.8 Causal refresh II：双边界 reseed、mask_low f≤t 扫描、mask_high f≥t 全量重刷（128×，2026-08-30）

用户 2026-08-30 指示：① `hash_reseed_e1` 在 epoch-3 没有再次 reseed、gap 回升，
要加一个 epoch-2+epoch-3 双边界 reseed 臂；② freeze_table / freeze_backbone
都仍然 fork，现象要做专图；③ mask_low 增加 `f≤t`（t=0,1,2,4,8）小阈值扫描、
10 步评估看轮廓变化；④ mask 必须同时在 train 和 eval 生效（代码事实：mask
是 model state，在 forward 内生效，val 与 exact-freq 诊断自动同步屏蔽；
low 模式下 novel f=0 恒被屏蔽）；⑤ mask_high 扫描按新 `f≥t` 语义全量重刷。

代码支撑（commit `8f7d198`，train.py）：
`--intervention_epochs 1,2`（多边界触发，各边界恰好一次）；
`--intervention_low_inclusive 1`（mask_low 从 `f<t` 改 `f≤t`，novel 恒屏蔽）；
summary 记录 `epochs_due / low_inclusive / 逐事件 frequency_mask`。
CPU 语义测试通过（mask 互补、novel 行为、双 reseed 保权重换映射、屏蔽位置零梯度）；
360-2 冒烟（50 步双 reseed + 30 步 mask）通过后删除。

| run_id | 唯一变量 | 状态 |
|---|---|---|
| `causalv5c_hash_reseed_e1e2` | epoch0=1 与 epoch0=2 各 reseed 一次 | ✅ done（360-2 gpu0）final gap 0.069 |
| `causalv5m_mask_low_le0_e1` | mask f≤0（仅 eval 屏蔽 novel 的 table 读出；train 无操作） | ✅ done（360-2 gpu1）final gap 2.945 |
| `causalv5m_mask_low_le1_e1` | mask f≤1 | ✅ done（360-2 gpu2）final gap 1.578 |
| `causalv5m_mask_low_le2_e1` | mask f≤2 | ✅ done（360-2 gpu3）final gap 1.320 |
| `causalv5m_mask_low_le4_e1` | mask f≤4 | ✅ done（360-2 gpu4）final gap 1.046 |
| `causalv5m_mask_low_le8_e1` | mask f≤8 | ✅ done（360-2 gpu5）final gap 0.765 |
| `causalv5m2_mask_high_t{1,2,5,10,25,50,100,200,400,800,1600,3200,6400,12800}_e1` | mask_high `f≥t` 全量重刷（14 点，含新 t200） | ✅ 14/14 done（360-2 gpu0-4；t200 在 ophis gpu3），全部 step-1000 回填至 §35.8.1 |

#### 35.8.1 已回填结果（step-1000，seed 42，online train / fixed val）

| run | train | val | gap | 说明 |
|---|---:|---:|---:|---|
| `causalv5c_hash_reseed_e1e2` | 4.304 | 4.373 | **0.069** | 双边界 reseed 后 gap 全程塌缩；单边界 1.354 → 双边界 0.069，证实 epoch-3 重写新映射会把 gap 拉回来 |
| `causalv5m_mask_low_le0_e1` | 2.523 | 5.468 | **2.945** | 仅屏蔽 novel（f=0）的 eval 读出即超过 control 2.724：novel table 读出对 val 是净正贡献 |
| `causalv5m_mask_low_le1_e1` | 2.975 | 4.554 | **1.578** | 屏蔽 f≤1 的 train+eval 信号 |
| `causalv5m_mask_low_le2_e1` | 3.074 | 4.394 | **1.320** | 单调下降 |
| `causalv5m_mask_low_le4_e1` | 3.186 | 4.231 | **1.046** | 单调下降 |
| `causalv5m_mask_low_le8_e1` | 3.355 | 4.119 | **0.765** | 单调下降；f≤200 端点 ≈0.101 |
| `causalv5m2_mask_high_t1_e1` | 3.867 | 5.794 | **1.927** | 屏蔽全部 f≥1（仅留 novel 读出）；事件记录 high_counts bigram 3,541,098 = f≥1 实测 |
| `causalv5m2_mask_high_t2_e1` | 4.210 | 5.962 | **1.752** | |
| `causalv5m2_mask_high_t5_e1` | 4.362 | 6.385 | **2.023** | high_counts bigram 809,058 = f≥5 实测 ✓ |
| `causalv5m2_mask_high_t10_e1` | 4.266 | 6.556 | **2.291** | 470,543 ✓ |
| `causalv5m2_mask_high_t25_e1` | 4.057 | 6.615 | **2.558** | 235,202 ✓ |
| `causalv5m2_mask_high_t50_e1` | 3.809 | 6.537 | **2.728** | 137,843 ✓ |
| `causalv5m2_mask_high_t100_e1` | 3.532 | 6.317 | **2.786** | 77,916 ✓ |
| `causalv5m2_mask_high_t200_e1` | 3.232 | 6.056 | **2.824** | 与旧语义 t=200（f>200，2.86）接近：40,307 vs 40,518 个 bigram context 之差；40,518 ✓ |
| `causalv5m2_mask_high_t400_e1` | 3.054 | 5.848 | **2.794** | 18,339 ✓ |
| `causalv5m2_mask_high_t800_e1` | 2.801 | 5.584 | **2.784** | 5,972 ✓ |
| `causalv5m2_mask_high_t1600_e1` | 2.702 | 5.462 | **2.760** | 2,038 ✓ |
| `causalv5m2_mask_high_t3200_e1` | 2.606 | 5.358 | **2.752** | 768 ✓ |
| `causalv5m2_mask_high_t6400_e1` | 2.598 | 5.329 | **2.731** | 302 ✓ |
| `causalv5m2_mask_high_t12800_e1` | 2.568 | 5.291 | **2.722** | 107 ✓ |

mask_high 刷新批的 summary 事件记录已逐一验证：`frequency_mask.high_context_counts`
与本地 `freq_index.npz` 的 f≥t 计数完全吻合（含边界语义生效），`novel_contexts_masked=false`。

其余口径与 §35.3/§35.7 一致：input 注入、clean 双表 R=2^20、RMSProp(0,0.99)、
128×、warmup_constant(100)、bf16 no-compile、seed 42、1000 步、val/freq=10、
train shard 1 / freq_index.npz。旧 `causalv5m_*`（f>t）与旧 f200 双臂保留为
历史记录，不与新批混用。绘图：`fig_v5_128x_causal_losses`（七臂）、
`fig_v5_128x_freeze_forking`（新）、`fig_v5_128x_mask_low_le_scan`（新）、
`fig_v5_128x_mask_high_threshold_scan`（切 causalv5m2 数据源）。

## §36 · S1 table-size 小 R 扩展批（R 从 1e4 扫到 1e0，2026-08-29 用户拍板）

### 36.1 动机与设计

用户 2026-08-29 指出：当前 table-size 双对数轴最小只到 R=16000
（bigram K=3.54M 时 K/R 最大 221；trigram K=19.0M 时 1189），看不到
极端 collision 区间的行为。本批把 R 向下扫到 1e0，负载率 K/R 跨越
4 个多数量级，检验 gap 在极端碰撞下是否塌缩到 no-gram 水平。

- R 点集（1/3 decade 间距，13 点）：
  `10000, 4642, 2154, 1000, 464, 215, 100, 46, 22, 10, 5, 2, 1`
- `table_size_bi1_small`：`--enable_bigram 1 --enable_trigram 0
  --bigram_clean_table R --trigram_clean_table 0`
- `table_size_tri1_small`：`--enable_bigram 0 --enable_trigram 1
  --bigram_clean_table 0 --trigram_clean_table R`
- 其余与 §34.1 完全同口径：1000 steps、val_steps 337,674,1000、
  128×、warmup_constant(100)、bf16 no-compile、seed 42。
- R=1 是合法端点：哈希 `% 1` 使所有 context 映射到同一行，
  表退化为常数向量注入（无法记忆 context 特定信息）。

### 36.2 run 登记

| run_id 模式 | 数量 | 状态 | 结果目录 |
|---|---:|---|---|
| `s1v5_128_tbl_bi1_R{10000,4642,…,1}` | 13 | ✅ done 2026-08-30 | `data/runs_scaling/` |
| `s1v5_128_tbl_tri1_R{10000,4642,…,1}` | 13 | ✅ done 2026-08-30 | `data/runs_scaling/` |

调度：360-1 GPU2-7 跑 bi1_small，360-2 GPU0-7 跑 tri1_small；
launcher `code/cluster/run_v5_s1_three_axis.sh`（GROUP=table_size_bi1_small /
table_size_tri1_small）。结果已回填 §36.3，并已把小 R 点并入
`fig_v5_s1_table_size_loglog_clean.png`。

### 36.3 验收与结果

26/26 runs 已完成，均达到 step 1000；每个目录均有
`summary.json`、`train_log.jsonl`、`freq_bin_loss.jsonl`、
`exact_freq_loss.jsonl` 和 `table_norm.jsonl`。R=215 与 R=2 的 bigram
run 首次启动遇到 transient CUDA launch failure，已隔离 partial 目录并在
GPU2 重跑；重跑结果才是下表与图的权威来源。所有 run 均为 seed 42、
128× table LR、RMSProp `(0,.99)`、`warmup_constant(100)`、bf16、
no-compile；gap 仍为同一步 fixed-val − current-batch online train。

| physical rows R | bigram-only final gap | trigram-only final gap |
|---:|---:|---:|
| 10000 | 0.1194 | 0.0920 |
| 4642 | 0.0819 | 0.0515 |
| 2154 | 0.0611 | 0.0484 |
| 1000 | 0.0760 | 0.0447 |
| 464 | 0.0640 | 0.0271 |
| 215 | 0.0611 | 0.0230 |
| 100 | 0.0331 | −0.0317 |
| 46 | −0.0062 | 0.0337 |
| 22 | 0.0459 | 0.0619 |
| 10 | 0.0310 | 0.0131 |
| 5 | 0.0216 | 0.0228 |
| 2 | −0.0043 | 0.0056 |
| 1 | 0.0160 | 0.0182 |

结果显示，R≤约 10⁴ 后两条单表轴都落入 no-gram floor 附近
（约 `|gap|≲0.06`，并包含有限 batch 噪声）；R=1 时所有 context
共享同一行，确实失去 context-specific memory。更新后的中间窗口拟合为：
bigram `(G−0.02)∝R^0.576`（R=2e3–2e5，n=12，R²=.997），
trigram `(G−0.02)∝R^0.665`（R=1e5–9.3e5，n=8，R²=.9997）。
raw-gap 敏感性斜率为 `.501/.653`；大 R 端分别进入饱和，不能用一条
`R≥16000` 直线代表全区间。

权威数据已回填：
`docs/appendices/s1_scaling_three_axis/s1_table_size_points.csv`、
`s1_scaling_analysis.md`、`s1_scaling_fits.csv`；图为
`docs/figs/main/fig_v5_s1_table_size_loglog_clean.png`。图中实心点是原
formal grid，空心点是本次 `R=10^4…1` 扩展；细线仅为 3-point visual
connector，虚线为大 R 描述性拟合。

## §37 · 理论审计与勘误批（零 GPU 分析，2026-08-30）

无新 GPU run。全部输入为已登记 run 目录与 `data/freq_index.npz`（shard-1 exact counts）；脚本
`docs/plot_scripts/plot_v5_theory_alpha_collapse.py`、`plot_v5_interference_model.py`。

**勘误 1（dose 批次归属）**：`nglab*_input_v5_freq10` 系列 config 实测 `table_lr_scale=2.0`
（2× 时代批），此前被当作现行 dose 证据引用（0.25×=11.536，slope −1.727）。128× 权威批为
`nglab*_input_v5_128x_freq10`：0.25×=10.895 → 5×=0.355，6×/8×=−0.087/−0.055，正 gap（≤5×，
n=10）slope −1.176（R²=.899）。新增 `docs/appendices/s1_scaling_three_axis/s1_dose_points_128x.csv`。
两批均保留，引用时必须注明批次。

**勘误 2（epoch-length 轴 >1×L4 段）**：12 点 epoch-length 批共用 shard-1 频率索引；>1×L4 的
“epoch”是 shard-1 wrap-around 重放，属 pass 数点（2×L4 3-epoch 终点 5.582 ≈ 长 replay 6-pass
5.609）。旧“U 形 + 二次拟合顶点 0.41×L4”作废；≤1×L4 段（嵌套前缀，3 pass）gap 随 L 从 3.552
降至 2.469。

**新分析产物**：
- `docs/figs/main/fig_v5_pass_collapse.png`：dose 128× 批、>1×L4 wrap 点、10-epoch 长 replay 在
  「完成 pass 数」轴上折叠；6× 恰在 0.99 pass 处穿零；0.75× 的 7.9-pass 点 7.21 ≈ replay 8-pass 7.14。
- `docs/figs/theory/fig_v5_zipf_context_alpha.png` + `theory_zipf_triangle.csv`：rank–frequency 局部
  α（table-size 拟合同窗口）bigram 0.994 / trigram 0.841；朴素代数关系 γ=1−α(1−β) 需要 α=0.57/0.49
  才闭合 → 被证伪。
- `docs/figs/theory/fig_v5_interference_model_vs_data.png` + `theory_interference_scan.csv`：干涉稀释
  卷积 G(R)∝Σ_f n(f)·(f/T)·f^{−β}·f/(f+T/R)，β 取频率轴实测值，预测窗口斜率 bigram 0.567 /
  trigram 0.648（实测 0.576/0.665），并复现大 R 饱和弯头；每 branch 仅一个自由幅度参数。
- 假说登记：`docs/notes/theory/hypothesis-dilution-amplification.md`（H-DILUTE，标注为假设），
  主报告新增 §6 与图 7c/14/15。

## §38 · κ 微观起源与稀释面零 GPU 判决 + causal_dynamics 批（2026-08-30 晚）

分析（无新 GPU）：
- `plot_v5_missing_mass_kernel.py`：H-KAPPA 两分量核 κ=B·M(f)+V·S_eff/f，bigram linR²=.928
  （B=4.17,V=3.40）；均场与 Poisson 行内竞争稀释形式在 f=1 处被数据否定。
  产物 `fig_v5_missing_mass_kernel.png`、`theory_missing_mass_bigram.csv`。
- `plot_v5_dilution_surface.py`：62 个单表 bigram run 的经验稀释面 s_emp(f,R)（ref R=2.35e6）；
  压制近乎 f-平坦、由 K/R 决定，load 10→1000 斜率 ≈ −0.59 ≈ R 轴实测 0.576。
  产物 `fig_v5_dilution_surface.png`、`theory_dilution_surface.csv`。
- 详细结论与降级措辞见 `docs/notes/theory/hypothesis-dilution-amplification.md` v2 节。

新 GPU 批（360-2，V5_GROUP=causal_dynamics，run_v5_clean.sh 128× 标准，2022 步=6 pass，
input 双表 1x，val=2,…,10,6542，freq10）：
| run_id | 单变量 |
|---|---|
| causalv5m3_none_2022 | 对照 |
| causalv5m3_freeze_backbone_e1/e2/e3 | epoch 1/2/3 边界冻结 backbone（A(t) 形状）|
| causalv5m3_freeze_table_e2 | epoch 2 边界冻结表（补 e1@1000 已有点）|
| causalv5m3_wd0 / causalv5m3_wd03 | backbone weight decay 0.0 / 0.3（A 饱和预测）|
状态：✅ done（2026-08-31 回填；7/7 均到 step 2022，seed 42）。远端权威 artifact：
`360-2:~/ngram-gap-lab/data/runs_fixed/<run_id>_fixed/`；逐 run 已核对 `summary.json`、
`train_log.jsonl`、`freq_bin_loss.jsonl`、`exact_freq_loss.jsonl`、`table_norm.jsonl`，并确认
128×、RMSProp `(0,0.99)`、2022 steps 与 intervention event。

| run_id | 实际事件 | train @2022 | fixed val @2022 | gap @2022 |
|---|---:|---:|---:|---:|
| `causalv5m3_none_2022` | none | 1.218 | 6.951 | **5.733** |
| `causalv5m3_freeze_backbone_e1` | step 338 | 3.279 | 4.865 | **1.585** |
| `causalv5m3_freeze_backbone_e2` | step 675 | 2.407 | 4.912 | **2.505** |
| `causalv5m3_freeze_backbone_e3` | step 1012 | 1.876 | 5.264 | **3.389** |
| `causalv5m3_freeze_table_e2` | step 675 | 1.180 | 6.566 | **5.386** |
| `causalv5m3_wd0` | wd 0.0 | 1.168 | 7.001 | **5.833** |
| `causalv5m3_wd03` | wd 0.3 | 1.284 | 6.893 | **5.609** |

**窄结论**：freeze-backbone 的终点 gap 随冻结时刻 e1→e3 单调上升，直接支持 backbone
读出能力跨 pass 累积；e2 后 freeze-table 仍保留对照的 93.95%，说明后半程不依赖继续写表。
wd 方向为 0.0 > 0.1 > 0.3，但只有单 seed 且相对对照差值约 ±0.12，只记弱支持，不据此声称
已观测长期平台。
CPU job（360-2）：`compute_fourgram_missing_mass.py` 产 trigram 支路 M(f)
→ `theory_missing_mass_trigram.csv`（回传后并入 H-KAPPA 图）。
混匀（pass 交错 replay）实验：规格=同 shard-1 tokens×3 pass 全局 sample 级 shuffle vs 分块 replay，
trigram-only 1011 步对照一对；需 `--replay_mix_passes` 新旗标（train.py 迭代器改动，改后新 run_id），
登记 planned，下一批实现。

## §39 · V5 backbone LR 扫描 · A 因子判决批（blrv5，2026-08-30 深夜，用户拍板）

> **2026-08-31 硬勘误**：本节原来把“固定 `table_lr_scale=128`”误写成
> “固定 table LR”。实际实现为
> `table_lr(t) = backbone_lr(t) × table_lr_scale`，所以六个点的绝对 table LR
> 同时从 0.0128 扫到 0.512。以下数据仍是有效的**耦合 LR 扫描**，但撤回
> “差异全部来自 backbone”“高 LR 回落证明 backbone 读出受损”等因果裁决。
> 纯 backbone 判决由绝对 table LR 锁定为 0.0768 的 §42 承担。

**科学问题**：H-DILUTE v2 把 gap 的后期增长归到 backbone 读出放大 A(t,passes)。直接判决：
固定 table LR ×128 与全部表侧参数不动，只扫 backbone `--lr`——若 net gap（input − nogram）随
backbone LR 变化，A 因子成立；若纹丝不动，A 解释被证伪。

**契约**：`run_v5_clean.sh` 128× 标准（input 双表 R=2²⁰、RMSProp (0,0.99)、×128、warmup_constant(100)、
bf16 不 compile、seed 42、1000 步≈3 pass、train shard 1、val 2,…,10,6542、val/freq=10）。
单变量 = `--lr` ∈ {0.0001, 0.0003, 0.0006, 0.001, 0.002, 0.004}；nogram 臂加
`--enable_bigram 0 --enable_trigram 0`。机器 360-1（train.py 已同步 md5=b5910b0b…，
run_v5_clean.sh=1c1ad9c2…，与本地 commit 一致）。⚠️ 360-1 GPU7 初始化 CUDA 错误复发（§31 同款硬件问题），
`blrv5_input_lr0p0040`/`blrv5_nogram_lr0p0001` 改排在 360-2 GPU7（执行器 md5 同 1c1ad9c2…）。
中心点 0.0006 与 `s1v5_128_frequency_main` 互为一致性检查（不同批同 setting）。

**预登记预测（假设）**：net gap 随 backbone LR 单调上升后趋于饱和/回落（A∝累计 backbone 更新，
过高 LR 也会抬高 nogram 自身 gap）；nogram gap 随 LR 单独演化，由对照臂剥离。

| run_id | --lr | 臂 | 状态 | gap @1000 |
|---|---|---|---|---|
| `blrv5_input_lr0p0001` / `blrv5_nogram_lr0p0001` | 0.0001 | input / nogram | ✅ done（input 360-1 / nogram 360-2） | 1.9754 / 0.0355 |
| `blrv5_input_lr0p0003` / `blrv5_nogram_lr0p0003` | 0.0003 | input / nogram | ✅ done | 2.5129 / 0.0273 |
| `blrv5_input_lr0p0006` / `blrv5_nogram_lr0p0006` | 0.0006 | input / nogram | ✅ done | 2.7179 / 0.0222 |
| `blrv5_input_lr0p0010` / `blrv5_nogram_lr0p0010` | 0.001 | input / nogram | ✅ done | 2.8525 / 0.0112 |
| `blrv5_input_lr0p0020` / `blrv5_nogram_lr0p0020` | 0.002 | input / nogram | ✅ done | 2.8338 / 0.0409 |
| `blrv5_input_lr0p0040` / `blrv5_nogram_lr0p0040` | 0.004 | input / nogram | ✅ done（input 360-2 / nogram 360-1） | 2.2950 / 0.0176 |

**结果（2026-08-31 回填）**：net gap = input − nogram：

| backbone lr | 1e-4 | 3e-4 | 6e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|---|---|
| net gap | 1.940 | 2.486 | 2.696 | **2.841（峰）** | 2.793 | 2.277 |

**对预登记预言的裁决**：
1. ⚠️ **因果裁决撤回**：net gap 的 1.94→2.84→2.28 是可靠描述性结果，但 backbone LR
   与绝对 table LR 同时变化，不能据此拆分 A 因子与表写入/过冲。
2. ⚠️ nogram gap 全程平坦（0.011–0.041），说明回落位于 input 路径；但它既可能来自
   backbone 读出动力学，也可能来自绝对 table LR=0.512 时的表侧过冲，现有批次不可辨识。
3. lr=1e-3 峰值 2.841 比主线 6e-4（2.696）高 +5.4%（单 seed，不据此改标准；6e-4 为
   vanilla nanoGPT 沿用值，改标准须用户拍板）。6e-4 点与 `s1v5_128_frequency_main` 同
   setting 跨批一致性检查通过（2.718，量级吻合）。
4. provenance：`run_v5_clean.sh` 不写 config.json，参数核验依赖启动时 `ps` 检查（5 臂 --lr
   覆盖生效）+ summary.json（steps=1000, seed 42）+ nogram 臂 gap≈0 交叉验证开关生效；
   360-1 上 GPU7 失败遗留的 2 个空壳目录（input_lr0p0040/nogram_lr0p0001）保留未删（P5）。

## §40 · V5 epoch 长度轴修复批（epfx，2026-08-31，用户拍板）

**背景与 bug 裁定**：旧 >1×L4 点（`s1v5_128_ep*_tri_{1p25,1p5,1p75,2p0}xL4_3ep`）作废——
`run_v5_s1_three_axis.sh` 的 `run_one` 把 train shards 硬编码为 `1`，而
`--epoch_batches 421..674` 超出 shard-1 池（337 batches）⇒ dataloader 在 shard-1 内
wrap-around 重复消费（用户诊断确认）。修复 = 扩大 train 池，使每 epoch 恰好一遍。

**协议**：trigram 单支路（`--enable_bigram 0 --enable_trigram 1`，trigram R=2²⁰，
bigram 关），table ×128，seed 42，3 epochs，val/freq=10；launcher `run_v5_clean.sh`
（按 shard 列表自动绑定 train-set 频率索引：`1,2→train2x_fine`、`1,2,3→train3x`、
`1,2,3,4→train4x`，与 dose 刷新批同索引）。**epoch_batches 取实际池大小**
（train.py 逐 shard floor 后求和，与 dose 日志对账：shard 批数 337/333/330/330）：
2 shards=670、3 shards=1000、4 shards=1330 batches；steps = 3×epoch_batches。
val 固定 `5,6,7,8,9,10,6542`（与全部 train 池不重叠；≤1×L4 段用旧 val 集，
段边界存在 val 组成切换，图注注明）。名义倍率按实际 batches/337 折算：
1.99×/2.97×/3.95×L4。小数倍率（1.25/1.5/1.75×）不重做：池>epoch 会引入
epoch 内部分重消费，语义不干净，旧点作废即可。输出 `data/runs_scaling/`（s1 轴命名空间）。
机器 360-1 GPU 3/4/5（shard 1–4 与三个频率索引均在位；train.py/launcher md5 未改动）。

| run_id | train shards | epoch_batches | steps | final gap @3ep | 状态 |
|---|---|---|---|---|---|
| `s1v5_128_epfx_tri_2p0xL4_3ep` | 1,2 | 670 | 2010 | **1.9548** | ✅ done |
| `s1v5_128_epfx_tri_3p0xL4_3ep` | 1,2,3 | 1000 | 3000 | **1.5192** | ✅ done |
| `s1v5_128_epfx_tri_4p0xL4_3ep` | 1,2,3,4 | 1330 | 3990 | **1.2675** | ✅ done |

**结果判决（2026-08-31 回填）**：>1×L4 段 gap 1.955 → 1.519 → 1.267，延续 ≤1×L4 段
（3.552 → 2.469）的**全程单调下降**；旧 wrap 批的"U 形增长"消失，确认纯属 artifact，
预登记假设成立。图 7（`fig_v5_s1_epoch_length_valid.png`）已并入 11 点全程版
（`plot_v5_epoch_length_valid.py`，段边界 1×L4 处标注 val 组成切换）；主文档 §5 勘误框
同步改为"已闭环"。train/val：2×L4 = 4.136/6.091，3×L4 = 4.209/5.728，4×L4 = 4.184/5.451
（val 随池增大而降低是数据多样性效应，与 gap 下降同向）。

---

## §41 · V5 20-epoch 长 replay + 二次 reseed 判决批（2026-08-31，用户拍板）

**动机**：① 主文档图 8（gap–epoch 编号）只有 3 个点（337/674/1000 步），10ep 递推的
"平台 13.58" 纯外推未验证——直接把 replay 拉到 20 个 epoch，披露长程渐进行为；
② §6 干预阵列缺关键臂：hash reseed 在 e2 击落后，e3 **再次** reseed——检验
"重对齐是否可再学、二次 reseed 是否再击落 gap"（用户指出的判决性实验）。

**设置**：360-2 GPU 0–4（全机空闲）；launcher `run_v5_clean.sh`（md5 未动，128× 写死）；
train shard 1、val `2,3,4,5,6,7,8,9,10,6542`（与 m3/三轴批一致）；seed 42。

| run_id | GPU | steps | 唯一变量 | 状态 |
|---|---|---|---|---|
| `s1v5_128_ep_tri_1xL4_20ep` | 0 | 6740 | trigram 单支路 R=2²⁰，`--epoch_batches 337`，val 每 337 步边界 | ✅ done（final gap 15.552） |
| `s1v5_128_ep1xL4_20ep_both` | 2 | 6740 | 双表（主线口径），同上 | ✅ done（final gap 12.674） |
| `s1v5_128_ep1xL4_20ep_nogram` | 1 | 6740 | nogram 对照，同上 | ✅ done（final gap 1.012） |
| `causalv5m3_hash_reseed_e2` | 3 | 2022 | `--intervention hash_reseed --intervention_epoch 2`（m3 批缺的单 reseed 臂） | ✅ done（final gap 4.202） |
| `causalv5m3_hash_reseed_e2e3` | 4 | 2022 | `--intervention hash_reseed --intervention_epochs 2,3`（二次 reseed） | ✅ done（final gap 2.881） |

**预登记判据**：20ep——若 net gap（tri−nogram）在 e>10 后进入平台，则单状态递推
\(a_e=a_\infty(1-q^{e-1})\) 获支持且平台为实测；若继续近似线性增长，则递推模型
缺慢变量，图 8b 的"平台"说法撤回。reseed——若 e2e3 臂在 e3 边界出现第二次幅度
相近的击落，说明 re-alignment 可重复学习；若第二次击落显著更小/无，说明
backbone scar 承担了不可恢复的读出成分。



### 41.1 回填与判决（2026-08-31 22:20 CST）

五 run 全部完成（6740 / 2022 步，summary.json 齐），360-2 数据已核对。

**20-epoch gap 轨迹**（epoch 边界点，gap = fixed val − online train）：

| epoch | tri 单支路 | 双表 | nogram |
|---|---|---|---|
| 1 | −0.043 | −0.062 | −0.050 |
| 3 | 2.516 | 2.730 | −0.029 |
| 6 | 5.596 | 5.681 | 0.186 |
| 10 | 8.611 | 8.988 | 0.479 |
| 15 | 12.406 | 11.088 | 0.779 |
| 20 | **15.552** | **12.674** | **1.012** |

- **判决一（递推模型）**：未进平台——tri 单支路 e19→20 增量仍 +0.57，双表 +0.24；
  但增量从 e2–3 的 ~1.5 单调衰减，是**亚线性增长**而非平台也非线性。预登记的
  「平台 13.58」外推说法**撤回**：单状态递推 \(a_e=a_\infty(1-q^{e-1})\) 缺慢变量。
  tri 净 gap（tri−nogram）e20 = 14.54 仍在增长。
- **nogram 自身**：e4 起 gap 转正，20 epoch 缓涨到 1.012（增量 ~0.05/epoch 近恒定）；
  **val 在 e6（3.323）后单调上升至 e20 3.564**——backbone 也有慢速过拟合，
  多 epoch replay 对 val 的伤害不是表独有。
- **判决二（二次 reseed）**：对照（同口径 input 128×，2022 步）gap 5.672；
  e2 单次 reseed → 4.202（击落 −1.47）；e2e3 二次 reseed → 2.881（再击落 −1.32）。
  **两次击落幅度相近 ⇒ 重对齐可重复学习**，支持表内容逐 epoch 重建的图景；
  残存 2.88 = backbone scar + 当 epoch 内重建的记忆。

---

## §42 · 纯 backbone LR 动力学：绝对 table LR 锁定批（blrabs，2026-08-31）

### 42.1 混淆修复与科学问题

§39 固定的是倍率 128，而代码实际使用

\[
\eta_T(t)=s_T\,\eta_B(t).
\]

因此扫 backbone `--lr` 时，table 的绝对 LR 与 warmup 轨迹也同时改变。本批固定
\(\eta_T^{\max}=0.0768\)，逐点反算

\[
s_T=\frac{0.0768}{\eta_B},
\]

使整个 warmup 中都满足相同的绝对 table-LR 轨迹，只留下 backbone LR 一个训练变量。
科学问题有两个：① §39 的高 LR 回落在去混淆后是否仍存在；② 不同 LR 只是改变达到
同一准平衡的速度，还是连准平衡位置也改变。

### 42.2 完整实验契约

- 主臂：input 注入、bigram+trigram clean 单表，R 均为 2²⁰；matched 对照关闭两张表。
- train shard 1；fixed val shards `2,3,4,5,6,7,8,9,10,6542`；`epoch_batches=337`。
- seed 42；bf16；no compile；backbone AdamW `(0.8,0.95)`、wd 0.1；
  table RMSProp 无动量 `(0,0.99)`；`warmup_constant(100)`。
- 绝对 table LR 恒为 0.0768；val / freq-bin / exact-freq / table-norm 全部每 50 步。
- launcher：`code/cluster/run_v5_backbone_lr_abslock.sh`；三队列：
  `code/cluster/queue_v5_backbone_lr_abslock.sh {2,4,5}`。
- 输出：`data/runs_fixed/<run_id>_fixed/`；gap 仍是同一步 fixed-val − 当前 batch online train。

| backbone LR | table scale | input / nogram run_id | steps | 机器/GPU | 状态 |
|---:|---:|---|---:|---|---|
| 0.00006 | 1280 | `blrabs_{input,nogram}_lr0p00006_tlr0p0768_3k` | 3000 | 360-2 GPU1 / GPU2 | done / done |
| 0.0001 | 768 | `blrabs_{input,nogram}_lr0p0001_tlr0p0768_3k` | 3000 | ophis GPU0 / 360-2 GPU0 | done / done |
| 0.0003 | 256 | `blrabs_{input,nogram}_lr0p0003_tlr0p0768_20k` | 20000 | ophis GPU2 / GPU4 | done / done |
| 0.0006 | 128 | `blrabs_{input,nogram}_lr0p0006_tlr0p0768_10k` | 10000 | ophis GPU2 / GPU4 | done / done |
| 0.001 | 76.8 | `blrabs_{input,nogram}_lr0p0010_tlr0p0768_6k` | 6000 | ophis GPU5 / GPU7 | done / done |
| 0.002 | 38.4 | `blrabs_{input,nogram}_lr0p0020_tlr0p0768_3k` | 3000 | ophis GPU5 | done / done |
| 0.004 | 19.2 | `blrabs_{input,nogram}_lr0p0040_tlr0p0768_3k` | 3000 | ophis GPU5 | done / done |

GPU2/4 各 30k steps，先跑 0.0006 再跑 0.0003 的 input/nogram 分臂；GPU5 约
36k steps，按 0.004→0.002→0.001→0.0001→0.00006 的顺序逐对运行，优先回答高 LR
回落是否为表侧混淆。总计 14 run；no-ngram 虽无表，仍保留同一调度与记录字段作 matched control。

### 42.3 预登记的定量判据

定义净 gap \(G_e=G_e^{input}-G_e^{nogram}\)。每个 LR 在 pass 2 以后拟合

\[
G_e=G_\star+(G_2-G_\star)q^{e-2}.
\]

若最小一阶读出动力学成立，则稳定 LR 区间应同时满足：

1. 以累计 backbone dose \(D_t=\sum_{i\le t}\eta_B(i)\) 为横轴后，早期曲线近似折叠；
2. \(-\log q/\eta_B\) 近似常数；
3. 各 LR 的 \(G_\star\) 在误差内相同，只是到达速度不同。

若锁定绝对 table LR 后 0.004 的回落消失，§39 的回落主要是 table 过冲；若仍回落但
\(G_\star\) 相同，是离散优化造成的速度/稳定性效应；若长期 \(G_\star\) 也系统变化，
则“唯一标准平衡位置”被证伪，模型必须引入 LR-dependent 的 backbone–table 共适应状态。
本批只在 run 完成并通过 summary、曲线连续性、参数乘积与 artifact QC 后作结论。

### 42.4 启动与首轮验收（2026-08-31 17:23 CST）

- source commits：`2afc905`（登记与两条 launcher）+ `8c260b9`（显式以 `bash`
  调用远端 0644 基础 launcher）；远端 `train.py` md5=`b5910b0b7d948ee567ca4aba7e28d8db`，
  abs-lock launcher md5=`0c96e4d1bcb9df214613cc9c6102c276`，queue launcher
  md5=`51d7c514f24ed908749e5e6fee7f1e7c`。
- 三条后台队列已启动：GPU2 PID 2183222、GPU4 PID 2183223、GPU5 PID 2183224；
  日志 `data/queue_logs/blrabs_gpu{2,4,5}.log`。14 个 run_id 均已入队。
- 首次调用暴露基础 `run_v5_clean.sh` 为 0644，所有 arm 在进入基础 launcher、创建结果目录
  和占用 GPU **之前**以 rc=126 退出；修复后重新启动，因此没有 partial run artifact 或数据污染。
- 修复后首批三臂均通过 step 50：input 6e-4 gap −0.0434、matched nogram 6e-4
  gap −0.0604、input 4e-3 gap −0.0072；GPU2/4/5 分别占用约 70/50/70 GB。
  `ps` 实参核验：6e-4×128=0.0768、4e-3×19.2=0.0768，且 val/freq/exact/table-norm
  的末次覆盖值均为 50。状态保持 running，完成前不填科学裁决。

### 42.5 全并行重排（2026-08-31，用户指示）

用户明确指出算力不缺，不应把独立 LR 臂串行排队。三条原队列父进程
`2183222/2183223/2183224` 已仅作 `SIGSTOP`，因此当时正在运行的 0.0003
input/no-gram 与 0.001 input 子进程继续训练，但父队列不会再发下一臂；没有中断、重启
或覆盖 active artifact。其余待启动 run 改为直接 launcher 并行：

- ophis GPU0：0.0001 input（已完成）；GPU7：0.001 no-gram（active）。
- 360-2 GPU0：0.0001 no-gram；GPU1/2：0.00006 input/no-gram（三臂均 active）。
- 原已 active 的 ophis GPU2/4 两个 0.0003 长臂继续运行。

重排验收时状态为 **8 done + 6 active + 0 queued**，14 个 run_id 一一覆盖且无重复。
360-2 的 `train.py/ngram_freq.py/run_v5_clean.sh` md5 与 ophis/local 权威版本完全一致；
该机没有 repo `.venv`，故显式设置 `NGLAB_PY=python3`，实测 torch
`2.13.0+cu130`、CUDA available、8 GPUs。第一次未设置 `NGLAB_PY` 的调用在基础 launcher
检查 Python 路径时退出，发生于结果目录创建和 GPU 占用之前，不产生 partial artifact。
三臂重启后 GPU0/1/2 分别占用约 50/70/50 GB，命令行实参复核 table scale
768/1280 与 absolute table LR 0.0768。最终回收必须合并 ophis 的 11 个目录与
360-2 的 3 个目录后再做 14-run QC 和拟合。

### 42.6 完成、QC 与动力学判决（2026-08-31）

14/14 run 全部到达预定终点。逐 run 核验 `summary.json`、`train_log.jsonl`、
`freq_bin_loss.jsonl`、`exact_freq_loss.jsonl`、`table_norm.jsonl`：五类文件均存在且
末条 step 等于预算；seed/data_seed=42，fixed replay，bf16/no-compile，val/freq/exact/
table-norm cadence=50，input/no-gram 开关与 clean-table 容量正确；七点均严格满足
`backbone_lr × table_lr_scale = 0.0768`。权威目录是 ophis 的 11 条与 360-2 的 3 条
远端 `_fixed` 目录。因本地 `data` 指向当时未挂载的 Extreme SSD，本次只把小型证据
暂存到 `/private/tmp/ngram-gap-lab-backbone-lr-artifacts/runs_fixed/` 作可复现分析；没有
传输合计数 GB 的 exact-frequency 日志，也没有改动悬空符号链接。

**各预算终点（seed 42）**：

| backbone LR | endpoint epoch | input gap | no-gram gap | net gap |
|---:|---:|---:|---:|---:|
| 0.00006 | 9 | 4.065 | 0.040 | **4.025** |
| 0.0001 | 9 | 4.750 | 0.095 | **4.655** |
| 0.0003 | 60 | 17.043 | 2.108 | **14.935** |
| 0.0006 | 30 | 14.485 | 1.398 | **13.087** |
| 0.001 | 18 | 12.240 | 0.779 | **11.461** |
| 0.002 | 9 | 8.556 | 0.056 | **8.500** |
| 0.004 | 9 | 0.076 | 0.041 | **0.035** |

**判决一：高 LR 回落不是 table-LR 共变造成的。** 绝对 table LR 锁定后，
`4e-3` 的 net gap 仍塌到 0.035；此时 input/no-gram 的 train loss 分别为 6.714/6.331，
val loss 为 6.790/6.372，说明这是 backbone 训练本身接近失效的优化区，不是“更快到达
一个低 gap 平衡”。因此 §39 高 LR 回落不能再归因于 table 的绝对 LR 被同步放大。

**判决二：累计 backbone dose 不足以解释 epoch 动力学。** 对 `3e-4/6e-4/1e-3/2e-3`
四点取共同可达 dose \(D=5.925\)，线性插值得：

| backbone LR | 该 dose 对应 pass | net gap |
|---:|---:|---:|
| 0.0003 | 58.72 | **14.735** |
| 0.0006 | 29.41 | **13.050** |
| 0.001 | 17.69 | **11.451** |
| 0.002 | 8.90 | **8.500** |

同 dose 的跨 LR gap 变异系数为 0.193，并且**经历更多 replay pass 的低 LR 臂 gap
系统性更大**。预登记判据 1（dose collapse）被证伪：epoch/replay 次数是独立状态变量，
不能只把它看成累计梯度步长的另一种计量单位。

**判决三：一阶递推只能作窗口内描述，不能给“标准平衡位置”。** 用每个 epoch 的
最后一个 50-step checkpoint，在 epoch 2 后拟合
\(G_e=G_\star+(G_2-G_\star)q^{e-2}\)：

| LR | q | τ(epoch) | −log(q)/LR | G★（外推） | R² | 最后观测增量 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00006 | .851 | 6.22 | 2680 | 5.54 | .9993 | +.289 |
| 0.0001 | .836 | 5.58 | 1791 | 6.11 | .9994 | +.347 |
| 0.0003 | .930 | 13.75 | 243 | 14.33 | .9880 | +.149 |
| 0.0006 | .897 | 9.16 | 182 | 13.37 | .9974 | +.176 |
| 0.001 | .880 | 7.81 | 128 | 13.06 | .9987 | +.079 |
| 0.002 | .892 | 8.72 | 57 | 14.10 | .9927 | +.747 |
| 0.004 | ≈0 | 0.09 | — | 0.03 | .0645 | +.040 |

稳定臂的 \(-\log q/\eta_B\) 跨约 47 倍，不是常数；`2e-3` 的半窗口/full-window
\(G_\star\) 更从 7.50 漂到 14.10，`3e-4` 从 12.78 漂到 14.33。所有非失败臂最后
增量仍为正，尤其最长 `3e-4` 的观测终点 14.935 已高于其 full-window 外推 14.33。
因此 R² 高只说明有限窗口内曲线平滑，**不等于观测到平台**。当前数据既不支持共享的
唯一 \(G_\star\)，也不足以断言每个 LR 各有真实平衡；更窄的结论是“一状态、LR 只改
时间常数”的模型被证伪，平台/平衡位置仍未识别。

**为什么多 epoch 会把 gap 拉得很大。** 配对恒等式把净 gap 精确拆成

\[
G_e=\underbrace{L_{tr}^{0}-L_{tr}^{table}}_{M_e\;\text{(train benefit)}}+
\underbrace{L_{val}^{table}-L_{val}^{0}}_{X_e\;\text{(validation penalty)}}.
\]

在最长 `3e-4` 臂的 epoch 60，\(M=2.119\)，\(X=12.816\)，合计 \(G=14.935\)；
**85.8% 的净 gap 来自 validation 被伤害，而不是 train loss 还能继续下降**。train
benefit 在约 10–15 epoch 已接近 2.5 后缓慢回落，但 validation penalty 继续跨 pass
累积。这与 §38 的因果证据合并后，最小机制必须至少有两个状态：

\[
z_{e+1}=F_T(z_e,\theta_e;D_{train}),\qquad
\theta_{e+1}=\theta_e-\eta_B\!\sum_{i\in e}\nabla_\theta
\ell_i(\theta_e,z_e),\qquad G_e=M(z_e,\theta_e)+X(z_e,\theta_e).
\]

\(z_e\) 是快速写入、下一 pass 可复用的 table 记忆；
\(\theta_e\) 是共享 backbone/readout。第一 pass 主要写入；从第二 pass 起，同一个
train-specific residual 再次出现，table 方向成为可预测输入，backbone 每轮都收到同向
共适应梯度。独立 val 不共享这份残差，于是后期主要表现为 \(X_e\) 累积。§38 的
freeze-table@e2 仍保留 93.95% gap，支持“table 可在后期近似成已写好的环境”；但 table
norm 仍随训练变化，所以**环境只是准静态近似，不是 table 已严格收敛**。要把主动项彻底
分开，仍需同一 e2 checkpoint 的 `control / freeze-table / freeze-backbone / freeze-both`
长程四因子（最好加 `(θ30,z2)` 与 `(θ2,z30)` cross-swap）。

复现脚本：`docs/plot_scripts/plot_v5_backbone_lr_epoch_dynamics.py`；图：
`docs/figs/theory/fig_v5_backbone_lr_epoch_dynamics.{png,svg}` 与
`fig_v5_backbone_lr_recurrence_diagnostics.{png,svg}`；逐点、拟合与 matched-dose 数据分别为
`theory_backbone_lr_epoch_points.csv`、`theory_backbone_lr_epoch_fits.csv`、
`theory_backbone_lr_matched_dose.csv`。所有数值均从 `_fixed` JSONL 读取，未手填进图。

## §43 · V5 净收益判决批（netv5，2026-08-31，用户拍板）

### 43.1 科学问题

指导大模型训练的核心问题：n-gram 模块是否**白忙活**。已有控制臂数据（128×，seed 42）
显示 val loss 全面高于 nogram——step 1000：input +1.671 / y +1.252 / v +1.617；
step 2000：+3.582 / +3.103 / +4.685。即无约束的表是纯记忆、净伤害 val。
本批检验两种**约束条件**下 val loss 能否压到 nogram 之下（净收益 = val(arm) − val(nogram) < 0）：

- **hash reseed 每个 epoch**：boundary 处重洗 context→row 映射，表内容随 epoch 重建，
  推理用最后一个 table。若高频统计的单 epoch 重建有泛化价值，val 应低于 nogram。
- **mask 低频 from start**（f ≤ 8 + novel，step 0 起）：表只学习高频 context，
  从源头禁止低频记忆。

### 43.2 代码变更（train.py md5 `d3a5a58464ac945d7963fcc804239038`，commit `f7ee97a`）

- 频率掩码从 input-only 扩展到 y/v 注入路径（per-layer bgve/tgve 上 `masked_fill`），
  解除 parser 的 input-only 限制。无掩码时 `_frequency_mask` 返回 None，默认路径不变。
- CPU 冒烟（n_embd=768/n_head=6 真实维度）：input/y/v 三位置掩码前向/反向正常；
  高频 context（key 5）三位置均正确穿透（1 行非零梯度），其余行零梯度。

### 43.3 实验契约

6 run，全部 128×、双表 R=2²⁰、train shard 1（1×L4，337 batches/epoch）、
val 2-10+6542、2000 步（=6 epoch）、val/freq/exact/table-norm 全部 interval 10、
seed 42、bf16 不 compile。唯一变量 = 注入位置 × 干预方式。

| run_id | 注入 | 干预 |
|---|---|---|
| `netv5_input_reseed_eall_128x_fixed` | input | hash_reseed @ due epochs {1,2,3,4,5}（= epoch 2–6 起点，共 5 次） |
| `netv5_y_reseed_eall_128x_fixed` | y | 同上 |
| `netv5_v_reseed_eall_128x_fixed` | v | 同上 |
| `netv5_input_masklowf8_e0_128x_fixed` | input | mask_low f≤8（inclusive）+ novel，epoch 0 起点 = step 0 起 |
| `netv5_y_masklowf8_e0_128x_fixed` | y | 同上 |
| `netv5_v_masklowf8_e0_128x_fixed` | v | 同上 |

对照（已有，不重跑）：`nglab1x_{input,y,v,nogram}_v5_128x_freq10_fixed`。
验收口径：每 10 步的 val(arm) − val(nogram matched step)；负 = 净收益。
launcher `run_v5_128x_rerun.sh` GROUP=net，360-2 GPU 0–5 一卡一 run。
状态：**done（2026-08-31 23:20 回填）**。

### 43.4 结果与判决（核心交付）

验收口径 = 每 10 步 val(arm) − val(nogram, matched step)；负 = 净收益。

**step 2000（6 epoch 末）**：

| arm | train | val | Δval | gap |
|---|---|---|---|---|
| masklow-input | 2.347 | 4.674 | **+1.326** | 2.327 |
| masklow-y | 2.224 | 4.582 | **+1.234** | 2.358 |
| masklow-v | 1.368 | 6.019 | **+2.671** | 4.651 |
| reseed-input | 4.114 | 4.156 | **+0.807** | 0.041 |
| reseed-y | 3.075 | 3.240 | **−0.108** | 0.166 |
| reseed-v | 3.021 | 3.217 | **−0.131** | 0.196 |
| control-input | 1.259 | 6.930 | +3.582 | 5.672 |
| control-y | 1.203 | 6.451 | +3.103 | 5.248 |
| control-v | 0.385 | 8.033 | +4.685 | 7.648 |
| nogram | 3.121 | 3.348 | 0 | 0.227 |

**判决**：
1. **mask_low f≤8 打不过 nogram**——但 val 伤害比无约束小 60–75%；残余伤害来自
   高频 context（f>8）的跨 epoch 记忆。低频屏蔽 ≠ 净收益。
2. **hash reseed 每 epoch：y/v 打赢、input 打不赢**。reseed-v 稳定净收益
   （e3–e6 每 epoch 中点 Δval ≈ −0.09，91% 的步为负）；reseed-y 边际净收益
   （终点 −0.108，后三 epoch 均值 −0.037，接近噪声）；reseed-input +0.807 且每个
   boundary 有 +2~+3 的 val 尖峰（input 读出对重映射极敏感，epoch 内才逐步重写回来）。
3. reseed 把 gap 杀到 0.04–0.20（对照 5.7–7.6）：**伤害的主要来源是跨 epoch 的
   记忆积累，而非低频或高频本身**；且 reseed-y/v 的 train loss（3.08/3.02）
   略优于 nogram（3.12）——不能记忆化时表提供的是真实有用的共现特征。
   这是「n-gram 不白忙活」的第一个正证据，但量级小（~−0.1）且注入位置敏感。
4. 图：`docs/figs/main/fig_v5_netval_benefit.{png,svg}`（三面板 Δval 轨迹，
   脚本 `plot_v5_netval_benefit.py`）。

## §44 · 分 branch mask 扫描 + 仅末 epoch 用表（2026-08-31 23:45，用户拍板）

### 44.0 前置事实（代码确认，§44.4）

**默认 readout 无任何预处理 mask**：`_freq_mask_index is None` 时 `_frequency_mask`
返回 None，任何 context——包括 train 从未出现的 novel（f=0）——都经 hash 映射读出行内容
（novel 读到的是碰撞行里别的 train context 写入的内容）。频率 mask 只在显式
`--intervention mask_low/high_freq`（或 readout_last_epoch）时在 boundary 启用，
一旦启用对 train forward 与 val forward 同一生效（无 train/eval 分支）。含义：
novel 的表读出**不是零**，这正是 le0（仅屏蔽 novel 读出）step-1000 gap 反升
（2.945 > 2.724）的原因——novel 行读出对 val 是净正贡献。

### 44.1 科学问题

1. **分 branch 的 mask trade-off**：trigram 静态归因 gap 摊在 f 上（f≤8 贡献 79.3%，
   质量保留仅 44%），bigram 集中（f≤8 贡献 33.8%，质量保留 87.6%）。
   「trigram mask 少量低频是否 gap 就很小」需要 trigram 单表动态实测。
2. **仅最后一个 epoch 用表**（readout_last_epoch）：epoch 1–5 全屏蔽（表零梯度、
   backbone 不见表），epoch 6 起点解除。对比 reseed-every-epoch（§43：reseed 后
   旧表内容变噪声）与 mask_low_f8——第三条约束路线。

### 44.2 实验契约

全部 128×、train shard 1、val 2-10+6542、val/freq=10、seed 42。
代码 commit（readout_last_epoch 干预 + launcher trim/lep 组）。

| run_id | setting | 预算 |
|---|---|---|
| `s1v5_128_tri_masklow_ctl` | trigram 单表对照（bigram 关），无干预 | 1000 步 |
| `s1v5_128_tri_masklowf{1,2,4,8}_e1` | 同上 + mask_low f≤t（inclusive）from epoch 2 | 1000 步 ×4 |
| `netv5_{input,y,v}_lastep_128x` | 双表 + readout 全屏蔽 epoch 1–5、epoch 6 解除 | 2000 步 ×3 |

判决口径：① tri 单表 step-1000 gap(t) vs tri 对照（动态去除率，对照静态归因 79.3%@f≤8）；
② lastep 终点 Δval vs nogram，与 reseed（y −0.108 / v −0.131）、masklowf8（+1.2~+2.7）对比。
360-2 GPU 0–7（trim GPU0-4，lep GPU5-7）。状态：**done（2026-09-01 回填）**。

### 44.3 结果与判决

**① trigram 单表动态 mask 扫描（step 1000，seed 42）**：

| arm | train | val | gap | 动态去除率 |
|---|---:|---:|---:|---:|
| tri 对照（无干预） | 4.191 | 6.676 | 2.486 | — |
| mask f≤1 from e2 | 3.834 | 4.858 | 1.023 | −58.8% |
| mask f≤2 | 3.756 | 4.611 | 0.855 | −65.6% |
| mask f≤4 | 3.744 | 4.369 | 0.626 | −74.8% |
| mask f≤8 | 3.740 | 4.155 | 0.415 | **−83.3%** |

动态去除（83.3%）陡于静态归因预期（79.3% → 残差 0.51），与双表批同向；mask 后 train
loss 单调下降（4.19→3.74），因为低频 context 不再读到碰撞行噪声、表容量集中到高频。
回答 44.1 问题 1：**trigram mask f≤8 一个变量就把单表 gap 从 2.49 压到 0.42**，
但不能压到零——残余 0.42 来自 f>8 高频 context 的记忆。

**② 仅末 epoch 用表（readout_last_epoch，step 2000 = epoch 6 末）**：

| arm | train | val | Δval vs nogram | gap | reseed 对照 Δval（§43） |
|---|---:|---:|---:|---:|---:|
| lastep-input | 4.151 | 4.170 | **+0.822** | 0.019 | +0.807 |
| lastep-y | 3.132 | 3.356 | **+0.008** | 0.224 | −0.108 |
| lastep-v | 3.167 | 3.355 | **+0.006** | 0.187 | −0.131 |

- lastep 与 reseed 同样把 gap 杀到 ≤0.22，再次确认**跨 epoch 记忆积累是 gap 主源**。
- input 臂解除屏蔽后 val 立即冲到 +2.9（step ~1700），epoch 6 内单调回落到 +0.82，
  终点与 reseed-input（+0.807）几乎相同：**input 位置的单 epoch 表读出一个 epoch 内
  就造成 ~0.8 的 val 伤害**，backbone 在 epoch 内只能部分适应。
- y/v 臂解除后 Δval 全程 ≈ 0（+0.006~0.008），无尖峰：y/v 位置的单 epoch 表对 val
  基本中性（reseed 还略负）。净收益问题因此进一步定位为**注入位置敏感**：
  input 单 epoch 也有害，y/v 中性至微收益。
- 与 reseed 的差别：lastep 的表从 epoch 6 起点空白写入（前 5 epoch 零梯度），reseed
  每个 epoch 重建；两者终点 Δval 相近说明对 val 而言关键是「表内容年龄 ≤1 epoch」，
  与表如何被重置无关。

图：`docs/figs/main/fig_v5_mask_sweep_lastep.{png,svg}`（脚本
`plot_v5_mask_sweep_lastep.py`）。数据：360-2 `runs_fixed` 远端权威目录，
本地分析镜像 `/tmp/ng_data/runs_fixed`。

## §45 · freeze 四因子长程批 + pass 混匀判决批（2026-09-01，用户「继续排任务」批）

**科学问题**：§42 判定一状态递推被推翻、模型升级为「快速 table 记忆 + 慢速 backbone
共适应」两状态后，缺的最关键拼图是四因子长程分离；另一会话指出的锋利判别是
pass 混匀（shard×3 全局 shuffle vs 分块 replay）——若终点 gap 显著变化，说明模型
缺「重复间隔」变量。

### 45.1 FFQ：freeze 四因子长程批（3 run，3370 步 = 10 epoch）

从同一 e2 边界（0-indexed epoch 2 = step 675，§38 语义）做长程四因子分离；
对照复用已有 `s1v5_128_ep1xL4_10ep_{both,nogram}`（同契约、无干预，不重跑）。

| run_id | 单变量 | 预期 |
|---|---|---|
| `ffqv5_freeze_table_e2_10ep` | e2 后冻结表 | gap 继续增长但慢于对照（§38 @2022 保留 94%）|
| `ffqv5_freeze_backbone_e2_10ep` | e2 后冻结 backbone | gap 增长大幅压低（A(t) 形状全窗）|
| `ffqv5_freeze_both_e2_10ep` | e2 后全冻结 | gap 应在边界后**完全停长**——两状态模型的硬预言 |

固定契约 = §33 长训契约 + 128× 标准：train shard 1、val 2,3,4,5,6,7,8,9,10,6542、
`epoch_batches 337`、双表 `2^20`、RMSProp `(0,0.99)` ×128、`warmup_constant(100)`、
seed 42、bf16、val/freq/exact/table-RMS 每 10 步 + epoch 边界显式 `--val_steps`。

代码改动（`code/train.py`，新 run_id 前缀 `ffqv5_`/`mixv5_`）：
1. `intervention` 新增 `freeze_both`（表 ∪ backbone 参数全部 `requires_grad=False`）；
   全冻结时图无 grad_fn，backward 按守卫跳过（优化器对 None 梯度本就 no-op）。
2. 新旗标 `--replay_mix_passes N`：一次性消费 N pass 的全局 chunk 置换流
   （chunk 级置换保 n-gram 连续性，token 级会破坏）；`_epoch` 恒 0，该臂禁用
   边界干预。CPU 单测通过：3×12 prefix 多重集相等、确定性置换、chunk 内连续性、
   有序 replay 回归不变、freeze_both 分区完备且无梯度流动。

验收：4 臂（含两个既有对照）到 step 3370，边界事件 `freeze_both` 落盘
`intervention_events`；freeze_both 臂边界后 train/val 曲线**平坦**（表 RMS 也不变）；
nogram 对照差值有限。状态：**done（2026-09-01 回填，见 45.3 附记）**。

### 45.2 MIX：pass 混匀判决批（2 run，1011 步 = 3 pass，trigram-only）

| run_id | 数据流 | 语义 |
|---|---|---|
| `mixv5_tri_replay_3pass` | 标准 fixed replay：3 个有序 epoch（337×3 步）| 分块 replay 对照 |
| `mixv5_tri_mixed_3pass` | `--replay_mix_passes 3`：同 3×24264 chunk 全局置换一次消费 | 无 epoch 结构 |

trigram 单表（`TRI_FLAGS`，同 §44 tri-only 批），val 于 337/674/1011。两臂 chunk
多重集完全相同，唯一变量 = 重复的时间结构。H-DILUTE 预言：终点 gap 不变、epoch 齿
消失；若 mixed 终点显著更低（重复间隔短→记忆弱）或更高，模型需加入重复间隔变量，
用户「熵正则/确信度」直觉获得支持。

**回填（360-2 GPU 4/5，均已完成，1011 步）**：

| run_id | step 337 | step 674 | step 1011 |
|---|---|---|---|
| `mixv5_tri_replay_3pass` | −0.0557 | +0.9901 | **+2.4738** |
| `mixv5_tri_mixed_3pass` | +0.8919 | +1.5369 | **+2.2880** |

- **epoch 齿消失**：replay 臂 step 337 gap ≈ 0（online 语义：pass 1 无记忆收益），
  边界后跳起；mixed 臂 step 337 即 +0.89 —— 全局置换流中窗口内已有重复曝光。
  「完成 pass 数」确实不是底层变量，重复曝光本身才是。
- **终点 gap −0.19（−7.5%）**：naive 预言「终点不变」近似成立但不精确。方向解释
  （假设级，待检验）：置换流中大量 chunk 两次出现间隔很短（massed repeats），此时
  行还很新鲜、第二次写入边际收益小；有序 replay 的间隔固定 337 步，行已被 336 次
  写入碰撞稀释，重写恢复更多——**间隔分布影响记忆效率**，支持把「重复间隔」加进
  状态变量（用户的熵/确信度直觉）。

启动器：`GROUP=ffq NGLAB_PY=python3 bash run_v5_128x_rerun.sh 0 1 2`（360-2）；
`GROUP=mix ... 4 5`。代码 commit 后同步 360-2 并 md5 核对。

### 45.3 零 GPU：g(f) 快照的 e·f 重标度检验（challenge ① κ(f) 微观起源）

**数据**：`s1v5_128_ep1xL4_10ep_{both,nogram}_fixed` 的 10 个 epoch 边界 exact-f
快照（337/674/…/3370 步），g(f) = val_both − val_nogram，逐 branch 逐 epoch。

**检验设计**（H-DILUTE 的 challenge ①）：若 epoch 增长主要来自 κ(f) 沿 f 轴平移
（每个 context 的有效重复次数 e·f 增加），则第 e 个快照应与 e=1 快照在水平重标度
f → e·f 后重合；若 κ(f) 形状固定、增长只是乘性放大，则垂直重标度 g → g/A_e 即可塌缩。

**判决（token 加权 log-RMSE，仅 g>0 且 f∈[8,512] 窗口，MIN_TOKENS=200）**：

| branch | 垂直重标度 RMSE（e=3→10） | 水平 e·f 重标度 RMSE | A_e（锚点起算） |
|---|---|---|---|
| bigram | 0.62–0.71（好） | 1.57→4.00（单调变差） | A_10 = 28.6（锚 e=1） |
| trigram | 0.73–0.90（好） | 3.09→4.79（单调变差） | A_10 = 66.4（锚 e=2） |

- **e·f 水平平移被彻底否定**：RMSE 随 epoch 单调恶化，κ(f) 并不沿 f 轴移动。
- **垂直乘性塌缩成立**：一个 A_e 就能把各 epoch 快照压回参照快照，κ(f) 形状在
  f 轴上稳定。
- 结论：**κ(f) 是空间形状（f 的函数），epoch 增长是纯乘性放大 A(e)**。表内容
  1 个 pass 内近饱和、后续增长来自 backbone 读出放大——直接支持两状态模型，
  同时否定「多 epoch = 每条 context 有效重复 e 次所以 κ(e·f)」的单变量图像。
- 图/CSV：`docs/figs/theory/fig_v5_epoch_kernel_rescale.{png,svg}`、
  `theory_epoch_kernel_rescale.csv`；脚本 `docs/plot_scripts/plot_v5_epoch_kernel_rescale.py`。
- 注意：log 轴上 g(f) 变号的桶呈梳齿伪影，判定只在 g>0 窗口内做，不影响结论。

**45.3 附记（§45.1 终点回填，全部 3370 步 = 10 epoch，e2=step 675 冻结）**：
- `ffqv5_freeze_both_e2_10ep`：final gap **+3.0759，且 8 个 epoch 位点 bit-exact 不变**
  （train 1.6898 / val 4.7656 每次完全相同）——两状态模型硬预言命中：无任何参数更新
  则 val 与 gap 完全冻结。注意边界处 gap 从 +1.18 跳到 +3.08 **不是 val 增长**：
  val=4.7656 与冻结值 bit-exact 相同，跳变全部来自 train 侧 online 口径——同一边界
  batch 在其自身最后一次 128× 写入前测得 3.586、写入后重放测得 1.690（单次 self-write
  对自身 batch 降 1.90 nats），正是用户早期预言的「自身 loss 结结实实下降」。
- `ffqv5_freeze_backbone_e2_10ep`：final gap **+2.7268**（train 2.2360 / val 4.9627）
  —— 仅表继续写 8 个 epoch，val 只从 4.7577 涨到 4.9627（+0.21，对照同期 +4.55）：
  **表写入本身几乎不产生 val 伤害，backbone 更新才是放大器**（晚期 val 增长 ≥95%
  需要 backbone 参与）。train 从 2.855 缓降到 2.236：冻结 backbone 下表逐 pass 收敛
  到该 backbone 的最优行。
- `ffqv5_freeze_table_e2_10ep`：final gap **+6.3991** = 对照 8.9167 的 71.8% —— 表
  内容在 e2 冻结后，backbone 单独继续适应即可带走约 3/4 的长程 gap。
- train 侧反直觉发现：freeze_table 的 online train @e3 = 1.35 **远低于**对照 2.54 ——
  冻结表反而**保住**了边界 batch 的 self-write；对照中后续 336 个 batch 的继续写入
  会通过碰撞**擦除/稀释**已写行（与 §46 实测 71.5%/94.5% context 碰撞率一致）。
- MIX 见 §45.2 回填。

## §46 · 零 GPU：offline hash collision 实测 owned token mass vs R（2026-09-01）

**动机**：H-DILUTE 的 s_c(R)=f_c/(f_c+T/R) 此前从未有过直接碰撞测量；模型用
「context 数碰撞率」外推 token 质量保留率，需实测校准。

**方法**（零 GPU，`docs/plot_scripts/offline_hash_collision.py`）：读
`data/freq_index.npz`（shard-1 exact context 计数，bigram K=3,541,098 / trigram
K=19,027,841，T=49,716,935），**用 train.py 的真实 hash（第一 hash 族，clean K=1）**
把每个 context 映到行：bigram `((prev·p1)^(cur·p2)) % R`，trigram
`((p2·q1)^(p1·q2)^(cur·q3)) % R`。对 R∈{2¹⁰,…,2²⁰} 统计：
context 碰撞率 (K−occupied)/K、solo token mass（行内仅 1 个 context）、
owner token mass（行内 dominant context 持有份额）。

**关键数字（R=2²⁰，主线 setting）**：

| branch | context 碰撞率 | solo token mass | owner token mass |
|---|---|---|---|
| bigram | 71.5% | 3.3% | **86.5%** |
| trigram | 94.5% | ≈0% | **43.7%** |

- **口径分离是全部重点**：按 context 数算碰撞很严重（trigram 94.5%），但按 token
  质量算，bigram 的 86.5% 仍由「行主」控制——gap ∝ token mass，所以高频 context
  在碰撞下依旧保有私有通道；trigram 只有 43.7%，私有通道减半，这正是 trigram 对 R
  更敏感（γ 0.665 > 0.576）的碰撞侧解释素材。
- solo（零碰撞）token mass 在主线 R 几乎为 0：**「collision rate」若用 (K−occupied)/K
  或 solo mass 口径都会严重夸大碰撞对 gap 的影响**；owner-mass 才是和 gap 同口径的量。
- owner_frac vs R 在 log-log 上近乎直线（bigram 0.14→0.86 / trigram 0.03→0.44 跨
  2¹⁰→2²⁰，斜率 ≈0.8–0.9），为 s_c(R) 的经验替代提供了直接可拟合对象——后续建模
  应以 owner_frac(R) 替换解析式 f/(f+T/R)。
- 图/CSV：`docs/figs/theory/fig_v5_hash_collision_mass.{png,svg}`、
  `theory_hash_collision_mass.csv`。注意：freq_index.npz 的 key 是 packed context id
  （prev·V+cur 等），不是表行——必须先分解回 token id 再过 hash，直接 key%R 会低估
  occupied（错把均匀 id 分布当 hash 分布）。

## §47 · fast-diag 测量管线 + 主实验四臂 fd 重刷（2026-09-01，用户拍板）

**动机**：eval block 剖析发现每 10 步的评占比 ~90% wall time（2000 步 run 77 min 中训练仅 226 s/1000 步）。
构成：36 次 fp32 诊断 forward（exact_freq 16 + shared 16 + freq_bin val 4，`compute_per_token_loss`
从不进 autocast）+ CPU 侧 `np.vectorize` dict 查找聚合。用户拍板：诊断 forward 换 bf16（与
evaluate_val 同口径）+ ptl 缓存复用 + 后台 CPU 进程异步聚合。

**代码**（commit 100619d + manifest 2c279d92；md5 已核对 ophis-gpu）：
- `ngram_freq.py`：`hit_count_numpy`（searchsorted，与 dict.get 逐 key 等价实测）、
  `ExactFreqLossAccumulator.update_numpy`、`FreqBinLossAccumulator.update_numpy`、
  `compute_shared_from_keys`（向量化，保留 last-occurrence 语义）、`compute_per_token_loss(amp_dtype=)`
- `diag_worker.py`（新）：torch-free 后台 worker（fork 继承 freq index），主进程每 eval block
  仅 8+4 次 bf16 forward，(keys, ptl) 入队；`NGLAB_ASYNC_DIAG=0` 回退同步路径
- `train.py`：统一 diag block（12 次 bf16 forward/block，原 40 次 fp32）；行尾 flush；
  summary 记录 `diag_forward_dtype`/`diag_async`
- 等价性单测（ophis CPU）：searchsorted==dict（250k keys 全等）、exact-freq/shared/freq-bin
  numpy 路径==纯 dict 参考、异步 worker==同步计算

**语义变更**：诊断量（exact-freq / freq-bin val / shared）forward 由 fp32 → bf16；
val 主指标（evaluate_val 本就 bf16）与训练路径完全不变 → 新 run_id 后缀 `_fd`。

**四臂重刷**（全部 ✅ done，ophis-gpu gpu5，seed 42，128×，2k steps，val/freq=10）：

| run_id | final train | final val | final gap @2k | 旧 fp32-diag gap | Δgap | wall time |
|---|---|---|---|---|---|---|
| `nglab1x_input_v5_128x_freq10_fd` | 1.271 | 6.947 | **5.676** | 5.672 | +0.004 | 18.7 min |
| `nglab1x_y_v5_128x_freq10_fd` | 1.179 | 6.507 | **5.328** | 5.248 | +0.080 | 11.9 min |
| `nglab1x_v_v5_128x_freq10_fd` | 0.355 | 8.161 | **7.806** | 7.648 | +0.158 | 11.5 min |
| `nglab1x_nogram_v5_128x_freq10_fd` | 3.108 | 3.340 | **0.233** | 0.227 | +0.005 | 10.3 min |

**曲线等价性**（`docs/figs/main/fig_v5_fd_vs_old_curves.png`）：新旧 train/val/gap 曲线全程重叠；
逐点 Δ 从 step ~300 起混沌放大（FP 噪声经训练动力学放大，GPU matmul 算法选择对内存布局敏感），
幅度始终在批间噪声尺度（±0.1–0.4）内；10 段均值 Δ≈0。最终 Δgap（最大 0.158 @v）
在历史 seed 复现差（±0.16）之内。结论：**训练动力学等价，现象与判决不变**。

**诊断量 bf16 效应**：freq-bin val trigram 各桶 mean_loss 新旧差 ~0.02–0.06（相对 ~0.3–0.8%），
低频桶略大，符合 bf16 精度预期；旧 `_fd` 前数据仍可作 fp32 诊断参考，不建议跨后缀混画单条曲线。

**提速**：eval block ~21 s → ~1.5 s（-93%）；2000 步 run 77 min → 10–19 min（4–7×）。
后续所有 freq=10 run 默认走该管线。

## §48 · 零 GPU：gap-vs-pass 动力学分解——推翻「表一次写完 / backbone 逐 pass 衰减写入」（2026-09-01）

**问题**（用户）：gap 为什么和 pass 数绑定？「pass」定义要明确；「backbone 逐 pass 衰减写入」听起来是 trivial 时间漂移。
**定义**：pass e = 对固定 train shard 的第 e 次完整顺序遍历（1x = 337 device steps，`fixed` replay，batch 顺序每 pass 相同）。
pass e 内任意 step 的 online batch 恰有 e−1 次先前曝光，且距上次曝光恒为 337 步（staleness 常数，不构成 pass 内 confound）。
`theory_backbone_lr_epoch_points.csv` 的 `pass` 列 = step/337（连续量）。

**数据源**（全部权威、seed 42、128×）：`nglab1x_input_v5_128x_freq10_fd_fixed` 的 `train_log.jsonl`/`table_norm.jsonl`；
20ep 长 replay（`s1_epoch_long_replay_points.csv`，trigram-only − nogram）；§42 blrabs（绝对 table LR 锁 0.0768）；§38 causalv5m3 freeze 臂。
脚本 `docs/plot_scripts/plot_v5_epoch_dynamics_decomposition.py` → `docs/figs/theory/fig_v5_epoch_dynamics_decomposition.png`。

| 判决 | 数据 | 结论 |
|---|---|---|
| ① 增长在哪：边界跳变 vs pass 内漂移 | input 臂 pass 2–6：边界跳 +0.35/+0.43/+0.40/+0.22/+0.18；pass 内漂移 +0.95/+1.02/+0.76/+0.70/+0.71；四分位剖面每 pass 单调上升 +0.54–0.72 | **72% 的增长发生在 pass 内部**、连续且不衰减；边界离散分量仅 28% 且随 pass 衰减。「逐 pass 写入」只描述了次要分量 |
| ② 是否饱和 | 20ep net gap 每 pass 增量 e2–e5 均 1.16 → e6–e12 0.72 → e13–e20 0.60；e5–e20 线性拟合斜率 0.664/pass，R²=0.9976 | **e5 起在 step 时间上线性、无饱和迹象**。「衰减写入 → 平台」被证伪（与 §42 判决③一致） |
| ③ 谁在长：train 还是 val | blrabs lr 3e-4/6e-4/1e-3：train_benefit 在 pass ~10 饱和于 ≈2.5 且三条 LR 曲线重合；val_penalty 持续增长且携带全部 LR 依赖（e29: 6e-4 臂 10.61） | 晚期增长 **100% 是 val 被伤害**、train 已无增益。这是「logit 尺度/margin 持续放大」的指纹：train CE 已饱和 ~0，val 上错误 residual 被同一尺度线性放大 |
| ④ 表是否一次写完 | trigram table RMS pass 1→6：0.80/1.49/2.01/2.41/2.73/3.01（×3.8，增量 0.69→0.28）；bigram 同型 | **否**，表范数持续增长（无 wd）。但 freeze_table@e2 保留 94% gap（5.386 vs 5.733）、freeze_table@e1@1000 甚至更大（3.452 vs 2.724）→ 表的后续增长对 gap **无关甚至略负**；放大发生在 backbone 侧（freeze_backbone@e1 → 1.585） |
| ⑤ pass 数是否独立于 dose | 固定 pass 6：gap 2.99(6e-5)/3.51(1e-4)/4.70(3e-4)/5.43(6e-4)/5.83(1e-3)/5.77(2e-3)，~η^0.18 且 ≥1e-3 饱和；匹配 dose D=5.925：8.5/11.5/13.1/14.7（pass 9→59） | 两组同时被「每步增量 h(η) 随 η 饱和、对 step 累加」解释：固定 pass ⇒ 弱 η 依赖；固定 dose=N·η ⇒ gap=D·h(η)/η 随 η 降而升。**不需要 pass 特有机制**；§42 判决②应改述为「gap 是 step 累加量，不是 dose 累加量」 |

**重新表述的动力学**（可证伪）：表在 pass 1 为每个 train context 提供私有 key，使 train 集对 backbone「可分」；
pass 2 起 CE 在归一化优化器（AdamW/RMSProp：步长不随梯度缩小）下持续放大 train margin，速率≈常数/step，
不与 pass 数绑定；epoch 结构只通过 (i) 泄漏开启时刻（pass 2 起点）、(ii) 28% 的边界刷新项进入。
预测：(P1) 关掉/加大 backbone wd 改变晚期斜率与平台（wd 是唯一反向力）；(P2) backbone 换 SGD → log 型凹增长；
(P3) train/val logit 尺度随 step 线性增长（需在 diag 中加记录，零成本）；(P4) 表行方向 cos(e,e+1)≈1、只长范数（需边界快照）；
(P5) shuffled replay 去掉台阶但不改斜率。以上均未做，属下一批 planned。

## §49 · 零 GPU：val 伤害 vs Good-Turing 缺失质量——「覆盖率是原罪，训练动态是放大器」（2026-09-03）

**问题**（用户）：gap 是否全由「val 中 train 未见过/权重错的 continuation」构成？train 见得少的 context 是否更「偏执」？重复看数据本身有无意义，还是覆盖率是原罪、多 epoch 只是训练动态？
**方法**：bigram 分支（2-token context）。本地 `data/freq_index.npz` 的 trigram keys 即每个 bigram context 的 continuation 计数 → 离线算每个精确 f 的 Good-Turing 缺失质量 M_f=N1/f 与经验条件熵 H_f（token 加权）。
从 `nglab1x_{input,nogram}_v5_128x_freq10_fd_fixed/exact_freq_loss.jsonl` 在 pass 边界取 val 伤害 d_e(f)=L_val^input(f)−L_val^nogram(f)。
脚本 `docs/plot_scripts/analyze_v5_val_damage_vs_missing_mass.py` → `docs/figs/theory/fig_v5_val_damage_vs_missing_mass_bigram.png`。
1x shard：3.54M 个 bigram context / 49.7M token；整体 GT 缺失质量 0.278（val 中 27.8% 的 (context,next) 三元组在 train 未出现）；f=1 context 占 token 3.5%。

| pass | d_e/M_f @ f=1 | f∈[2,320) 中位数 A_e | CV | f∈[560,1000) | f∈[5600,10⁴) |
|---|---|---|---|---|---|
| 1 (337) | 2.3 | −0.18 | — | −6.2 | −7.3 |
| 2 (674) | 3.5 | 2.15 | 0.57 | 0.5 | 3.1 |
| 3 (1010) | 5.1 | 4.79 | 0.15 | 5.6 | 11.7 |
| 4 (1350) | 6.3 | 7.25 | 0.06 | 9.2 | 20.3 |
| 5 (1680) | 7.4 | 8.90 | 0.04 | 12.2 | 27.5 |
| 6 (2000) | 8.2 | 10.57 | 0.06 | 15.7 | 36.3 |

**判决**：
1. **不更偏执**：pass ≥3 时 d_e(f)/M_f 在 f∈[2,320) 两个数量级内恒定（CV 4–6%）——见 3 次和见 300 次的 context，每单位缺失质量受到同样伤害。f=1 反而略低（GT 在 f=1 处 N1/f=1 高估真缺失质量，部分是估计偏差）。**gap 的 f 形状 = 数据覆盖统计 M_f，训练只贴一个全局幅度 A_e**（与 kernel rescale 的 vertical collapse 一致）。
2. **幅度 A_e 随 pass 线性**：2.10 nats/pass（e≥2），与 §48 的 step 线性增长同源；val 伤害在 **pass 1 已出现**（f≤4 处 d/M>0，此时无任何重复），train 侧则必须重见样本才显现。重复本身不引入新机制。
3. **高 f 剩余项**：f>10³ 处 M_f<0.07 但 d_e 不归零，平台 ≈2 nats（pass 6）→ 缺失质量之外的「已见但权重错」(sampling error ~S_eff/f，H-KAPPA 的 V 项) 与/或 backbone 泛化被共适应拖累；本数据不能拆分二者（需按 f 的 freeze_backbone 分解）。
4. **方法注记**：以 2-token 经验条件熵 H_f 定义「记忆分数」不成立——backbone 用 2048-token context，pass 1 时 f≳100 的 train loss 已低于 H_f；不作图。
**结论**：novel continuation 与 wrong-weight continuation 是同一件事（p̂_c 相对 p_c 的采样误差，前者是 p̂=0 的条目），二者被同一 A_e 放大故「平等」；覆盖率决定 what/shape，训练步数（归一化优化器下线性）决定 how much。多 epoch = 在有限样本上继续走步，是训练动态现象。

## §50 · 表写入 cosine 轨迹 + seen/novel/margin 分解（step-for-step replay，2026-09-03）

**问题**（用户）：测一下表写入的 cosine/写入强度；gap 增长是否来自「对 train continuation 信心增强 → novel token 概率被压得更低」的动力学。
**方法**：`docs/plot_scripts/analyze_v5_table_cosine_decomp.py` 在 ophis 上对三个权威 run 做**逐位同序 replay**（同数据顺序/优化器/schedule，seed 42，128×，2022 步）：input（=nglab1x_input_v5_128x_freq10_fd）、nogram、freeze_backbone_e1（=causalv5m3）。pass 边界（337k）快照 bigram/trigram 表权重（fp16，`data/runs_theory/<run>_snap/rows_*.npy`，19GB 留在集群），并在 fixed val pool 上按 context f 聚合 seen/novel 损失与 margin（logit(train-dominant continuation) − logit(true novel token)；seen/novel 由 1x shard 的 trigram keys 离线判定，dominant = train 中该 context 最高频 continuation）。后处理 `analyze_v5_cosine_margin_posthoc.py`；图 `plot_v5_table_cosine_margin.py` → `docs/figs/theory/fig_v5_table_cosine_and_margin.png`。行 hash 与 `train.py` clean-table primes 逐位一致。

**A. 表不是「一次写死后只长范数」**（§48 判决④修正）：
- cos(row@e1, row@e6) 中位数：bigram 0.52（f=2 桶）→0.76（f=10⁴ 桶）；trigram ~0.57（f 不敏感）。pass 2 后仍 0.85，pass 3 起持续旋转。
- 每 pass 相对写入 ||ΔW||/||W||：bigram 0.91/0.52/0.48/0.45/0.41，trigram 1.15/0.58/0.43/0.38/0.36（pass 2–6）——是纯尺度增长参考（norm 比 −1）的 3–6 倍 → **持续真实重写**，不是只放大。
- 但 §48 的因果结论不变：freeze_table@e2 保留 94% gap → 表的重写对 gap 无关紧要；表动力学是 RMSProp 无衰减、梯度不衰减下的惯性滑行（row 近似 dW/(λ+dW) 随机游走，cos(e1,e6)~1/√6≈0.41 量级吻合）。
- bigram 低频行旋转更快（cos 0.52 vs 高频 0.76）：f 小 → 单位时间曝光少 → 相对游走更快，符合曝光计数模型。

**B. gap 增长 = novel-token 抑制动力学（用户假设证实，且是 backbone 侧）**：
- val 伤害按 continuation 身份拆开（input − nogram，pass 6，bigram 分支）：novel token（该 (context,next) 在 train 未出现）d_novel = 9.6(f=1) → 4.5(f=10⁴)；seen token d_seen = 3.5(f=1) → 2.1(f=10⁴)。pass 1 时 d_seen 在高 f 为 **−1.2**（表模型对 seen continuation 更好），pass 2 起翻正。
- margin（train-dominant continuation 的 logit − 真实 novel token 的 logit）随 pass 单调增长：f=1 桶 1.6→3.9→5.3→5.9→6.5→7.0（pass 1–6）；f=13/100 桶同样线性增长。**freeze_backbone@e1 后 margin 基本冻结**（f=1：pass 1 的 3.3 → pass 6 仅 4.0，增量 −76%；f=100 完全平）→ 抑制由 backbone 共适应驱动，表单独做不到。
- 机制闭环：表给每个 train context 私有 key → backbone 学会「认出 context 就输出 train continuation」→ 对任何未见 continuation 的 logit 被持续压低（CE 对可分集的 margin 最大化，归一化优化器速率恒定）→ val 上 novel token（占 token 流 ~28%，Good-Turing）损失线性上升。f 越小 margin 越大（记忆更强），与 §49 的 d/M 平坦互补：M_f 决定质量项权重，margin 决定每单位质量的伤害，二者在 f∈[2,320) 近似相互抵消成平坦。
**遗留**：seen 侧高 f 的 2-nat 平台（sampling error vs backbone 泛化拖累）仍不可分；trigram 分支的 seen/novel 需 fourgram 计数（未做）。replay 复现口径与权威 run 的 train loss 一致（step 2022: 1.2263 vs 权威 ~1.22），但 decomp/snap 属 `data/runs_theory/` 分析产物，非新训练 run。

## §51 · f × pass 全账目表（seen/novel × table-direct/backbone-carried，2026-09-06）

**问题**（用户）：把所有 gap 找出来——seen-but-rare 是否也有大 gap；r(f) 的函数形式；table 直接作用与 backbone 携带作用的拆分。
**方法**：`docs/plot_scripts/analyze_v5_f_by_pass_table.py` 聚合 §50 replay 的 `decomp.jsonl`（input/nogram/freezebb，bigram 分支，6 个 pass 边界 step，f 从 0 到 2×10⁵ 分 26 桶）。新增拆分：`table-direct = input_full − input_同一权重关表`（`_c` 列是 CleanForward 关表前向），`backbone-carried = input_关表 − nogram`。产物：`docs/figs/theory/tab_v5_f_by_pass_accounting.csv` + `fig_v5_f_by_pass_accounting.png`。

**判决**：
1. **r(f) 函数形式**：margin 每 pass 增量 r_margin(f) ≈ 1.05 − 0.077·log₂f（f=1: 1.01 → f≥2049: 0.21，log f 线性）；novel 伤害增量 r_novel(f) ≈ 1.55 − 0.05·log₂f（f=1: 1.50 → 高频: 0.95）——**有 ~0.95 nats/pass 的 f 无关地板**，margin 机制（×(1−p̂)≈0.87）只能解释 f 依赖部分，地板项是 backbone 全局漂移。f=0（train 从未见过的 context）novel 伤害增量 +1.57/pass，为全桶最高之一。
2. **seen 侧精确形状**（修正 §50「常数退化+低频保护」的粗粒度表述）：pass 1 时 seen 伤害从 f=1 的 **+2.29** 单调下降、f≥7 转负（保护），谷底 −1.2 @ f≈100–900；每 pass 斜率则是 **f=1 最小（+0.23）→ f≥55 起平坦 +0.55**。即低频 seen 的「保护」在斜率上，但 pass-1 的环境门控首击（+2.29）太大，pass 6 时 f=1 seen（+3.45）反而是全 seen 桶里最差的。到 pass 6 所有 seen 桶全为正（+1.9 ~ +3.5）。
3. **table-direct vs backbone-carried（pass 6）**：table 直接作用 = seen **−5.5 ~ −7.7**（巨大收益，符合记忆定位）+ novel 仅在 f≤13 为正（f=1: +1.9），f≳13 转负（高频 novel 上表反而小幅有益）；**backbone-carried = +6.5 ~ +10.8，seen/novel 全 f 平坦**，是总伤害的主体（~2/3）。⚠️ 口径注意：关表前向对共适应的 backbone 是分布外输入，backbone-carried 含「共适应耦合 artifact」，是真 backbone 伤害的上界；干净归因须靠 freeze 系因果 run。
4. **对主线故事的修正**：novel 抑制的 table-direct 部分集中在 f≤13 的低频；f 无关的伤害地板与 seen 侧普遍退化都来自 backbone 侧。「table 立锚点、backbone 锐化 margin」的定性分工不变，但定量上 backbone 项比 §50 表述的更大、更平。
**遗留**：trigram 分支（需 fourgram 计数）；backbone-carried 的 co-adaptation artifact 分离（freeze_table 后关表前向对比）；§49 高 f 平台现在可拆 = backbone-carried 地板项。
