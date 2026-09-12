# Experiment log duplicate review · 2026-09-13

Top-level headings were scanned after the §5/§10 repair. Exact section-number collisions are absent; historical aliases remain intentionally preserved.

## Heading inventory

- L7: ## 实验登记总表
- L117: ## 1. 注入点消融（2026-08-05，OPHIS 旧 run 迁移）
- L158: ## 2. 新 repo 复现验证（2026-08-05 完成）
- L176: ## 3. 频率 bin 分解（2026-08-05 完成）
- L182: ## 4. 双倍 training size 延长实验（2026-08-06，ophis-gpu）
- L250: ## 5. 基础实验统计与图表归档（2026-08-06）
- L310: ## 6. 双倍训练集 v10 细曲线（2026-08-06，input）
- L342: ## 7. 半 epoch 训练集（2026-08-06，input）
- L383: ## 8. 标准 1x v10 重跑（2026-08-06，blog 克隆任务）
- L420: ## 9. Table 优化器消融（2026-08-06，input，计划）
- L582: ## 10. shard 大小扫描（epoch 长度剂量，2026-08-07，彻夜批）
- L775: ## 11. toy-model 台阶清晰度溯源（2026-08-07，郭绍阳 + 2 workers）
- L869: ## 12. epoch 对齐批（同 epoch 数 × 同 LR-per-epoch 轨迹，2026-08-07 进行中）
- L946: ## 13. toy 严格 Zipf 分布 · per-bucket gap 双对数（2026-08-07，planned）
- L1010: ## 14. 干净 vanilla 复现（2026-08-23，input 主臂 + nogram 对照）
- L1048: ## 15. P1/P2 因果干预 · 极简 setting 重跑（2026-08-24）
- L1108: ## 16. bf16 精度验证 + 提速（2026-08-24）
- L1149: ## 17. S1 三轴 scaling 验证（2026-08-24，plan-5）
- L1269: ## 18. 训练提速工程：freq-bin eval 瓶颈 + `--val_steps`（2026-08-24）
- L1321: ## 19. 自然语言 5gram（order=5）· 极简 setting 重跑（2026-08-24）
- L1397: ## 20. bigram 大表 + 免碰撞（perfect-map）极限臂（2026-08-25）⚠️ **历史 4 层框架**
- L1503: ## 21. v3 波次：freq-bin train 侧改为当前 batch（online，零额外 forward）（2026-08-25）
- L1536: ## 22. clean 单表 bigram R 网格（新 SSOT 框架首扫，2026-08-25）
- L1637: ## §21 · V4 波次（uniform LR 基线 · 2026-08-25）
- L1664: ## §21a · Clean-table backbone LR 快速扫描（warmup 后恒定 · 2026-08-25）
- L1731: ## §24 · V5 优化器与学习率定标（clean-table 主线 gate，2026-08-25）
- L1787: ## §24b · V5 证据刷新：完整曲线 optimizer、causal 与剂量频率（2026-08-26）
- L1969: ## §24c · 高 table-LR 的 β₂ 不敏感性 gate（2026-08-26）
- L2054: ## §25 · V5 主线全量重刷队列（2026-08-25）
- L2094: ## §21b · clean 单表 v4 加密网格（uniform LR 重刷，2026-08-25）
- L2136: ## §26 · 5-gram condition sample285 trunk 对照（2026-08-26）
- L2183: ## §23 · L6 残差—响应精确模型（2026-08-25）
- L2244: ## §27 · X1 优化器三臂 × seed 复现（2026-08-26 登记）
- L2314: ## §28 · X2 clean 表行宽扫描 d ∈ {768,192,48,12}（2026-08-26 登记）
- L2365: ## §29 · X3 语料侧 r̄(f) 支撑宽度统计（2026-08-26 登记）
- L2413: ## §30 · V5 zero-warmup constant schedule 配对消融（2026-08-26）
- L2463: ## §31 · V5 warmup 起始倍率敏感性（2026-08-26）
- L2531: ## §32（§24d 归档编号）· V5 高 table-LR × β₂ 收敛批（optv5f，2026-08-26）
- L2624: ## §33 · V5 三轴 scaling 快速批（table / frequency / epoch，2026-08-26）
- L2743: ## §34 · V5 三轴 scaling 单表重刷批（用户 2026-08-27 拍板修正）
- L2806: ## §35 · V5 标准 table LR 切到 128× 的全量标准实验重刷批（用户 2026-08-29 拍板）
- L3005: ## §36 · S1 table-size 小 R 扩展批（R 从 1e4 扫到 1e0，2026-08-29 用户拍板）
- L3078: ## §37 · 理论审计与勘误批（零 GPU 分析，2026-08-30）
- L3106: ## §38 · κ 微观起源与稀释面零 GPU 判决 + causal_dynamics 批（2026-08-30 晚）
- L3150: ## §39 · V5 backbone LR 扫描 · A 因子判决批（blrv5，2026-08-30 深夜，用户拍板）
- L3201: ## §40 · V5 epoch 长度轴修复批（epfx，2026-08-31，用户拍板）
- L3235: ## §41 · V5 20-epoch 长 replay + 二次 reseed 判决批（2026-08-31，用户拍板）
- L3290: ## §42 · 纯 backbone LR 动力学：绝对 table LR 锁定批（blrabs，2026-08-31）
- L3486: ## §43 · V5 净收益判决批（netv5，2026-08-31，用户拍板）
- L3560: ## §44 · 分 branch mask 扫描 + 仅末 epoch 用表（2026-08-31 23:45，用户拍板）
- L3636: ## §45 · freeze 四因子长程批 + pass 混匀判决批（2026-09-01，用户「继续排任务」批）
- L3746: ## §46 · 零 GPU：offline hash collision 实测 owned token mass vs R（2026-09-01）
- L3780: ## §47 · fast-diag 测量管线 + 主实验四臂 fd 重刷（2026-09-01，用户拍板）
- L3821: ## §48 · 零 GPU：gap-vs-pass 动力学分解——推翻「表一次写完 / backbone 逐 pass 衰减写入」（2026-09-01）
- L3847: ## §49 · 零 GPU：val 伤害 vs Good-Turing 缺失质量——「覆盖率是原罪，训练动态是放大器」（2026-09-03）
- L3871: ## §50 · 表写入 cosine 轨迹 + seen/novel/margin 分解（step-for-step replay，2026-09-03）
- L3888: ## §51 · f × pass 全账目表（seen/novel × table-direct/backbone-carried，2026-09-06）
- L3900: ## §52 · y/v 三轴 scaling + 多层注入波次（2026-09-06，用户拍板）
- L3947: ## §53 · 置信度极化探针 + 20-epoch logit-sharpening 直接验证（2026-09-12，planned→running）
- L3997: ## §54 · Transformer backbone 架构图资产（2026-09-12）

## Repeated normalized titles

None.

## Review conclusion

- No automatic deletion or merge is justified from heading similarity.
- Sections §21/§24 aliases and historical runs need claim-level review, not renumbering.
- Next manual pass should inspect repeated numeric tables and figure references within §§35–53.
