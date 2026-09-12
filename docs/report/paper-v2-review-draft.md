# Forking：文章 v2 本地审阅稿（2026-09-13）

这份文件把飞书论文 v2 中仍带“待补”的正文段落改成一版可审阅的本地文字。它不是发布源，也不覆盖 `docs/report/index.html`；发布时仍应先在 sibling blog 仓库审阅，再同步回本地副本。所有数值都限定在已登记的 `_fixed` 运行和对应 section。

## 6 · 可检验预测与验证

### 6.1 Replay 次数与相干更新

固定训练 shard 并增加 replay 时，context 的续接覆盖基本不变，变化来自同一批训练残差被重复写入和读取。我们把局部影响写成 `g_e(f) ≈ a_e M(f)`：`M(f)` 是由训练计数得到的缺失质量核，`a_e` 是完成第 `e` 个 pass 后 backbone 对表方向的可用读出强度。该式是核心区间的工作模型，不是对所有频率和所有训练时长的恒等式。

10-epoch 边界快照显示，按 `e·f` 水平重标度不能把各 epoch 折叠，而按每个 epoch 的幅度 `A_e` 垂直重标度可以较好保留频率形状。freeze 判决进一步区分了两条路径：`ffqv5_freeze_both_e2_10ep_fixed` 在 step 675 后的 8 个评测点上 train/val/gap bit-exact 不变；`ffqv5_freeze_backbone_e2_10ep_fixed` 仅继续写表时，validation 只增加约 0.205，而正常对照增加约 4.549。由此，当前证据支持“表快速写入、backbone 慢速放大”的两状态描述。

这不支持一个固定恢复率或已观测的平台。`docs/report/index.html` §8.6 记录的累计 dose 折叠、常数恢复率和有限平台三个一状态预测均已被判决批否定；因此正文应报告有限窗口内的增长和增量变小，并注明长期平台仍未观测。

### 6.2 学习率与 table / backbone 更新

table 学习率和 backbone 学习率改变的是不同更新算子。`plot_v5_backbone_lr_epoch_dynamics.py` 对固定绝对 table LR=0.0768 的 14 条 run 显示，同一累计 backbone dose 下，不同 pass 数仍给出不同净 gap；pass 数不能被 dose 消去。高 table LR 只会加快写入，不能替代再次看到同一训练样本，也不能立即产生多 pass 的 backbone 读出状态。

因此正文只写两条有证据的结论：第一，table LR 进入高值区后曲线趋于聚拢，说明单纯继续加快写入的收益受限；第二，freeze-table / freeze-backbone 对照表明后期 validation damage 主要需要 backbone 更新。weight decay 的 0/0.1/0.3 单 seed 差异约 0.12，只作弱方向性证据，不能写成恢复率定律。

### 6.3 数据量、epoch 长度与两种对齐

“更多独立数据”和“同一数据重复更多次”不是同一变量。固定总步数会让短 epoch 产生更多 replay，固定 pass 数则会改变每次更新看到的数据量。因而任何 `F→kF` 缩放都必须同时声明：context 集合是否保持不变、验证 token 质量权重是否保持不变、总步数还是 pass 数被固定。

当前 v5 S1 提供两类受控证据：clean table-size 轴在有限 R 中段给出 bigram 斜率约 0.576、trigram 斜率约 0.665；epoch-length 轴固定 3 个 epoch，真正可比的 ≤1×L4 段随数据池增大而单调下降或持平（3.552 → 2.469），而 >1×L4 的旧点是 shard-1 wrap-around 后的 replay/pass 数，不能并入长度轴。L4 10-epoch 长训中 trigram gap 在 step 3370 达 8.675，而 no-gram 对照为 0.480。它们分别支持“容量改变幅度”和“epoch 长度/重复暴露需要分开对齐”的工作判断，但不构成跨数据分布的普适幂律。

### 6.4 容量、映射与干预收益

容量效应应通过 collision、owner mass 和 gap 同时描述。R 改变在当前区间主要改变共同幅度 `ρ(R)`，而不改变 `M(f)` 的形状；bigram 与 trigram 的净 gap slope 分别约为 0.576 和 0.665。直接碰撞测量显示，在 R=2²⁰ 时 bigram context collision 约 71.5%，但 dominant-row token mass 仍约 86.5%；trigram collision 约 94.5%，dominant-row mass 约 43.7%。因此不能用单一 collision rate 代替 token-mass 口径。

低频屏蔽、reseed、freeze 等干预只在声明的 setting 中检验机制。它们可以作为“减少额外 validation damage 的候选操作”，不能被写成对所有 n-gram memory 都保证净收益的处方。正文必须同时报告 absolute validation、train benefit 保留量和额外成本。

## 7 · 跨结构与后训练：设计而非结果

DeepSeek-like / Engram-style 复现和小数据 SFT 压力测试目前都是 protocol。没有固定基座、模块版本、数据构造、代码 commit、run_id 和评估包前，不把它们写成实验结果。SFT 方案至少需要表 frozen / trainable 两臂、目标 continuation 与替代 continuation、无关 context、整体验证集四类指标，并区分局部伤害与一般 SFT 遗忘。

约 189 GiB 的表搬运、跨集群大文件、模型下载和正式 GPU 运行属于单独授权节点；在授权前只保留协议、风险矩阵和 no-go 条件。

## Appendix B · 标准设置与测量合同

- 模型：vanilla nanoGPT，8L/6H/768D，vocab 8192，learned absolute PE，LayerNorm，tied embedding。
- n-gram：`input` / wte 注入，bigram+trigram clean single table，R=2²⁰；unigram/fourgram off。
- 优化：backbone AdamW lr=0.0006，betas=(0.8,0.95)，wd=0.1；table RMSProp betas=(0,0.99)，scale=128，实际 lr=0.0768；warmup_constant 100 steps，起始 0.25×。
- 数据：fixed replay，train shard 与 validation shards 不重叠；标准 seed=42，bf16，无 compile。
- 测量：online train loss 在更新前记录，fixed validation 在更新后记录；`gap = val_loss - train_loss`。novel context 没有 train loss，不定义 novel gap。
- 复现链：图 → plot script → run_id → `experiment-log.md` / `claims-ledger.md`；只有 `data/runs_fixed/<run_id>_fixed/` 或明确标记的 remote-only 产物可作为主线证据。

未解决部分保留为明确的预测或设计：epoch-length 积分在改变频率分布时的修正、重复间隔的定量律、跨结构复现和 SFT 风险，均不因写入本稿而升级为已验证结论。
