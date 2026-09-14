# 图 7-2：20 点 epoch 长度 scaling 的 epoch 2 快照

冻结时间：2026-09-15 00:08:39 CST。共 20 个长度点，19 个已采到精确 epoch 2 边界，18 个已完成全部 3 epochs；20× 的 epoch 2 尚未观测。图中 epoch 2 为主曲线，epoch 1 为浅灰背景，缺失点留空，轴下三角仅标识待采样位置。

## 图片和复现

- 图片：[PNG](../../figs/main/fig_s1_epoch20_epoch2_live.png) / [SVG](../../figs/main/fig_s1_epoch20_epoch2_live.svg)。
- 边界数据：[CSV](../../figs/main/fig_s1_epoch20_epoch2_live.csv)，60 行对应 20 个 run 的 epoch 1/2/3，未观测数值为空。
- 证据：[epoch2_snapshot.json](epoch2_snapshot.json) 包含解析后的训练日志、summary 或运行中配置、文件 SHA256、源路径、数据容量及代码校验信息；[epoch2_status.json](epoch2_status.json) 记录进度和 ETA。
- 采集脚本：[collect_epoch20_snapshot.py](../../plot_scripts/collect_epoch20_snapshot.py)；绘图脚本：[plot_epoch20_epoch2_live.py](../../plot_scripts/plot_epoch20_epoch2_live.py)。

从仓库根目录执行，现有快照可离线重画；第一条命令需要 SSH 到 ophis-gpu，只读取小型日志和元数据：

```bash
.venv/bin/python docs/plot_scripts/collect_epoch20_snapshot.py
.venv/bin/python docs/plot_scripts/plot_epoch20_epoch2_live.py
```

刷新命令覆盖当前快照和图片；需要保留旧快照时，采集与绘图脚本均支持显式快照路径参数（分别为 `--output` 和 `--snapshot`）。本说明中的进度和 ETA 只对应上述冻结时间，后续以生成的 status JSON 为准。

## 数据坐标和口径

实验登记见 `docs/experiment-log.md` §55。run_id 为 `s1v5_128_ep20_tri_<label>xL4_3ep_v2`，源目录为 `ophis-gpu:/data4/guoshaoyang/ngram-gap-lab/data/runs_scaling/<run_id>_fixed/`；沿用该实验已登记的 S1 输出目录例外。原始 `train_log.jsonl` 与 `summary.json` 的本地镜像在 `~/.cache/ngram-gap-lab/epoch20/data/runs_fixed/<run_id>_fixed/`。仓库 `data` 指向当前未挂载的外置盘，采集使用缓存路径。

数据池为 `ophis-gpu:/data4/guoshaoyang/ngram-gap-lab/data/tokenized_epoch20/`：train shards 0–20，val shards 21–23；训练池 6981 batches，大于最大 epoch 的 6740 batches。这里确认的是单个 epoch 内不会因池容量不足而绕回；§55 的 chunk 审计记录 train 内有 1 个自然重复块，不能称为完全去重数据。

每个点使用 seed 42、trigram-only、input 注入、clean R=2^20、table RMSProp(0,0.99)/LR=128×、backbone LR=0.0006、warmup_constant(100)、vanilla 8L/6H/768D、bf16/no compile。fixed val 为 4 batches，日志每 10 步及精确 epoch 边界采样。epoch 2 的取值 step 严格等于 `2 × epoch_batches`。

纵轴为同一 logged step 的 `fixed-val CE − online current-batch train CE`，单位 nats；train 在更新前、val 在更新后测量。横轴采用实际 `epoch_batches / 337`，标称倍率在 CSV 中另列。图展示原始 gap 边界值，不减 epoch 1、不平滑，也没有插补待完成点。

## 当前观察和进度

seed 42 的 epoch 2 raw gap：1× 在 step 674 为 1.081173；4× 在 step 2696 为 0.600700；10× 在 step 6740 为 0.145931；15× 在 step 10110 为 0.186611。精确 run_id 和每点源文件哈希见 CSV。整体随数据长度下降，10×→15× 小幅回升；单 seed、当前训练 batch 边界值不足以支持严格单调或幂律结论。

在冻结时间，15× 位于 14170/15165 steps，20× 位于 10780/20220 steps，两个日志均在 2 秒内更新。按最近 1000 steps 的耗时估算，20× 的 epoch 2 边界（step 13480）约在 00:20:26，整批全部 3 epochs 约在 00:49:55 CST 完成；合理报告范围为 00:45–01:00。ETA 是该快照下的速度外推。
