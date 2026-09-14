# Epoch-length / Good–Turing 检验

分析 ID：`epoch20_gt_prefix_20260915`，对应 experiment-log.md §55 的 20 个已完成训练 run，seed 42。不启动新训练；CPU 统计在 ophis-gpu，本地拟合与绘图。

预先指定的检验：

1. 从既有纯语料统计重算 trigram 条件频率核指数 β（f=1…100，n(f) 加权），检验直接迁移的 `gap_e(L)=A_e L^-β`，每个 epoch 只拟合幅度。
2. 在本批次实际 train nested prefix 上重新计数，保持与 train.py 相同的 chunk 边界和前两位置 padding，以相同 4-batch fixed val 作为权重，计算 `Q_seen(L)=Σ_{f_c>0} μ_c N1(c)/f_c`。每个长度只统计一次独立训练池，不把 replay 次数乘到 f 上。检验 `gap_e(L)=A_e Q_seen(L)`。
3. 独立列出 novel-context mass；f=0 没有 per-context train loss，不能把 novel 计入定义良好的 per-context gap。`Q_all=Q_seen+novel_mass` 仅作把未见 context 赋 missing mass 1 的覆盖诊断。实际 val 未见 continuation 比例也单独记录，不能与训练侧 Good–Turing 估计混称。
4. 主拟合为全部 20 点、各点等权的 log 残差最小二乘；同时报告原始 nats RMSE/R²、log RMSE，防止只看一个有利指标。与自由指数幂律比较；报告预定 L≤4 拟合→L>4 外推以及 1×/4× 起始窗口的敏感性。划分是在已看过这批数据后制定，只是诊断外推，不是盲测。

本轮先完成了 20 个实际 train prefix 的 Good–Turing 计数：`Q_all` 从 0.83398（0.125×）降至 0.38202（20×），对应全程 log–log 指数约 0.153；`Q_seen` 从 0.30683 升到 0.27161，几乎平坦（其拟合指数约 0.026），因为 novel-context mass 同时从 0.52715 降到 0.11041。epoch 2/3 raw gap 的自由幂律指数分别约 0.508/0.530，明显陡于 Q_all。

证据上限：理论核心对象是 matched no-gram 差分后的 net gap。本批只有 trigram-only raw gap，不能把其拟合直接称为理论验证。epoch 固定而 L 增加时，总训练 steps、间隔、context 组成和固定 R 下的负载也变；`a_e` 对 L 不变是额外可检验假设。旧频率核指数来自另一 shard 和诊断窗口，不能先验等同于全局长度轴指数。

CPU 统计代码：`docs/plot_scripts/epoch20_good_turing_counts.cpp`。远程输出目录：`ophis-gpu:/data4/guoshaoyang/ngram-gap-lab/docs/appendices/epoch20_gt_prefix_20260915/`。预期输入为 `data/tokenized_epoch20/shard_00000.bin`…`shard_00020.bin` 和 fixed-val shard 21 的前 288 chunks；完整模型输入为每 chunk 2048 tokens，target 为后一 token。20× 消费 485280 个完整 chunks。

直接结论：模型**有能力给出方向正确、可检验的预测**，但目前还不能把它称为定量预测模型。它预测更长的 unique 数据池会降低 Good–Turing 缺失质量，从而降低 n-gram 造成的 net gap；这与 20 点 epoch 2/3 的总体下降一致。它没有预测 raw gap 的绝对值，因为 raw gap 还乘上训练状态 `a_e`，并叠加 no-gram/backbone baseline。当前固定绝对 LR 的 sweep 又让 `a_e` 随总 steps 改变，因此不能用这批数据单独估计大模型的 universal gap。

对 DeepSeek 类大规模真实训练，最稳妥的外推对象是**每个 n-gram/context memory 分支的 net gap**，而不是整模型 validation loss：

```text
G_net(e, L; D, eta) ≈ A(e, D, eta, R) · Q_GT(L; corpus, n-gram order)
Gap_raw ≈ Gap_nogram(D, eta) + G_net
```

其中 `Q_GT` 由目标语料的真实 context/continuation 计数计算，`A` 必须用相同模型、优化器、table LR 和累计 optimizer dose 标定。把本实验的 `A` 或 0.508/0.530 指数直接搬到 DeepSeek 会越界；模型规模、tokenizer、数据混合、重复率、n-gram order、table capacity 和 backbone dose 都会改变它。可迁移的是函数结构和需要测量的坐标，不是当前单 seed 的系数。
