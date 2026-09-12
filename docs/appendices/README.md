# 附录实验（docs/appendices/）

> 主线实验之外的**专题深挖**放这里。每个附录一个子文件夹，**自包含**：
> 实验报告（`report.md` + 给用户看的 `report.html`）、绘图与提取代码、`figs/`、`results/`。
>
> 与 `tasks/`（敏捷验证：toy / 数学模型）的区别：附录面向**主线实验线的深挖**，
> 通常复用 `data/runs_fixed/` 的既有 run + 少量补跑；`tasks/` 是独立的轻量验证。
>
> ⚠️ 附录**不写入**已弃用的历史报告；
> 主汇报只在 blog 仓库的 `index.html`（本地副本 `docs/report/index.html`）。

## 目录

| 附录 | 主题 | 状态 |
|---|---|---|
| ls20ep_logit_stats/ | 20-epoch logit sharpening、CE damage 与 exemplar 分布 | 已完成（§53；input/nogram，seed 42） |
| [`lr_beta_ablation/`](lr_beta_ablation/report.html) | 表学习率 × β₂ 消融及交互；高表学习率体检（发现 ×2/×4 崩坏） | 🟡 进行中（2 个补点跑中） |
| [`s1_scaling_three_axis/`](s1_scaling_three_axis/report.md) | Epoch length / exact frequency / table size 三轴 scaling | 🟢 seed 42 full grid + table 加密取点完成（98 run，QC 通过）；多 seed 待补 |

跨轴覆盖表见 `s1_scaling_three_axis/c4_cross_axis_coverage.md`；它不会把未测的三维交叉项当成已测。H.1.3 静态 run_id 导航见 `run_id_index.md`；索引存在不等于 run 已完成。

## 新增附录的约定

1. 子文件夹名用小写下划线短名。
2. 必备：`report.md`（给 agent 与协作）+ `report.html`（给用户看，SVG 内联）+
   `extract_data.py`（数据提取）+ `make_figures.py`（绘图）+ `build_report.py`（HTML 渲染）。
3. 数据一律来自 `data/runs_fixed/*_fixed/`；新补跑的 run 命名要能进 `runs_fixed/`。
4. 在本索引表登记一行。
