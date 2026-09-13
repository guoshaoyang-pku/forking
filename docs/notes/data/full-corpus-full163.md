# 完整语料数据集（full-163）

> **这份文档的唯一目的**：让以后的仿真实验知道这个完整数据集存在、在哪里、怎么生成。
> 它**不是**极简主线的一部分；对应的启动脚本已于 2026-08-23 删除（从未跑完、非主线）。
> 如果以后要用完整数据做仿真实验，按 `agents.md` §1 的极简 setting 重新写启动流程，
> 下面只保留**数据坐标和生成参数**，不复活旧脚本。

---

## 1. 数据坐标（ophis-gpu 集群）

| 项 | 路径 |
|---|---|
| **原始语料（163 个分片）** | `/data2/shared/ncpl-pathA/harry_autoresearch_full_data/data` |
| **tokenizer（固定，勿换）** | `/data2/ncpl-pathA/work/vbird_autoresearch/cache/tokenizer` |
| 历史工作区 | `/data3/guoshaoyang/ngram-gap-exp/` |
| 计划输出目录 | `ngram5_data/trigram_exact_alpha0.0_full163_20260808` |
| token 缓存目录 | `ngram5_data/token_cache_full163` |

⚠️ 使用前先上机确认这两个 `/data2` 路径仍然存在且可读（它们是别的项目的共享目录）。

## 2. 数据划分方案

- **train** = 全部 163 个分片中**除 06542 以外**的所有分片
- **val** = 分片 **06542**（group-canonical 的「未见过」验证分片）
- 划分清单文件 `data_split.full163.json` **从未入库**；重做时按上面的规则重新生成即可，
  不需要找回旧文件。

## 3. 生成参数（历史方案，供参考）

当时计划用旧的 `ngram5_freq_gap/data_gen.py` 生成**无干预基线**数据集；该 trainer 已退役，以下仅作历史坐标：

```
--alpha 0.0            # 不做低频上采样（纯基线）
--order 3              # trigram（注意：包名里的 "5" 是历史误称）
--bucket-count 5000000
--f-train 0.8  --f-val 0.2
--k-min 0.25   --k-max 8.0
--r-ref-mode median
--dataset-seed 20260808
--doc-len 2048
--val-source test  --val-frac 0.02
--emit-format bin      # 二进制 uint16 输出
--fast-scan --fast-emit
```

## 4. 状态

- **数据集从未生成完成**。卡在数据生成阶段（closure 状态 `DATA_GEN_RUNNING`），
  `meta.json` 从未产出。集群上 `ngram5_data/trigram_exact_alpha0.0_full163_20260808/`
  若存在残留，视为不可信。
- 原计划训练是 4-GPU DDP、70000 步持续预训练，**从未启动**。
- 已删除的脚本：`run_big_continuous.sh`、`launch_ddp_train.sh`、`monitor_full163.sh`
  （git 历史里可查，但**不要复活**——它们依赖集群外部文件且绑定旧代码结构）。

## 5. 以后要用它做仿真实验时

1. 按 §1 确认数据还在。
2. 按 `agents.md` §1 极简 setting 写**新的**数据准备与训练流程（主线 `code/prepare_data.py`
   与 `code/train.py` 是起点，不要恢复旧 `ngram5_freq_gap` trainer）。
3. 明确这次仿真的**唯一自变量**，并在 `docs/experiment-lines.md` 登记新实验线。
