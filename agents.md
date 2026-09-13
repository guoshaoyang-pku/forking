# ngram-gap-lab · Agent 指南

> 本仓库是本课题**唯一的开发仓库**。`OPHIS_gap` 已弃用。
> 本文件只写**不变的规则与坐标**：工作原则、极简 setting SSOT、文档权威性、可用算力。
> 会变的东西一律 offload：实验进度→`docs/experiment-log.md` + `experiment-lines.md`；
> 存储细则→`docs/notes/data/cluster-infra.md`；废弃台账→`docs/notes/method/deprecated-list.md`；
> 常用命令→`docs/notes/method/common-commands.md`；测量基础设施→`docs/notes/method/measurement-infra.md`；
> OPHIS 迁移→`docs/_archive/docs/ophis-migration.md`。

---

## 0. 工作原则

冲突时编号小的优先。

- **P1 极简优先**：任何实验只用 §1 极简基线，只改当前要检验的一个变量。禁止 `current shell`、Muon、RoPE、RMSNorm、fourgram、gate、MuonAdamW grouping。偏离须先在 `experiment-log.md` 写明原因。
- **P2 先读再改**：改任何文件先完整读一遍；改 `code/train.py` 影响口径须新起 `run_id`；不确定结论查 `docs/claims-ledger.md`。
- **P3 一次一件事**：一实验 = 一 `run_id` = 一 `data/runs_fixed/<run_id>_fixed/` = `experiment-log.md` 一行+section。生命周期：planned → 占 GPU → running → done(24h 内回填) → 沉淀。未回填标 `stalled`。
- **P4 口径一致**：gap = `val_loss − train_loss`（同 logged step 的 fixed-val − 当前 batch 的 online train loss）。**权威数据只有 `data/runs_fixed/` 带 `_fixed` 的 run**（`data/runs/` 受 freq-bin bug 污染作废，修正幅度 +210%）。val 必须 fixed batches；`VAL_LOSS_INTERVAL_STEPS=10`，freq 同步。`novel`(hit=0) 无 train loss，不定义 gap。多机跑前 `md5sum` 核对 `code/`。
- **P5 先问**：大改 setting/新架构变体、跨集群大文件(>1GB)、删已完成 run、`git push`/改分支——先征得用户同意。
- **P6 并行优先**：非阻塞调查/作图/分析派 subagent 并行；但理解代码不能外包。
- **P7 文档规矩**：结论必须附 run_id/step/seed，否则写「假设」；current-shell 系数字标 `[DEPRECATED SETTING]`；图进 `docs/figs/`、脚本进 `docs/plot_scripts/`，一起 commit。

---

## 1. 极简 setting（SSOT，唯一权威）

> 任何实验的 setting 表以此为基准，只用粗体标差异。1000 步见 train/val forking；2000 步为标准延长口径。

### 1.0 最小可执行契约（交接用）

协作者按下表锁定主线，只改当前实验要检验的一项。**「没写出」≠ 可沿用 CLI 历史默认值。**

| 坐标 | 主线默认 |
|---|---|
| 注入 | `input` / wte |
| backbone LR | `0.0006` |
| table optimizer | RMSProp 无动量 `--table_betas 0.0,0.99` |
| table LR | `--table_lr_scale 128.0`，实际 `0.0768` |
| LR schedule | `--lr_schedule warmup_constant --warmup_steps 100`：step 1–100 从 0.25×(`0.00015`) 线性升到 1×(`0.0006`)，之后固定；禁 warmdown |
| 默认预算 | seed 42，1000 steps，bf16，不 `torch.compile`（H200 实测 compile 反慢 3.5x，因 n-gram 字典 graph break，见 `experiment-log.md` §18） |
| 口径 | 当前 batch 的 online train loss；fixed val；`gap = val − train` |
| 测量管线 | 诊断 forward bf16 + 异步 CPU 聚合（`diag_worker.py`，2026-09-01 拍板为标准；commit 100619d）；此类 run_id 加 `_fd` 后缀，eval block ~1.5s。旧 fp32-diag run 仍有效（训练动态逐位同分布），诊断曲线不跨后缀混画单条 |

非 table-size 实验固定 `R_bigram = R_trigram = 2^20 = 1,048,576`；table-size 才可改 R 且 R 是唯一变量。完整 setting 仍须写明 train/val shards、frequency index、eval 节奏。旧 `table_mult=64` / "1M table" 不是 clean-table 的隐式来源。

### 1.1 模型 / n-gram / 优化器 / 数据

**模型**：vanilla nanoGPT（非 current shell/Muon）· 8L·6H·768D · vocab 8192 · seq 2048 · learned absolute PE（非 RoPE）· LayerNorm（非 RMSNorm）· tied embedding · LLLL 全 attention。

**n-gram**：注入点 `input`/wte（`x = wte(idx) + Σ ngram_ve`，不走 attention）· bigram+trigram（unigram/fourgram 关）· **clean 单表** `nn.Embedding(R, n_embd)`（单层、无 2-hash、无 4 层求和，见 `clean-table-rework.md`）· 单一 hash（`--*_clean_table R`，R=distinct+1 零碰撞）。旧 1M/4层/2-hash 框架仅作历史溯源。`v`/`y` 注入只作消融对照，非主线。

**优化器**：表 RMSProp 无动量 `(0.0, 0.99)`（2026-08-24 拍板；历史 β₂=0.999 因 B2 bug 无证据）· table LR **128×**（实际 `0.0768`；2026-08-29 用户拍板标准切到 128×，来自 LR 扫描 step-1000 gap 峰值 ~2.73 @128×）· backbone AdamW `(0.8,0.95)` wd 0.1 · lr `0.0006`(2026-08-25 v5，来自单 seed 筛选：6e-4 gap 1.534 > 4e-4 1.187 > 4e-3 0.060) · schedule `warmup_constant(100)`，禁 warmdown；零 warmup `constant` 仅诊断。

**数据/训练**：`fixed` replay · data_seed 42 · train shards 1(标准 1x) · val shards 与 train **完全不重叠** · device batch 72 · total batch 147,456 tok · seed 42(43/44) · steps 1000/2000 · bf16 autocast 不 compile · eval 三层默认：主实验 `freq=10`、只看曲线 `freq=50`、只要末端 `--val_steps 1000`（freq 跟随 val_steps 对齐；fast-diag 标准管线下 eval block ~1.5s，不再是瓶颈）。

**时间点**：online train loss 在参数更新前记录，fixed-val 在同 step 更新后。改动它属测量语义改变，须新起 run_id。

**参考数值**：权威数据在 `data/runs_fixed/`；数值与全景见 `docs/experiment-lines.md`（v10 四臂 final gap：input 1.867 / y 5.804 / v 5.450 / nogram 0.245）。

---

## 2. 文档权威性（冲突解决规则）

历史上存在三份同名主文档冲突，从即日起按下表判定：

| 层级 | 文档 | 地位 |
|---|---|---|
| 权威主汇报（发布地） | blog 仓库 `blogs/ngram-gap-mechanism-guide/index.html` | ✅ 唯一权威版，已剔除 current shell |
| 权威主汇报（本地副本） | `docs/report/index.html` | ✅ 与发布地同步的只读副本 |
| 权威背景页 | `docs/report/background.html` | ✅ 术语、伪代码、debug、current-shell 废弃说明 |
| 历史版本 | `docs/report/versions/`（20260805 / guide-0728 / guide-0730） | 🗄️ 溯源用 |
| 历史版本 | `docs/report/versions/guide-full-chapter0-19.html` | ⛔ 建在 `baseline_current` 上，不再维护 |
| 历史版本 | `docs/report/versions/regime-bridge-DEPRECATED.html` | ⛔ 同事的 current-shell 实验 |

**规则**：
1. 引用主实验结论一律引 `docs/report/index.html`（= blog 发布版）。
2. **主文档只在 blog 仓库手工编辑**，改完 copy 回 `docs/report/index.html`。`sync_to_blog.sh` 只同步图与独立报告，**绝不覆盖 index.html**，也不自动 push。
3. 内部实验事实来源：`experiment-lines.md`（全景+权威数据）→ `experiment-log.md`（登记簿）→ `claims-ledger.md`（断言台账）。
4. `versions/` 历史版本只在查「历史怎么做过」时打开，不作结论来源。

---

## 3. 仓库结构

设计原则：`docs/` 只保留 7 个子目录；独立敏捷验证任务放 `tasks/`（自包含：脚本+`results/`+fixture）；bug 内容彻底删除不归档。

```
ngram-gap-lab/
├── agents.md                    # 本文件
├── .agents/skills/ + README.md
├── code/                        # 主线 nanoGPT（train.py / ngram_freq.py / cluster/ / tools/）
├── tasks/                       # 自包含的自然语料 scaling 任务
├── docs/
│   ├── experiment-lines.md      # ★★ 实验线全景 + 权威数据源 + 待办（入口）
│   ├── experiment-log.md        # ★ 实验登记簿
│   ├── claims-ledger.md         # ★ 断言台账 C1–C9
│   ├── report/                  # index.html / background.html / versions/
│   ├── notes/                   # theory / literature / method / data
│   ├── plans/                   # plan-1..5
│   ├── figs/ + plot_scripts/ + appendices/ + _archive/docs/
│   └── sync_to_blog.sh
└── data/                        # gitignored：tokenized / freq_index*.npz / runs_fixed/
```

**入口顺序**：agents.md → `experiment-lines.md` → `experiment-log.md` / `claims-ledger.md`。

**tasks/ 约定**：独立、自包含、单机快速跑完。目录 `lN_<短名>`，脚本在根、结果在 `results/`、fixture 在 `results/inputs/`；不依赖 `code/train.py`、不依赖 GPU、不写 `data/`；新增时在 `tasks/README.md` 补一行说明科学问题。

---

## 4. 算力与存储（可用算力总览）

**完整目录细则见 `docs/notes/data/cluster-infra.md`。**

| 集群 | 连接 | GPU | 公网 | 环境 | 状态 |
|---|---|---|---|---|---|
| **ophis-gpu**（主） | SSH `guoshaoyang@223.167.85.180:50002`，别名 `ophis-gpu`/`ophis_gpu`/`fcloud-223` | 8×H200 (141GB) | ✅ | `uv` + torch 2.9.1 | ✅ |
| **360-1** | VPN → `10.234.161.2:22`，`ssh 360-1` | 8×H200 (143.7GB) | ❌ | python3.10 + torch 2.13.0+cu130 | ✅ |
| **360-2** | VPN → `10.234.161.3:22`，`ssh 360-2` | 8×H200 (143.7GB) | ❌ | 同上 | ✅ |

- 360 系无公网且不能直连 ophis-gpu；跨集群搬运走 **ophis-gpu → Mac → 360** 中转。
- 分工：主线 nanoGPT 三机均可（代码同步后口径一致）；不再启动 toy / synthetic harness。

**跑前强制检查**：① `nvidia-smi` 确认卡空闲，`CUDA_VISIBLE_DEVICES=<id>` 占卡，一 Agent 只用自己的卡；② 同步代码后 `md5sum` 核对 `code/train.py`/`ngram_freq.py`/`cluster/*.sh`（权威源=本地已 commit 版本；教训：360 曾残留旧 `train.py` `f9388473` 口径不一致）；③ 改代码先 commit 再同步，同批实验跨机用同一份代码。

---

## 5. 废弃清单（不得进入主线）

**判定标准（规则）**：命中任一即非极简 setting，不能进主线：
- backbone 是 `current shell`（`nanogpt_current_shell`）
- 用了 Muon / MuonAdamW grouping
- 用了 RoPE / RMSNorm
- n-gram table betas 的 β₁ ≠ 0（有动量），如 `(0.5, 0.999)`
- `VAL_LOSS_INTERVAL_STEPS` > 10，或 val 用移动窗口

**具体废弃项台账与「废弃结论，保留问题」见 `docs/notes/method/deprecated-list.md`**。重跑 current-shell 结论是当前最高价值实验队列。

---

## 6. 论文与外部研究对照（2026-09-13）

本项目的论文计划与中文初稿以以下两份飞书文档为入口；英文投稿稿仍以文档中声明的本地 TeX 为准，飞书页面中的计划、数字和机制只有在对应 run、step、seed 和图源核验后才可升级为论文结论。

- 论文计划：<https://gcnsa0liqppq.feishu.cn/wiki/G4KKwHMWriuHDOkARHCcHWBenwc>
- 中文论文初稿 v2：<https://gcnsa0liqppq.feishu.cn/wiki/Lyr3wfH8FiIWT6kdQmrc6dGUnEf>
- 对照论文：**Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data**，<https://arxiv.org/pdf/2609.11917>

### 6.1 与 MoE 论文的关系：现象一致，机制部分同构

外部论文在固定总 token 预算、改变 unique token 数的实验中发现：数据 repetition 提高时，dense 与 MoE 的 validation CE 恶化、training loss 继续下降；MoE 更早且更快退化，且总参数量／稀疏度比 active 参数更能解释退化。其机制证据是 router 很早稳定，重复数据随后反复更新近似固定的 expert 子集，expert specialization 随 repetition 增强；dropout、FFN/expert output masking 可缓解，router jitter、gradient clipping 和普通 weight decay 作用较弱。

这与本项目的 Forking 现象在操作层面一致：固定 replay 后 train loss 与 fixed-val loss 分叉，gap 随重复曝光累积；局部 n-gram context→row 对齐使更新集中到一小部分条件记忆和相容的 backbone 状态，低频 context/novel continuation 受损，mask 或破坏映射可显著降低 gap。两者可以共享一个较高层的抽象：**数据重复把有限的独立样本反复送入一个稀疏或局部激活的参数子空间，导致局部 specialization／logit sharpening，训练集收益与未覆盖条件的泛化损失分离。**

但目前不能把两者写成“同一机制已被证明”：

1. MoE 论文直接测量的是跨架构 validation CE 随 repetition rate 的退化，以及 router ossification 和 expert knockout specialization；本项目直接测量的是 clean n-gram 的 train/val gap、频率分解、hash reseed、freeze 和 low-frequency mask。
2. MoE 的关键中间变量是**学习到的 router 分区和 expert 函数专门化**；n-gram 的关键中间变量是**确定性 context→table-row 查找、有限续接覆盖、表与 backbone 的共适应**。n-gram 没有可直接等同于 MoE router 的 learned routing。
3. 本项目的“over-encoding 会锐化 logits，因而重复压低尾部并增强 memorization”是很有解释力的统一假说，但应由 logit margin/entropy、seen-vs-novel continuation 概率轨迹和干预实验直接检验；在没有这些证据前，正文应写为机制模型或预测，不能当作已证实的唯一因果链。
4. “n-gram 是极其 sparsely activating 的 local MoE”适合作为结构类比和实验假说：它说明局部条件路径的重复更新为何可能类似 expert specialization；它不意味着 n-gram 在参数化、路由学习或负载均衡意义上就是 MoE。

因此，论文对照结论应使用以下口径：**MoE 论文独立支持“数据稀缺 × 局部/稀疏参数化会放大 repetition overfitting”这一现象级命题；本项目进一步在 n-gram memory 中给出条件续接、频率、映射对齐和 backbone 共适应的因果证据。二者的共同机制候选是重复驱动的局部 specialization 与 logit sharpening，但跨架构等价性仍是待验证预测。**

### 6.2 对照时的实验与写作要求

- 将“相同现象”拆成 train loss、fixed-val loss、raw/net gap、重复次数/epoch 和绝对 validation loss，避免只凭终点 gap 类比。
- 对齐 unique-data budget、total-token budget、active/total parameter、局部激活比例和 regularization；MoE 论文的 repetition rate 不等于本项目的 context frequency F。
- 对照图和结论必须回到 `docs/experiment-lines.md`、`docs/experiment-log.md`、`docs/claims-ledger.md` 及 `data/runs_fixed/`；飞书页面中的机制段落保留假说、待核和已验证三种状态。

## 7. 相关 skill

- repo-level skills 在 `.agents/skills/`，随仓库交接；可用 `$` 显式调用或按 description 自动匹配。个人全局安装可软链到这些目录，但**不以绝对个人路径作为仓库唯一来源**。
- `ngram-gap-settings`：新 setting/ablation/launcher 审计，锁定极简契约、table R、口径、单变量差异。
- `ngram-gap-experiment-registration`：run 注册、跨机 handover、回填、evidence packet。
- `ngram-gap-plotting`：图表规范（injection loss/gap 曲线、loss-gap-table-RMS 对齐、频率 bin 双轴、log-x/log-log、Plotly、blog 非滚动嵌入）。改图前先读它 + `docs/plot_scripts/README.md`。
- `blog-deploy`：发布 docs/ 产物到 GitHub Pages 并验证 HTTP 200。
