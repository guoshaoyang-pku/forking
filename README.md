# ngram-gap-lab

> **文章阶段的唯一开发仓库**：研究 n-gram value memory 在 fixed-order replay 下如何把训练优势转化为 train/validation gap。

当前工作重点是把已完成实验收敛成一条可审计的论文证据链：正文叙事、附录数据、图表、引用和复现实验都必须能回到明确的 run_id、代码版本和测量口径。

## 从这里开始

| 你要做什么 | 先读什么 |
|---|---|
| 接管项目或修改 setting | agents.md → docs/experiment-lines.md |
| 查实验事实、数值和状态 | docs/experiment-log.md |
| 判断一句话能否写进文章 | docs/claims-ledger.md |
| 查代码、数据、图、文献和远端位置 | docs/resource-index.md |
| 分配 agent、回收结果 | docs/coordination/README.md |
| 写正文或准备附录 | docs/report/index.html、docs/plans/plan-6-appendix-structure.md |

仓库规则和极简 setting 以 agents.md 为准。本文只负责导航，不复制另一份 SSOT。

## 当前论文主线

最小模型是 vanilla nanoGPT 加上可训练的 bigram/trigram clean value table。固定训练 shard 被多次顺序 replay 后，表先提供短暂的 held-out benefit，随后模型逐渐把训练 continuation 当成读出目标，validation damage 持续累积。

当前主线使用：

- input / wte 注入；bigram + trigram 各一张 clean table，R = 2^20；
- table RMSProp (β1, β2) = (0.0, 0.99)，table_lr_scale = 128.0，实际 LR 0.0768；
- backbone AdamW，LR 0.0006，betas (0.8, 0.95)，weight decay 0.1；
- warmup_constant，100 步 warmup，bf16，无 torch.compile；
- train loss 是更新前当前 batch 的 online loss，validation 是更新后 fixed val，gap = val − train；
- 标准诊断使用 bf16 forward + 异步 CPU 聚合，相关 run 使用 _fd 后缀。

这些坐标只在 agents.md §1 维护。旧的 2× table LR、β2 bug、移动窗口 val、current shell、Muon、RoPE、RMSNorm 和旧 multi-hash table 只能作为历史溯源，不能作为新主线默认值。

## 已完成的证据链

| 证据层 | 当前交付 | 权威入口 |
|---|---|---|
| 主现象 | 四臂 injection 对照、fixed-val gap 曲线 | docs/experiment-lines.md M2 / V5-refresh |
| 动力学 | pass 内漂移、backbone 驱动的 novel suppression、表 cosine 与 margin | docs/experiment-log.md §48–§52 |
| 直接分布证据 | 20-epoch logit sharpening probe；input 与 nogram 对照 | docs/experiment-log.md §53；docs/appendices/ls20ep_logit_stats/ |
| 理论与 toy | Markov exact、sampling law、residual-response moments、synthetic harness | tasks/README.md |
| 文献与 related work | over-encoding、Engram、duplication、Good–Turing citation 库 | docs/notes/literature/ |
| 正文与附录 | 主报告、背景页、实验 registry、C–H appendix plan | docs/report/、docs/plans/plan-6-appendix-structure.md |

文章中应保持窄口径：20-epoch replay 下，input 表臂的 logit shape 会全局锐化，但 novel context 的真实 token 概率不随之上升；“更自信”与“更正确”是两条轴。具体数字、step 和 seed 见 §53 与对应 run 目录。

## 运行入口

### 新的主线 GPU 实验

    bash code/cluster/setup_env.sh
    bash code/cluster/run_baseline.sh <gpu> <run_id> [steps]

run_baseline.sh 只适合非 table-size 的新实验。正式运行前必须登记 run_id、确认 GPU 空闲、同步并核对 code 的 md5、确认 train/val shard 不重叠。完整流程见 docs/coordination/README.md 和 docs/notes/method/common-commands.md。

### CPU / 本地验证

    python3 -m py_compile code/train.py code/diag_worker.py
    python3 tasks/l2_markov_exact/markov_clean_unigram.py
    python3 tasks/l3_sampling_law/gap_vs_samples_unigram.py
    python3 -m pytest ngram5_freq_gap/tests -q

CPU smoke 只能验证代码路径，不能替代正式实验。主线 run 的原始产物在 gitignored 的 data/runs_fixed/*_fixed/；不要把 data/runs/ 中的旧结果当作证据。

### 图表与报告

作图源码只放在 docs/plot_scripts/，生成图放在 docs/figs/，专题报告放在 docs/appendices/。先读 docs/plot_scripts/README.md，再运行对应脚本；不要在 HTML 里手写实验数字。

公开主报告的发布源是 sibling blog 仓库，当前 checkout 的只读副本是 docs/report/index.html。发布流程见 docs/sync_to_blog.sh 和 blog-deploy skill；README 更新不会自动发布博客。

## 代码和资源布局

    code/                  主线 nanoGPT、频率索引、诊断和 GPU launchers
    ngram5_freq_gap/       受控数据干预运行时，只动数据侧
    tasks/                 自包含的 toy / theory / synthetic 验证任务
    docs/experiment-*.md   实验全景、登记簿和 claim ledger
    docs/coordination/     多 agent 分工、登记、回收和证据包规范
    docs/notes/            方法、理论、数据、文献和研究判断
    docs/plot_scripts/     canonical 作图源码
    docs/figs/             当前图、理论图和历史素材
    docs/appendices/       面向论文附录的报告与提取产物
    docs/report/           对外 HTML 报告和历史版本
    data/                  gitignored 的 tokenized 数据、索引和 run 产物

按主题查找资源时使用 docs/resource-index.md，不要从文件名猜测某个结果是否权威。索引明确区分 current、historical、deprecated、orphan 和 remote-only。

## 文章阶段的工作规则

1. 先查 claim ledger，再决定一句话能否写成结论；没有 run_id / step / seed 的内容写成假设或背景。
2. 新实验遵守“一实验 = 一 run_id = 一个 _fixed 目录 = experiment-log 一行 + 一个 section”。
3. 修改 code/train.py、测量语义或数据划分后必须新起 run；不同诊断后缀不跨曲线直接混画。
4. 图、表、CSV 和文字必须能回链：图 → plot script → run_id → experiment log。
5. 多 agent 任务必须有 owner、输入、交付路径、验收条件和状态；只返回一段文字不算完成。
6. 大 setting 改动、删除已完成 run、跨集群搬运大文件、改分支或 push 前遵守 agents.md 的 P5 门槛。

## 维护状态

当前仓库处于论文收敛与资源整理阶段。0912 新增的 benefit-side、logit-sharpening、文献和 appendix 素材已经落地；下一步优先级是统一正文语言和符号、整理 appendix C–H、维护实验/协作索引，再处理低优先级的 V4.1-Flash SFT 攻击线。统一 backlog 和资源状态见 docs/resource-index.md；整晚执行队列见 docs/coordination/overnight-queue-20260913.md。

## 许可

Apache-2.0，继承自 nanoGPT upstream。
