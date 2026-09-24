# Plan-7 Handoff：ICLR 2027 投稿冲刺（2026-09-25 04:10）

> **截止**：正文 2026-09-25 23:59 AOE = **北京时间 09-26 19:59**。正文 ≤ 9 页，超页直接 desk reject。
> **相关文档**：审阅方案 `docs/plans/plan-7-iclr27-submission-review.md`；实验登记 `docs/experiment-log.md` §64、§66；本文件。
> **数字口径**：本文所有数字都能追溯到 run_id / step / seed。标「单 seed」或「初步」的，写进正文前需要补证。

---

## 0. 一分钟版本

1. **§5 post-training 的数据 tokenizer 用错了（最严重）。** Magicoder、OpenR1、RLVR-MATH 三个源是用另一个词表编码的，论文 §5 的对应数字、Fig 8/9、摘要末句都要改；ID 源不受影响。
   - 已用正确 tokenizer 重跑（seed 42/43 已完成，44 在跑）。
   - 新结论：可训练表的 gap 是 no-gram 的 1.6–7.8 倍。
   - 冻表后，ID、Magicoder、OpenR1 三个源的 gap 回到 no-gram 水平（多出 ≤0.04）；RLVR 仍多约 0.13。
   - 所以旧表述「冻表后仍有残余 forking」基本不成立。
2. **正文超页。** 第 6 节 Future Work 整节落在第 10 页（占到页面约 30% 处），需要压掉约 1/4 页。
3. **Figure 1 右上「无重放时 OE 更好」只在第一遍成立。**
   - 128× table LR 下，单遍 both 在约 600 步后被 no-gram 反超（3 seed）。
   - 2× table LR 下，单遍 both 全程与 no-gram 持平或略好；重放 forking 同样强（2000 步 gap 5.73 vs 128× 的 5.64），但只有 seed 42。
4. **§4.2–4.3 你想讲的两条证据都有了。**
   - dog/cat 挤出：结果很强。
   - 跨 context 迁移：做了 48 对统计，n-gram 模型的迁移更低，而且随重放继续下降。
5. **Overleaf 状态。**
   - 已 push：RW 重写、参考文献修正、Reproducibility / LLM-use statements。
   - 线上又多了一个 svbird 的提交 `d04f9ee`（拆分附录 A–H），本地落后 1 个提交；下次编辑前先 `git -C paper pull --ff-only`。
6. **集群。**
   - ophis：E8 的 seed 44 还在跑，约 30 分钟。
   - 360-2：no-gram 多对探针还在跑，约 5 分钟；另有 2 个 run 因 OOM 失败，待重跑。

---

## 1. 任务与状态

| 你的要求 | 状态 | 产物 / 位置 |
|---|---|---|
| 审阅刘老师批注，给出修改建议 | ✅ | `docs/plans/plan-7-iclr27-submission-review.md`；`docs/plans/plan-7-assets/*.bib` |
| 先把可能要跑的实验跑掉，存数据和 final ckpt（E1/E2/E3） | ✅ | §64、§66。single-pass 臂数据有问题，已隔离并换成 clean 版（见 2.3） |
| RW 全权修改 | ✅ 已 push（`6e6b4d9`） | 线上已改名为 `sections/appendix_a_related_work.tex`，内容未动 |
| 参考文献核验与修正 | ✅ 已 push | 渲染出 56 条，0 条 undefined，每条都有出处 |
| AI use / Reproducibility statement | ✅ 已 push（`63db354`） | `sections/statements.tex`，线上新结构里仍然保留 |
| 优先 push Overleaf、定期 push、兼容线上修改 | ✅ | `scripts/push_to_overleaf.sh`：先 merge 线上（不 rebase）→ 编译检查 → 普通 push（不 force） |
| §4.2–4.3 dog/cat 例子 + 子空间干扰，跑 5-epoch ckpt | 数据 ✅ / 正文 ⏳ | 3 个 5-epoch run（每个 epoch 都存 ckpt）；探针结果见 2.4 |
| insight 示意图初步草案 | ✅ v1 | `docs/figs/plan7_insight_draft/insight_draft_v1.png`；脚本 `docs/plot_scripts/plot_plan7_insight_draft.py`。panel (d) 还是单对数据，要换成多对 |
| 标题 / 摘要更有野心，做更强更干净的实验 | ⏳ 未成稿 | 讨论要点见第 6 节 |
| Figure 1（合作者在画） | 对照数据 ✅ | clean spc/rp × 3 seed；E7 低 LR 对照（seed 42）已完成 |

---

## 2. 影响论文结论的新发现

### 2.1 SFT/RLVR 数据 tokenizer 错配（§5 必须改）

**错配情况**
- 主语料用的 tokenizer：`/data2/ncpl-pathA/work/vbird_autoresearch/cache/tokenizer/tokenizer.pkl`（与 `docs/notes/data/full-corpus-full163.md` 一致）。
- `ood_tokenized_magicoder/`、`ood_tokenized_openr1_10shards/`、`rlvr_tokenized/math/` 用的是 `/data4/shared/ph/.cache/autoresearch/tokenizer/tokenizer.pkl`。
- 两个词表的 8192 个 id 里，有 7635 个解码结果不同。
- `manifest.json` 里的 tokenizer 字段是 `prepare_ood_sft.py` 硬编码写入的，不可信。这是代码 bug，之后要修。

**修正**
- 旧 tokenizer 能逐字节复现旧 bin，所以流程本身没问题，只换 tokenizer 重新编码：`*_fixtok/` 三个目录，每个目录都带 `TOKENIZER.json`。

**重跑结果**（终点 gap；`runs_sft_fixtok/sft_{src}_fixtok_e5_{arm}_s{seed}`；基座：trigram 用 `sftbase_ep20_tri_ckpt`，no-gram 用 `s1v5_128_ep20_nogram_20p0xL4_3ep_ckpt`）：

| 源 | trainable | frozen | no-gram | 论文现写（旧，作废） |
|---|---|---|---|---|
| ID（tokenizer 本就正确，s42） | 2.463 | 0.341 | 0.316 | 2.463 |
| Magicoder（s42/s43） | 0.928 / 0.927 | 0.192 / 0.192 | 0.154 / 0.120 | 1.012 |
| OpenR1（s42/s43） | 0.686 / 0.688 | 0.159 / 0.156 | 0.157 / 0.141 | 0.726 |
| RLVR-MATH（s42/s43/s44） | 0.499 / 0.515 / 0.513 | 0.422 / 0.447 / 0.432 | 0.311 / 0.255 / 0.320 | 0.612 / 0.502 |

**要点**
- trainable / no-gram 的比值为 1.6–7.8 倍。
- 冻表去掉的超额 gap：ID 99%、Magicoder 95%、OpenR1 约 100%、RLVR 41%。
- 注意：trigram 基座的绝对 loss 比 no-gram 基座差。20× 预训练终点 val 为 3.35 vs 2.72；Magicoder 起点 4.40 vs 1.80。审稿人可能会问。
- 与 §3.3 的关系：预训练时在 epoch 1 后冻表，仍然 fork 且更糟（3 seed：gap 3.45/3.43/3.43，对照 2.72）。
- 可以写进正文的解释（**假设**）：
  - §3.3 里，冻结的表已经在 epoch 1 记住了正在重放的那份数据，backbone 可以继续利用它；
  - post-training 里，冻结的表只装着预训练统计，新语料无法写进私有坐标，只能走共享参数，所以表现和 no-gram 一样。
  - 这两个结果合起来，反而支持「可训练的私有坐标是 forking 的来源」。

### 2.2 Figure 1 与「无重放时 OE 更好」

**clean 对照**：3 seed 均值（val）；spc = 单遍（epoch20 shard 1,11–16），rp = 重放 shard 1；前 337 步两臂的 batch 逐位相同。

| step | spc both | spc no-gram | rp both | rp no-gram |
|---:|---:|---:|---:|---:|
| 300 | 4.816 | 5.119 | 4.822 | 5.128 |
| 600 | 4.385 | 4.368 | 4.809 | 4.354 |
| 1000 | 4.036 | 3.608 | 5.273 | 3.590 |
| 2000 | 3.599 | 3.213 | 6.909 | 3.324 |

- 重放下 2000 步 gap 为 5.644 / 5.633 / 5.679（sd 0.024）。单遍两臂的 gap 始终约 −0.02，不 fork。

**E7：table LR 对照**（seed 42，`fig1v5_{spc,rp}_both_tls{2,8,32}_s42`）

| table LR | 单遍 both − no-gram（300 / 600 / 1000 / 2000 步） | 重放 2000 步 gap |
|---|---|---|
| 2× | −0.40 / −0.28 / +0.02 / −0.06 | 5.73 |
| 8× | −0.33 / −0.12 / +0.23 / +0.06 | 6.09 |
| 32× | −0.26 / −0.01 / +0.39 / +0.31 | 5.66 |
| 128×（3 seed） | −0.30 / +0.02 / +0.43 / +0.39 | 5.64 |

**结论**
- forking 不是 128× 造成的：2×–128× 下 gap 都在 5.6–6.1。这可以直接回应「超参是为了放大效应挑的」这类质疑。
- 「OE 有益」只在第一遍稳健成立。
- 合作者的 `fig01_overview.pdf` 右上只画了 0–337 步（"benefit before repetition"）。只要 caption 写明是第一遍，就和数据一致。
- 2× 的 seed 43/44：360-2 上单遍两臂快跑完；重放两臂 OOM 失败，待重跑（见 4.2）。

### 2.3 数据独立性问题（已处理，但要知道）

- `data/tokenized/shard_00060–66` 不是新数据：逐行比对，它们与 train shard 1 或 val shard 2/3 完全相同。
- 用这些 shard 跑的 `fig1v5_sp_*`（ophis）和 `fig1v5_sp6_*`（360-2）作废，已移入两台机器各自的 `runs_quarantine_plan7/`，没有删除。
- 干净的单遍数据池是 `tokenized_epoch20/` 的 shard 11–16：行级零重叠；shard 1–10 和 6542 与旧目录逐字节相同。

### 2.4 §4.2–4.3 的证据

以下都来自 5-epoch ckpt `ep5ckpt_v5_{both,trigram,nogram}_s42`（360-2），seed 42。

**(a) dog/cat 挤出**：80 个低频 trigram context，train 里只跟 ` dog`、从不跟 ` cat`，held-out 里出现 ` cat`；只给 context 探针。

| 模型 | log p(dog)，epoch 1→5 | log p(cat)，epoch 1→5 | dog:cat log-odds |
|---|---|---|---|
| no-gram | −5.40 → −3.24 | −7.04 → −4.67 | 1.63 → 1.43 |
| both | −3.93 → −4.29 | −5.90 → **−9.53** | 1.98 → **5.24** |

- 真实例子 ` petting your`：train 里 F=3，只出现 1 次 dog；held-out 中 dog 15 次、cat 17 次。
  - both：p(cat) 0.0073 → 0.0008。
  - no-gram：p(cat) 0.020 → 0.066。
- 高频的 ` there is a`（F=5211）适合当示意图例子，但真实数据里的「挤出」应该用低频 context 来讲。

**(b) 跨 context 迁移，多对统计**（`code/tools/probe_pair_transfer.py`）
- 设计：48 对共享 bigram 后缀的 context（例如 "that is a" 和 "there is a"）；对照组是后缀不同、但续接词相同的 context。
- 指标 T：在 c1 上沿训练 LR 比例走一步，c2 获得的一阶增益占 c1 自身增益的比例。下表为中位数：

| 模型 | T_pair e1 → e5 | T_ctrl e1 → e5 |
|---|---|---|
| no-gram | 0.73 → 0.60 → 0.46（e4、e5 在跑） | 0.26 → 0.22 → 0.14 |
| both | 0.50 → 0.46 → 0.36 → 0.32 → 0.24 | 0.14 → 0.19 → 0.16 → 0.12 → 0.07 |
| trigram | 0.25 → 0.22 → 0.19 → 0.13 → 0.10 | 0.16 → 0.15 → 0.15 → 0.09 → 0.07 |

- **与你想讲的叙事对照**：数据支持「加了 n-gram 之后，that is a → dog 对 there is a → dog 的影响变弱」。
- **需要写准的 caveat**：
  - no-gram 的 T 也会随重放下降；
  - "that is a" 和 "there is a" 的 trigram 行不同，表本身对 T 的贡献约为 0，both 模型 T 下降主要来自 backbone 变得更 context-specific。
  - 更准确的说法是：「有私有行时，模型一开始就把 context 分开，重放后分得更开」，而不是「表挡住了迁移」。

**(c) 比例放大（初步）**
- 做法：低频 context 里 top1 与 top2 续接的训练样本比为 a:b，回归模型在 held-out 上的 log-odds 对 log(a/b) 的斜率。
- 结果：both 2.25 → 2.32，no-gram 1.22 → 1.92（epoch 3）。
- 这对应你说的「dog 比例过高会把 cat 异常压低」，但目前只是回归斜率，可以放附录作为支持。
- 另外，ophis 上 `data/queue_logs/plan7_major_ctx.json` 有 691 个 dog 和 cat 都出现过、但次数不等的 context，还没有评估。

**(d) 单一最小对**（18 个前缀，"that is a → dog" 对 "there is a → dog"）
- T：no-gram 0.98 → 0.65，both 0.85 → 0.47。
- 适合当示意图的例子，定量结论用 (b) 的多对统计。

**你提出的正文结构评估**：6a → dog/cat 统计 → 多对 T → 7a（mask 后 gap 不再增长）→ 7b，其他放附录。这个结构可行，证据都齐了。

---

## 3. 论文（Overleaf）状态

- **本地镜像** `paper/`：HEAD 是 `63db354`（我最后一次 push），线上 HEAD 是 `d04f9ee`（svbird，09-25 03:51，拆分附录 A–H、更新 OPHIS / mechanism / DeepSeek 协议）。本地没有未提交的修改，`git -C paper pull --ff-only` 可以直接快进。
- **本轮已 push**：
  - `8251dfc`：合并线上 haoranzhao 的附录修改；
  - `e6617e0`：补上缺失的 `huang2024mlagentbench`；
  - `6e6b4d9`：RW 重写成 6 个主题段 + bib 新增 19 条、修正 16 条（Engram 引 arXiv v2，21 位作者全列）；
  - `63db354`：Reproducibility / LLM-use statements。
- **编译**：`paper.tex` 能编译，0 条 undefined，共 29 页（以 `63db354` 计）。`main.tex` 与 `paper.tex` 内容相同，push 脚本编译的是 `paper.tex`。
- **页数**：第 6 节 Future Work 从第 10 页顶部开始，到约 30% 处结束。**正文超出约 1/4 页。**
- **遗留占位**（按 `63db354` 统计，拆分后要重查）：
  - `\pending{...}` / `\figpending{...}` 若干处；
  - "the Chinese draft" 1 处。
- **plan-7 里仍未改的正文问题**：
  - 摘要里 "larger training budgets" 应改为 "larger training corpora"；
  - 摘要末句（冻表那句，见 2.1）；
  - §4.2 文字与 5-epoch 版 Fig 6b 的数字不一致；
  - "no-gram stays near zero"：20 epoch 时是 1.01；
  - "three decades"；
  - Fig 5b caption 要解释 online train loss 的锯齿；
  - Fig 9 没有台阶的原因要解释；
  - "which we consider reasonably complete" 要删。
  - 合作者的大版本可能已经改掉了一部分，需要对照线上核实。
- **匿名性**：svbird 刚更新了附录 B（OPHIS），需要检查有没有第一人称或可识别的细节。

---

## 4. 集群状态

### 4.1 ophis-gpu（`/data4/guoshaoyang/ngram-gap-lab`）

- **在跑**：`data/queue_logs/plan7_e7e8_watcher.py`，日志是同目录下的 `plan7_e7e8_watcher.log`。
  - E7（6 个 run）已全部完成。
  - E8 seed 43（9 个 run）已完成。
  - E8 seed 44：进行中，共 9 个，2 个在跑、7 个排队，约 30 分钟跑完（前提是 GPU 3/4/5 一直空闲）。
- **停止**：`pkill -f plan7_e7e8_watcher.py`。已经启动的 `train.py` 不会跟着停；如需停止，按 run_id 单独 kill。
- **产物**（每个 run 都有 `final_model.pt`）：

| 内容 | 路径 |
|---|---|
| Fig 1 对照 | `data/runs_fixed/fig1v5_{spc,rp}_{both,nogram}_s{42,43,44}_fixed` |
| E7 | `data/runs_fixed/fig1v5_{spc,rp}_both_tls{2,8,32}_s42_fixed` |
| E3 | `data/runs_fixed/causalv5m_mask_low_le8_e1_{r1,s43,s44}_fixed`、`causalv5c_freeze_table_e1_128x_{s43,s44}_fixed` |
| E2 no-gram 20× 基座 | `data/runs_scaling/s1v5_128_ep20_nogram_20p0xL4_3ep_ckpt_fixed/final_model.pt` |
| post-training（正确 tokenizer） | `data/runs_sft_fixtok/`；ID 的 no-gram 臂在 `data/runs_sft/sft_id_e5_nogram_s42` |
| 重编码数据 | `data/ood_tokenized_magicoder_fixtok/`、`data/ood_tokenized_openr1_10shards_fixtok/`、`data/rlvr_tokenized/math_fixtok/` |
| RLVR 原始数据（已从 /dev/shm 落盘） | `data/rlvr_raw/`、`data/rlvr_tokenized/math/` |
| 作废（错 tokenizer，只标记、未移动） | `data/runs_sft/sft_{ood_magicoder,ood_openr1_10shards,rlvr_math}_e5_nogram_s42`，以及 §57/§58 和 RLVR 静态代理里的对应臂 |
| 隔离 | `data/runs_quarantine_plan7/`（shard 60–66 那批 sp 臂） |

### 4.2 360-2（`/data/home/guoshaoyang/ngram-gap-lab-plan7`，独立代码目录；`train.py` md5 与本地 HEAD 一致，`2d4bcb49…`）

- **5-epoch ckpt**：`data/runs_fixed/ep5ckpt_v5_{both,trigram,nogram}_s42_fixed`，含 `epoch{1..4}_model.pt` 和 `final_model.pt`。
- **探针**：结果在 `data/probe_subspace/`。
  - 单对探针的 JSON 已拷回本地 `docs/appendices/subspace_probe_20260925/`。
  - 多对探针 `pairs_*.json` **还没拷回**；no-gram 那份还在跑，约 5 分钟。
- **E7 的 2× seed 43/44**：
  - 单遍两臂即将跑完；
  - 重放两臂 `fig1v5_rp_both_tls2_s4{3,4}` **OOM 失败**（同一张卡上另一个进程占了 98 GB）。
  - 重跑方法：先把这两个失败目录移进 `runs_quarantine_plan7/`，再用 `queue/e7seeds.sh` 里的同一组参数重跑。
- **其他目录**：
  - `data/tokenized_spc/`：软链目录，shard 1、11–16 指向 `~/ngram-gap-lab/data/tokenized_epoch20_ophis`（sha 校验过），val shard 指向 `tokenized/`；
  - `runs_quarantine_plan7/`：作废的 sp6 臂；
  - `data/tokenized_sp_PARTIAL_DO_NOT_USE/`：中转中断留下的残文件。
- **注意**：Mac 中转到 360-2 只有约 160 KB/s，不要再中转 GB 级文件。

---

## 5. 下一步（按优先级，提交前）

**P0**
1. `git -C paper pull --ff-only`，同步 svbird 的附录拆分，之后所有编辑都基于线上最新版。push 统一用 `bash scripts/push_to_overleaf.sh`。
2. **重写 §5。**
   - 用 `runs_sft_fixtok/` 的结果重画 Fig 8/9，加上 no-gram 臂。ID 数据来自 `runs_sft_freq10/` 和 `runs_sft/`。
   - 绘图脚本是 `docs/plot_scripts/plot_sft_paper_style.py` 和 `plot_rlvr_static_proxy.py`，需要改成读新的 run 目录；`fig_sft_combined.pdf` 目前找不到生成来源。
   - 同步修改文字和摘要末句。
3. **正文压回 9 页**：Future Work 目前整节落在第 10 页。
4. **匿名性**：检查附录 B（OPHIS）的最新改动。
5. **清掉所有 `\pending` / `\figpending` / "Chinese draft" 占位。**

**P1**
6. 按 2.4 重写 §4.2–4.3。insight 图由你设计；草案 v1 的 panel (d) 换成多对统计的中位数，panel (c) 可以加 ` petting your` 这个真实例子。
7. 拍板 Figure 1 右上用哪种口径（见第 6 节）。
8. 拍板标题和摘要。

**P2（记账）**
9. 重跑 360-2 上 OOM 失败的 2 个 run；把 `pairs_*.json` 拷进 `docs/appendices/subspace_probe_20260925/`。
10. 回填 §66：
    - 补 E7、E8 和多对探针结果；
    - 修正结论句：「冻表后 gap 回到 no-gram 水平」对 RLVR 不成立（0.43 vs 0.30，3 seed）。
11. 在日志索引表里补 §64 那一行；为新图补 FIGURE_PROVENANCE；修 `prepare_ood_sft.py` 硬编码 tokenizer 字符串的 bug。

---

## 6. 需要你拍板 / 讨论要点

1. **标题与摘要。**
   - 现在最强、最干净的证据有四条：
     - forking 对 table LR 2×–128× 都稳健；
     - 3 seed 的 sd 只有 0.02；
     - 单遍对照不 fork，说明触发条件就是重放；
     - post-training 中可训练表是 no-gram gap 的 1.6–7.8 倍，而冻表后基本回到 no-gram 水平。
   - 最弱的是：在我们的极简设置下，「OE 有益」只在第一遍成立。所以 OE 在生产规模上的收益应该引文献（Engram、Over-Tokenized、SCONE），不要靠自己的数据来证明。
   - 标题候选（沿用 plan-7 §2.2）：
     - "Forking: Lookup Memory Turns Data Replay into Overfitting"
     - "Forking: N-gram Memory Makes Language Models Overfit When Data Repeats"
   - 摘要末句建议改为：
     > These trends predict that short, heavily repeated regimes such as SFT and RL should be prone to forking. Continued training on small replayed corpora confirms this: a trainable n-gram table yields 1.6–7.8 times the train–validation gap of a model without it, and freezing the table removes nearly all of the excess on three of four sources.
2. **Figure 1 右上用哪种口径。**
   - A：128×，只画第一遍。和全文设置一致，合作者现在的图就是这样。
   - B：2× table LR，单遍全程持平或略好，重放同样 fork。视觉上更强，但 Figure 1 的设置就和正文不同了，需要补 2× 的 3 seed。
3. **§5 的基座。**
   - 保持 trigram-only 基座：现状，需要说明它的绝对 loss 比 no-gram 差。
   - 或者改用 both-table 基座重跑：20× 预训练约 2 小时，加 12 个 SFT run 约 40 分钟。今天白天开跑来得及。
4. **OPHIS 的引用方式。** plan-7 的遗留问题，需要和刘老师确认。
5. **§4.2–4.3 的图怎么分配到正文和附录。** 你的方案是 6a + dog/cat + 多对 T + 7a/7b，其余放附录。

**如果还要做「更强更干净的实验」，按性价比排序：**
1. 2× 的 Figure 1 补齐 3 seed：只差 2 个 run，约 10 分钟。
2. 以 both-table 为基座重跑 §5：约 3 小时。
3. 多对迁移补 3 seed：需要 ep5ckpt seed 43/44 共 6 个 run 加探针，约 1 小时。

---

## 7. 注意事项

- **本地仓库里的工作副本**：
  - `~/Desktop/workdir/forking-paper-local`：本轮建的临时 clone（分支 `local/plan7-rw`，push 地址已禁用），没有独有内容，可以手动删除；
  - `~/Desktop/workdir/overleaf-paper`：更早的只读 clone。
  - 两个都已过时，不要在里面编辑。论文只在 `paper/` 里改。
- **lab 仓库的暂存区**：里面有别人暂存的 paper-mirror 重构改动（`paper/` 下的删除、`agents.md`、`.gitignore` 等），我没有动；我的 commit 都用 pathspec 只提交自己的文件。以下未跟踪文件也不是我的：`docs/figs/continue_pretrain_*`、`rlvr_static_proxy_*` 的修改，`docs/appendices/gt_independent_20260920/`，`code/prepare_rlvr_proxy.py`，`code/tools/compare_ckpt_logits.py`。
- **安全策略拦过的两类操作**：
  - 读取其他用户的目录（`/data1/shared/vanilla_user_directory`）；正确的 tokenizer 在 `/data2/ncpl-pathA/...`，可以正常读。
  - 集群命令里带 `rm -rf`。以后隔离文件一律用 `mv`。
- **lab 仓库只 commit 在本地，没有 push**（按 P5，push 需要你同意）；Overleaf 按你的授权定期 push。
