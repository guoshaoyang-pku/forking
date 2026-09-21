# §3–§4 Review 与压缩施工 Handoff（2026-09-21）

> 交接对象：新进程，专职 §3–§4（机制章）的后续施工。
> 交接人：Verdent 主会话。§5/§6 由用户本人 review，**不要碰**。
> 本目录包含：review 全文、已做修改清单、修改前资产、canonical 数值表、待办与协作约束。

---

## 0. 仓库与提交快照

| 仓库 | 路径 | 相关 commit | push 状态 |
|---|---|---|---|
| overleaf-paper | `/Users/guoshaoyang/Desktop/workdir/overleaf-paper` | `22494d6`（§5/§6 重写）→ `d6816c6`（§3–§4 压缩） | **ahead 2，未 push** |
| ngram-gap-lab | `/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab` | `16d6805`（fig3 binned-only 开关+新图） | **ahead，未 push** |

编译：`cd overleaf-paper && latexmk -pdf main.tex`；当前全文 24 页（含附录），正文图 11 张。
协作者俞善斌、赵浩然在 Overleaf 网页端并行编辑（俞负责 §4.2 实验与正文、§5.1 已上传过 DeepSeek 文字；赵负责绘图数据）。**push 前必须 `git fetch && git merge origin/main`，且 push 需用户同意。**

### 图编号映射（正文）

| 修改前 | 修改后 | 文件 |
|---|---|---|
| Fig 3 frequency（灰点+拟合线） | Fig 3 frequency（**仅 binned display**） | `fig03_frequency.pdf` |
| Fig 9 G-T f(F) scaling | → 附录（Frequency-scaling fits 小节） | `fig_b_t_MF.pdf` |
| Fig 10 GT kernel vs 直拟 | → 附录（同上） | `fig_v5_good_turing_kernel.pdf` |
| Fig 11 table cosine/margin 4-panel | Fig 9（**待善斌 sharpening 图替换**） | `fig_v5_table_cosine_and_margin.png` |
| Fig 12 SFT combined | Fig 10 | `fig_sft_combined.pdf` |
| Fig 13 RLVR-MATH | Fig 11 | `fig_rlvr_math.pdf` |

---

## 1. Review 结论（2026-09-21，对修改前版本）

### §3（用户评价"已经比较好"，review 同意）
- 结构：3.1 setting / 3.2 现象特征 / 3.3 标度律（freq、capacity、corpus、epoch）/ 3.4 干预（low-freq mask、freeze）。叙事顺序"现象→律→机制"正确。
- 小问题（已修）：§3.4 两处笔误（"supports the the relationship"；"in ~\ref{sec:forking-factors}" 缺 Section）；§3.3 有一段注释掉的 parenthetical（corpus 幂律一致性检查），已从源码删除（git 历史可找回）。
- Fig3 与 §4 的拟合口径冲突（赵浩然在群聊提出）：§3.3 的指数 0.253/0.318 是 **binned（7 个 mass-pooled bin 等权 log-log）** 拟合；§4 的直拟 0.233/0.318 是 **unbinned 全距**。俞善斌曾提议"删 Fig3 直接 refer §4"——**review 否决**：Fig3 是现象层第一个定量锚点（带 within-bin 误差棒），应保留为 display；删/降级的是 Fig9/10。最终解法：§3.3 的拟合公式整块删除，Fig3 只保留 binned display，正文幂律系数只保留 §4 的一套（unbinned），冲突自然消解。
- §3.2 末尾已加跨架构一句（DeepSeek-like 复现正结果 + refer 附录 G），来自 §5 瘦身决策。

### §4（冗余诊断）
- α 指数在修改前正文出现**三套**：§3.3 binned（0.253/0.318）、§4 G-T missing-mass f(F)（0.340/0.383）、§4 gap 直拟（0.233/0.318）。信息密度低、审稿人必问。
- Fig9（f(F) scaling）与 Fig3 并非严格重复（f(F) 是新量），但正文价值只有两个系数；Fig10 与 Fig3 是"同一测量两种拟合口径"。→ 两图移附录，正文留系数。
- **必须显式写的弱点**：GT 预测指数（0.340/0.383）与 gap 直拟指数（0.233/0.318）有 **~0.10 系统差**；振幅差 bigram 26%、trigram 6%。修改前只写 "approximately captures"，太软。已改写为显式句 + 归因（validation token 加权 vs 等权 bin 平均、coverage 差异）。
- Fig11（4-panel）：(c)(d) 是 sharpening 证据，(a)(b)（table cosine、relative update）是支撑性证据。群聊已定：整图由善斌的 sharpening 新图替换，并补 epoch 动力学（引 §3 Fig6 staircase）；(a)(b) 随旧图进附录。**新进程等图，不要先动。**
- 注释掉的 "Further predictions and tests" 块已删。
- 公式 lead-in：用户要求"每个公式前一行解释"，已全部补齐（见 §3 清单）。

---

## 2. 已做修改清单

overleaf-paper `d6816c6`：
1. §3.3：删 eq:empirical-power 及拟合句 → 改为 binned display 说明 + 指向 §4；Fig3 caption 改为 binned-only 描述；删注释 parenthetical；capacity 公式 lead-in 补含义。
2. §3.4：修两处笔误。
3. §4.1：eq:novel-contribution / eq:av / eq:zt 前补一行解释；f(F) 系数句改引附录 Fig；GT-vs-直拟段重写（显式 26%/0.10 + 归因）；Fig9/10 两个 figure 环境移附录新小节 "Frequency-scaling fits"（caption 双空格已修）。
4. §4.2：删注释块；**其余未动**（等善斌）。
5. `figures/fig03_frequency.pdf` 替换为 binned-only 版。

ngram-gap-lab `16d6805`：
6. `docs/plot_scripts/plot_v5_fig04_frequency_exact.py` 加 `NGLAB_PAPER_BINNED` 环境开关：置 1 时静态图只画 binned means±std（无灰点、无拟合线、无 legend），并额外输出 PDF；默认行为不变（blog/interactive 不受影响）。重画 `docs/figs/main/fig_v5_s1_frequency_exact_f.{pdf,png,svg}`。

---

## 3. 修改前资产清单（本目录）

```
sec34-pre-edit-assets/
├── overleaf-tex/
│   ├── 03_empirical.tex                  # 修改前 §3（commit 22494d6）
│   └── 04_mechanism.tex                  # 修改前 §4（commit 22494d6）
├── figures/
│   ├── fig03_frequency_WITH_fit_and_graypoints.pdf   # 旧 Fig3（含灰点+拟合线）
│   ├── fig03_frequency_NEW_binned_only.pdf           # 新 Fig3（对照用）
│   ├── fig_b_t_MF.pdf                     # 旧 Fig9（现附录；文件未变）
│   ├── fig_v5_good_turing_kernel.pdf      # 旧 Fig10（现附录；文件未变）
│   └── fig_v5_table_cosine_and_margin.png # 旧 Fig11/现 Fig9（待替换）
├── plot_v5_fig04_frequency_exact_PRE_EDIT.py  # 修改前绘图脚本
├── overleaf-sec34-edit.patch              # 22494d6..d6816c6 的 sections+figures diff
└── nglab-fig-script.patch                 # 16d6805^..16d6805 的脚本+图 diff
```

更旧的资产也可从 git 取：`git -C overleaf-paper show 22494d6:<path>`、`git -C ngram-gap-lab show 16d6805^:<path>`。

---

## 4. Canonical 数值表（正文一致性核对用）

| 量 | bigram | trigram | 出处/口径 |
|---|---|---|---|
| G-T missing mass $\hat f(F)=CF^{1-\alpha}$ | C=0.981, α=0.340 | C=0.991, α=0.383 | unbinned 全距；正文 §4.1 + 附录 Fig |
| GT gap 预测 $g_jCF^{-\alpha}$ | 9.090 F^-0.340 (g_j=9.263) | 3.996 F^-0.383 (g_j=4.032) | 正文 §4.1 |
| gap unbinned 直拟 | 7.206 F^-0.233 | 3.759 F^-0.318 | 正文 §4.1 + 附录 Fig |
| ~~§3.3 binned 拟合 0.253/0.318~~ | — | — | **已删除，禁止再引用** |
| capacity $G(R)=0.02+AR^\beta$ | β=0.437 | β=0.745 | §3.3，未动 |
| corpus 20× epoch2/3 gap | 0.13 / 0.22 | | §3.3；与 §5 base gap 0.214@20220 闭环 |

绘图数据位置（赵浩然提供）：
- gt-kernel（旧 Fig10）：`/data4/guoshaoyang/ngram-gap-lab/data/runs_scaling/s1v5_128_marm_both_fixed/`（ophis-gpu）
- b_t_MF（旧 Fig9）：`/data4/guoshaoyang/ngram-gap-lab/data/freq_index.npz`（ophis-gpu）
- 本地 Fig3 数据：`ngram-gap-lab/docs/appendices/s1_scaling_three_axis/s1_frequency_exact_points.csv` + `s1_scaling_fits.csv`

---

## 5. 待办（新进程的职责）

1. **等善斌的 sharpening 图**替换现正文 Fig9（`fig_v5_table_cosine_and_margin.png`）：新图应含 sharpening 证据 + epoch 动力学（引 §3 Fig6 staircase）；旧图 panels (a)(b) 移附录。替换时同步 §4.2 文字（现 §4.2 只有 3 段推导+观测句，够用，勿扩）。
2. **幂律示意图**：用户明确"幂律很激动人心，可能放一个示意图"。留位不画；等善斌新图风格定了再画，避免二次改风格。建议内容=因果链：replay → coverage 不变 → missing mass $F^{1-\alpha}$ → gap $F^{-\alpha}$ → sharpening 放大。
3. 群聊结构分工落地（若与善斌/浩然不冲突）：4.1 = 理论 I+II（coverage 推导 + sharpening 推导），4.2 = sharpening 实证 + epoch 动力学。当前 §4.1/§4.2 切分已基本吻合，**不要大动**。
4. 赵浩然若要把附录 Fig（b_t_MF/gt_kernel）改成 bigram 单排、trigram 再附录：可做，但需与他确认数据管线（他的 fig9/10 是 unbinned，与附录现状一致）。
5. ICLR 正文 9 页目标：§3–§4 本轮已省 ~0.6 页；后续空间来自 §4.2 换图（4-panel→2-panel 或示意）与 §1/§2 压缩（不在本 handoff 范围）。

## 6. 协作约束（硬）

- 俞善斌、赵浩然在 Overleaf 网页并行编辑：**每次动手前 `git fetch origin && git merge`**；冲突优先保留他们的 §4.2/§5.1 内容。
- **push 需用户同意**（项目规则 P5）；当前 ahead 2 未 push 是故意的。
- 不碰 §4.2 图与文字（等善斌）、不碰 §5/§6（用户自审）、不碰 DeepSeek 附录文字（善斌已调）。
- 图一律走 ngram-gap-lab `docs/plot_scripts/` 出图 + 两仓库同批 commit（P7）；改图前先读 `.agents/skills/ngram-gap-plotting/SKILL.md`。
- 正文任何数值必须与 §4 表（上文 §4 节）一致；binned 0.253/0.318 已死，勿复活。

## 7. 复现命令

```bash
# 新 Fig3（binned-only，论文用）
cd /Users/guoshaoyang/Desktop/workdir/ngram-gap-lab
NGLAB_PAPER_BINNED=1 python3 docs/plot_scripts/plot_v5_fig04_frequency_exact.py
cp docs/figs/main/fig_v5_s1_frequency_exact_f.pdf ../overleaf-paper/figures/fig03_frequency.pdf

# 旧风格（灰点+拟合线，blog/interactive 用）：去掉环境变量同命令
# 编译
cd ../overleaf-paper && latexmk -pdf -interaction=nonstopmode main.tex
```
