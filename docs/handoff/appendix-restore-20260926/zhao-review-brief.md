# 给赵浩然的 review 请求（可直接粘贴到飞书群 / 邮件）

**渠道（按优先级）**
1. 飞书群：你们现在就在群里讨论附录，最快。
2. 邮件：haoranzhao@stu.pku.edu.cn（他 Overleaf 提交用的邮箱，来自 `git log` 的 author 字段）。
3. 行级意见：让他在 Overleaf 网页端打开 `sections/appendix_e_theory.tex`，用 comment 功能逐条标，我这边能拉到并逐条落实。**请他不要直接改文件**——现在还有一个 codex agent 在往 Overleaf 推提交（今天 12:47 的 `4f0c470` 就改了附录 E 的措辞），直接改会被并发提交冲掉。

**截止时间**：正文投稿截止 = 北京时间今天 19:59，请他在那之前回。

---

## 可直接发的消息

浩然，附录的事我跟你对一下，只占用你十几分钟，只看**附录 E 的 E.1、E.2 两节**。

背景：昨晚有一个 AI agent 拿到"大幅精简附录"的授权后，把 A–H 全部压缩改写（提交 `0a0e317`），你手写的 E.1（条件续接尾部 / occupancy 推导）和 E.2（crowding-out）被改写并打散到后面的小节里。我已经按 git 历史把它们**逐字恢复**成附录 E 的前两节，现在 Overleaf 最新提交是 `1c59890`。你之前问"现在附录里还有人写的东西吗"——E.1、E.2 现在是你的原文，一个字没动，除了下面第 1 条我改的那一句。正文 §1–6 是善斌在负责，其他附录不用你看。

请你拍板 5 件事（都在 `sections/appendix_e_theory.tex`）：

**1. E.2 最后一句，我改了三处，请你确认或给我你的写法。**
- 改前（你的原文）："…the validation loss on V rises **by the same logit shift that lowers the training loss**. Section 4.2 **identifies** ln z, averaged over V, **with the depth D_j** of Equation (gap-decomp)."
- 改后："…the validation loss on V **rises relative to the training loss, since ℓ_u − ℓ_C = ln z for every u in V**. Section 4.2 **uses** ln z, averaged over V, **as an idealized model of the depth D_t** of Equation (gap-decomp)."
- 理由：(a) "上升量等于训练损失下降量"在一阶不成立——一步对 ℓ_C 的改变是 −η(1−2p_C+Σp_v²)，对 ℓ_u 是 η(p_u+p_C−Σp_v²)，两者一般不互为相反数；真正精确成立的是 Eq.(zt) 给的 ℓ_u−ℓ_C = ln z 增大。(b) 正文 §4.2 现在把 crowding-out 称为"理想化解释"，不再说"identifies"。(c) 正文里没有 D_j，`eq:gap-decomp` 里定义的是 D_t(F)。
- 备选写法：如果你想保留一阶公式，可以改成显式写出 ℓ_C 和 ℓ_u 各自的一阶变化量。

**2. 有一句 caveat 被删掉了，要不要加回？**
精简前的版本里 E.2 末尾有："In a parameterized network a small gradient step instead gives Δa ≃ −η JJᵀ(p−e_C), where J is the logit Jacobian; optimizer preconditioning changes this coupling further." 恢复你原文时这句没带回来。我建议加回：它说的是"理想化推导 ≠ 真实网络"，和现在正文 §4.3 用 mask 实验证明的"经验 locality"是互补的，不冲突。

**3. 记号统一。** 你 E.1 里写 𝔼[f_i(F)]，而 E.4 里 m_i(F) 本身已经定义成期望、又写成 𝔼 m(F)。请统一成一种（我建议 E.4 直接写 m_i(F)，不再套 𝔼）。

**4. §4.2 与你 E.2 的措辞对齐。** §4.2 写 "an idealized step **directly in logit coordinates**"，你 E.2 写 "one idealized gradient step"。要不要在 E.2 也加上 "directly in logit coordinates"？另外你 E.2 开头"these logits can move without moving those of other contexts"对 table-row 更新成立（哈希碰撞除外），§4.3 现在说独立性已被 mask 实验证明，你这句和它一致，可以不动。

**5. 两块旧内容要不要恢复。** (a) 你原来注释掉的两段（data volume / multi-epoch，带 `\pending` 注记）我删了，因为里面有 run ID 和 draft note；如果你想保留内容，我可以去掉注记后作为正文恢复。(b) 旧的 "Frequency-scaling fits" 小节（两张 Good–Turing 图、指数 0.287/0.314）**没有**恢复，因为"指数匹配"这个主张已被实验审计撤回；如果你想以修正口径恢复，告诉我。

另外你之前问参考文献：59 条我已经逐条对官方源（ACL Anthology / PMLR / NeurIPS / OpenReview / arXiv）核实，**全部真实存在**；修的是标题大小写保护、会议信息和 arXiv 编号。你今早改的 bib 条目我全部保留（做了三路合并，没有覆盖你）。剩两个需要作者决定的：Engram 那篇正式发表是 ACL 2026、且把模块改名为 Deep Sparse Embedding，正文仍叫 Engram（我在 bib 里加了 note 说明）；OPHIS blog 的署名里有我们自己（双盲风险），绍阳说不用处理。

麻烦你用 Overleaf comment 逐条回这 5 点就行，别直接改文件。

---

## 给绍阳的备注（不发群里）
- 他若对第 1 条不满意，最安全的回退是把 E.2 最后一句恢复成他的原文并把 D_j 改成 D_t；数学上那句仍不严格，但属于"理想化叙述"，可以接受。
- 若他要恢复第 5(b) 的 GT 指数小节，注意正文 §3/§4 已不再引用那两张图，恢复后需同步正文措辞，且要按 `docs/experiment-log.md` §67 的撤回口径写。
- 并发风险：codex agent 仍在推 Overleaf（`4f0c470` 改了附录 E 措辞）。建议在群里明确"现在起只有浩然/善斌/你改正文和附录，agent 停手"。
