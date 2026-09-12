# D1 paper language and symbol audit · 2026-09-13

This is a review queue generated from Feishu paper v2 rev 1082. It does not alter the cloud document or claim completion.

## Open markers

- L22: 【再加一段】
- L147: 和 table 相关的幂律关系。假如 table 撞车了，novel 有可能会被废掉？【待补充】【这个不重要，是上面的自然推论】
- L172: 实测闭合：f=1 环境 margin +1.08/pass、传递系数 1−p̂≈0.87、novel val 伤害 +0.94/pass；20 遍 gap 线性 0.664/pass（图 6-1）【几何衰减与线性增长的精确对应待核】。 备用引用：[17] 独立复现 PPL 大降而下游持平，与锐化-伤害解耦互证；[14] 梯度下降隐式偏置作锐化方向参照。
- L205: ［待补：多 epoch 动力学的解析预测及实测对齐；区分有限窗口中的近线性累计、增量衰减和长期平台，给出模型失效的观测条件。］
- L211: ［待补：学习率—步数—相干更新强度的关系、表侧饱和区间与 backbone 响应曲线；区分已有经验结果和仍未由模型推出的部分。］
- L217: ［待补：按验证 token 质量加权的积分表达、频率分布改变时的修正，以及固定步数／固定 pass 的配对验证。］
- L227: # 7 · 补充实验（预留）
- L229: 本节预留两项补充实验，用于检验机制的跨结构适用性与后训练风险。有限续接覆盖、局部预测环境和重复强化提供共同的检验线索，但不预设任意模型均会复现同样的现象。
- L231: ## 7.1 DeepSeek-like 模型的复现与干预（预留）
- L235: 可进一步比较低频屏蔽和熵正则等候选措施，评价其是否保留原有训练收益并改善绝对 val loss；这些是待验证的方案，不预设对具体模型已有净收益。［待补：基座与模块版本、与标准 setting 的对应、复现曲线、干预图表及适用边界。］
- L237: ## 7.2 小数据 SFT 压力测试（预留）
- L263: ［待补：SFT 数据构造、冻结与可训练对照、目标／替代续接的概率变化及整体能力评估。上述画板保留为待验证的风险设计；本节结论及 §8 中对应的后训练讨论，以实际实验支持为准。］
- L305: # Appendix（结构预留）
- L388: ［待补：代码版本、CLI、数据划分、online / fixed-probe、raw / net gap、F / f 映射和完整统计账目。］
- L445: # 本次修改记录与待核项（2026-09-08）
- L447: 按本轮要求，v2 的 Section 1 以绍阳现有手写稿为准，全部文字、原有块及“【引用】”“【再加一段】”标记保持不动。标题加入 AutoResearch 关联；Section 2 改为 AutoResearch／RSI、n-gram、Over-Encoding、data replay training、natural language features 五个方向。引用核对了原始论文、会议／期刊页面和官方仓库／报告，并明确标注来源类型。
- L459: - v1 中 MiniMax M2.5 事件的具体机制归因未获本轮核实，不能作为本文机制的证据；新 Section 8 未采用这一外部事件。历史“实验进行中”也不代表本轮核准了实时进度。
- L463: **同日补充修订（2026-09-08）。**按后续要求合并 §2 的 n-gram 与 Over-Encoding 介绍，补入数学工具相关文献 [13]–[15]；适度补清 §3–§7 的测量口径、因果解释、数学假设、可检验预测与 SFT 协议，并更正正文两处标准单目标交叉熵写法。§8.3 改为围绕实测频率幂律及条件续接长尾的可能解释。前一轮记录保留为历史记录，其中交叉熵等部分待核项已在本轮处理；原有图表、历史数值和其余提纲尚未全面复核，最终公式、图表与实验结论仍由绍阳确认。本轮没有开展或宣称完成新实验。Introduction、标题、§2 其余段落、§8 其余小节、附录及既有参考文献均未改动；v1、plan.md、英文 TeX 和 Overleaf 未同步修改。
- L465: **后半部分结构调整（2026-09-08）。**将原 §5.4 的预测与验证独立为新 §6，依次组织 replay、学习率、唯一数据／epoch 长度及容量／干预收益；原 §6 的跨模型与后训练内容归入新 §7，分为 DeepSeek-like 和小数据 SFT 两项补充实验预留。原 §7 的工作总结并入 §8 开头，局限并入 §8.6；原 §8 五个讨论方向保留。净验证收益段落及配图移入 §6.4，SFT 风险画板保留在 §7.2；只移动原生资源块，未重建或修改图像和画板内容。正文的相关章节引用随之调整，其他章节、v1、本地 plan.md 和 TeX 未同步修改。

## Suggested pass order

1. Resolve factual and measurement wording before stylistic edits.
2. Reserve F(i) for deduplicated context count and f(i) for novel continuation mass; keep exact-f only in historical figure labels.
3. Define gap(i,j), raw/net gap, and absolute validation damage once, then reuse the definitions.
4. Replace generic claims of necessity, universality, or exact geometry with setting-bounded statements unless a run_id/step/seed is attached.
5. Remove placeholder prose such as “再加一段” and convert unresolved sections to explicit prediction/design text.

## Claim-safe edit checklist

- Every numeric sentence links to a local experiment-log section or appendix artifact.
- Novel continuation has no online train loss; do not imply a novel train/val subtraction.
- “Power law” is limited to the measured frequency/range and fit; do not present it as a Good–Turing theorem.
- §7.1 and §7.2 remain proposals until a pinned model, data, run, and evaluation packet exists.

## Local draft disposition · 2026-09-13 continuation

The local review draft at docs/report/paper-v2-review-draft.md now supplies claim-safe replacements for the three analytical placeholders in §6.1–§6.4 and Appendix B. It deliberately keeps long-term plateau, cross-architecture transfer, and SFT/DeepSeek results as unverified predictions or protocols. The draft is ready for author review, while the Feishu and blog copies remain unchanged.

The following terms remain prohibited as unconditional article claims: “all n-gram memory”, “always”, “only post-training has multiple epochs”, “universal power law”, and any direct MiniMax causal attribution. They may appear only when explicitly labelled as a hypothesis, proxy, historical setting, or external report.
