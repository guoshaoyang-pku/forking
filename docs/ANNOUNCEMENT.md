# Introducing Forking

## Short introduction

What happens when a language model sees the same text again?

In **Forking: Sudden Overfitting Under Replay**, we study a sharp divergence between fitting and generalization in language models with local n-gram memory. Repeated exposure can drive training loss down while held-out predictions deteriorate. The timing follows replay, and the harm is concentrated in continuations the training corpus did not cover.

The study connects replay trajectories, context-level probability measurements, continuation coverage, and interventions on the memory pathway. The accompanying materials provide explicit baseline settings, original figures, and a small numerical packet readers can inspect and recompute on CPU.

The practical message: evaluate memory-augmented models across repeated passes, and inspect what happens to plausible continuations beyond those observed in training.

## 中文简介

同样的文本再学一遍，模型一定会更好吗？

**Forking: Sudden Overfitting Under Replay** 研究了一种突出的训练现象：在带有局部 n-gram 记忆的语言模型中，数据重放可以让训练损失继续下降，同时让未覆盖续接的预测变差。训练拟合与泛化开始分叉，而分叉的时间和增长与 replay 紧密相关。

我们把这一现象拆解为可以逐步核查的证据：跨轮次的损失轨迹、具体上下文中的概率变化、续接覆盖率与 gap 的关系，以及对局部记忆路径的干预。配套材料提供明确的基线配置、论文原图和可以在 CPU 上重算的小型数据包。

这提醒我们：评估记忆增强模型时，需要观察重复使用数据后的行为，以及训练中没有覆盖到、但仍然合理的续接。

## Visual assets

- [Share card, 1200 × 630](site-assets/share-card.png): title and a clearly labeled conceptual illustration.
- [Context examples](figs/context-examples.pdf): measured epoch-1/epoch-5 endpoints; seed 42.
- [Replay sharpening](figs/sharpening-vs-ce.png): a finite 20-pass diagnostic that keeps distribution shape separate from true-token cross-entropy.
- [Selected exemplars](figs/sharpening-exemplars.png): three context-level top-20 probability trajectories, explicitly labeled as selected examples.
- [Coverage relationship](figs/coverage.pdf): three-seed measurements with executable numerical checks.
- [Figure source manifest](figs/manifest.json): source identities and scope for every displayed paper figure.
- [Sharpening evidence packet](SHARPENING.md): CSVs, metric definitions, run coordinates, and limitations.

Publication copy should attach the approved final paper URL and author citation. The [release branch](https://github.com/guoshaoyang-pku/forking/tree/release) contains the curated materials; the [availability record](AVAILABILITY.md) identifies what remains before the final release.
