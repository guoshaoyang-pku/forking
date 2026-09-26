# Forking: Sudden Overfitting Under Replay

**When the same text returns, fitting and generalization can take different paths.**

Forking studies a sharp, replay-aligned increase in the training–validation loss gap in language models with local n-gram memory. Controlled experiments connect this behavior to continuation coverage, memory capacity, and repeated exposure to a finite corpus.

[Website](https://guoshaoyang-pku.github.io/forking/) · [Paper (to be added)](#paper) · [License (to be selected)](#license)

[Reproduce the analysis](reproduce/README.md) · [Training protocol](docs/TRAINING.md) · [Evidence guide](docs/REVIEW_GUIDE.md) · [Availability](docs/AVAILABILITY.md)

![Local memory and diverging training and validation trajectories.](docs/figs/overview.png)

## Start here

Clone the release branch and run the two numerical checks with Python 3.10 or newer. The checks need no packages, GPU, or model download.

~~~sh
git clone --branch release --single-branch https://github.com/guoshaoyang-pku/forking.git
cd forking
python3 reproduce/verify_coverage.py
python3 reproduce/verify_contexts.py
~~~

The first command independently recomputes **eight coverage fits from three training seeds**, including residuals and an alternating-bin holdout check. The second checks **72 context-level endpoint observations**, their token classes, probability transforms, and checkpoint coordinates. Both print PASS and exit with code 0 on the packaged inputs; disagreements fail with a specific error.

These commands reproduce analysis of bundled observations. Training requires the additional data assets described in the [training protocol](docs/TRAINING.md).

## Three questions, one evidence trail

| Question | Evidence | What to inspect |
| --- | --- | --- |
| When does generalization separate from fitting? | Replay-aligned trajectories and repeated-pass controls | [Replay figure](docs/figs/replay.pdf) and fixed replay iteration in [train.py](code/train.py) |
| Which predictions are hurt? | Seen contexts with novel continuations; concrete context-level probes | [Measured endpoints](docs/figs/context-examples.pdf), [CSV](evidence/context-examples/context_examples_plotted_values.csv), [checkpoint metadata](evidence/context-examples/context_examples_manifest.json) |
| Does replay sharpen the distribution? | Shape and correctness separate: entropy/margin fall while novel-token CE can rise | [20-pass diagnostic](docs/figs/sharpening-vs-ce.png), [endpoint packet](evidence/sharpening/agg_epoch_boundary.csv), [scope](docs/SHARPENING.md) |
| Can coverage organize the gap? | Three-seed frequency-bin observations and affine fits | [Coverage figure](docs/figs/coverage.pdf), [observations](evidence/coverage/online_points.csv), [independent calculation](reproduce/verify_coverage.py) |

## Inspect the next-token predictions

The central prediction-level diagnostic is **replay sharpening**. Over 20 passes, the local-memory model becomes increasingly concentrated on a few continuations, while its validation cross-entropy on novel continuations rises. The matched no-gram control also becomes more concentrated, but its novel-token cross-entropy improves early and remains far lower. This separates distribution shape from correctness: a sharper distribution is not automatically a better one.

[![Twenty-pass logit-sharpening and cross-entropy diagnostic: shape metrics and true-token damage for local memory and no-gram control.](docs/figs/sharpening-vs-ce.png)](docs/figs/sharpening-vs-ce.png)

The figure reports seed 42, 128× table learning-rate scale, fixed shard-1 replay, and epoch-boundary samples from steps 337 through 6740. Solid lines are the local-memory input run; dashed lines are the matched no-gram control. Entropy and top-1 minus top-2 margin describe the output shape. Validation CE is the true-token damage axis. The plot is a finite-budget observation, not a claim of indefinite growth or universal behavior.

For an individual-context view, the companion panel follows the top-20 softmax probabilities in high-frequency, low-frequency, and novel contexts:

[![Exemplar context top-20 probability evolution over replay epochs.](docs/figs/sharpening-exemplars.png)](docs/figs/sharpening-exemplars.png)

These probes show endpoints at epochs 1, 5, 10, 15, and 20. They illustrate how a local distribution can become peaked; they are selected exemplars rather than a population estimate. The full aggregate CSV, exemplar CSV, metric definitions, run coordinates, and limitations are in [the sharpening evidence note](docs/SHARPENING.md).

The context probes below give a complementary, token-level comparison between the first and fifth pass. A context can occur in training while some plausible continuations remain unseen.

[![Measured next-token log probabilities for three contexts, comparing no-gram and local-memory models at epochs 1 and 5.](docs/figs/context-examples.png)](docs/figs/context-examples.pdf)

**Open points: epoch 1. Filled points: epoch 5. Higher means more probable.** Colors describe corpus membership. The measurements use seed 42 at steps 337 and 1685; connecting lines show two endpoints, not a monotone intermediate trajectory.

The input is BOS plus exactly three context tokens. “Others” is the log of the mean probability of the top 20 remaining tokens, reranked separately for each distribution. See the [measurement details](docs/EVIDENCE.md), [endpoint values](evidence/context-examples/context_examples_plotted_values.csv), and [checkpoint metadata](evidence/context-examples/context_examples_manifest.json).

### Sharpening and recovery

The paper's mechanism picture makes the proposed contrast explicit: replay can sharpen the continuation observed after a context, while cross-context updates can recover alternatives. The middle panel is labeled **“Sharpening”**; the blue annotation says **“Only no-gram can do”** for that recovery path. This is an explanatory schematic, so its bar heights are not measured probabilities.

[![Schematic of local sharpening and cross-context recovery, including the “Only no-gram can do” annotation.](docs/figs/mechanism.png)](docs/figs/mechanism.pdf)

The corresponding measured evidence is separate: context masking, gap contribution, and seen-versus-novel validation cross-entropy are shown below. In the $n$-gram model, replay increases train-seen mass and the novel-continuation CE curve rises over the measured epochs; the no-gram control follows a different trajectory. This panel is an intervention and aggregate diagnostic, not a claim that every individual context is monotone.

[![Measured sharpening evidence: train-seen mass, masked-context gap contribution, and seen-versus-novel validation cross-entropy.](docs/figs/sharpening-evidence.png)](docs/figs/sharpening-evidence.pdf)

See the [figure provenance](docs/FIGURES.html#mechanism) and [evidence scope](docs/EVIDENCE.md) before comparing the schematic with measured endpoints.

## Why it matters

Memory capacity can improve fitting while exposing a weakness under data reuse. Evaluating only the first pass can miss this later change. Forking makes that risk inspectable through repeated-pass controls, continuation-level measurements, and interventions on the local-memory pathway.

The fixed context-to-row mapping does not receive a train/validation membership flag. The same context uses the same addressing rule. The distinction emerges through learning and the finite set of continuations available in training. Broader claims about independent context subspaces or all sparse architectures require additional evidence.

## A small repository with explicit boundaries

~~~text
code/          Model, trainer, frequency measurement, diagnostics
configs/       Explicit local-memory and no-gram baseline settings
reproduce/     Numerical checks and independent replotting
scripts/       Data validation and portable GPU launch wrapper
evidence/      Small CSV packets and source/checkpoint identities
tests/         Measurement implementation checks
docs/          Project page, paper figures, and concise guides
tools/         Site rendering and release integrity checks
~~~

To redraw the coverage relationships from their observations:

~~~sh
python3 -m pip install -r requirements-figures.txt
python3 reproduce/plot_coverage.py --output outputs/coverage.png
~~~

To check selected measurement paths:

~~~sh
python3 -m pip install -r requirements-test.txt
python3 -m pytest -q
~~~

The numerical check, implementation tests, and GPU training have distinct scopes. [Availability](docs/AVAILABILITY.md) records what is included and what is still needed for a complete training rerun, final paper link, and licensed public release.

To preview the project page locally, run python3 -m http.server 8000 --directory docs and open http://localhost:8000. No website build is needed to view the included HTML.

## Paper

**Forking: Sudden Overfitting Under Replay** — paper link and citation to be added.

## License

License to be selected. A formal LICENSE file will be added here once the license is confirmed.
