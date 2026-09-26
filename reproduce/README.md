# Reproduce the packaged results

Start from the repository root with Python 3.10 or newer:

~~~sh
python3 reproduce/verify_coverage.py
python3 reproduce/verify_contexts.py
~~~

These commands are read-only and use the Python standard library. Expected output: PASS, eight coverage fits, and 72 context endpoints respectively. They can also be called by absolute path from another directory.

## Coverage analysis

The [coverage packet](../evidence/coverage/snapshot.json) pins three CSV files by SHA-256. The checker reads 1,512 observations and selects 432 observations from fig1v5_rp_both_s{42,43,44}_fixed at steps 670, 1000, 1340, and 2000. Each of 18 bins receives equal weight; frequencies range from 1 to 1,000, with at least 300 positions per side per seed.

For each branch and step it recomputes:

~~~text
mean online gap = slope × measured novel-continuation fraction + intercept
~~~

It checks slope, intercept, RMSE, relative RMSE, R², through-origin RMSE, alternating-bin holdout RMSE, and maximum sample standard deviation across seeds against the recorded results. Alternating-bin holdout trains on bins 1, 3, …, 17 and evaluates bins 2, 4, …, 18; it is not a held-out corpus or run.

| Context | Step | Slope | Intercept | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bigram | 2000 | 10.902815 | 1.889015 | 0.202266 | 0.990261 |
| Trigram | 2000 | 10.417929 | −0.786639 | 0.129244 | 0.995368 |

The fit is descriptive within the measured setting. Online gap uses post-update fixed-validation CE minus pre-update current-batch training CE at the same logged step. A context absent from training has no training-side loss and is excluded from this fit.

## Context endpoints

The [context packet](../evidence/context-examples/context_examples_manifest.json) records seed 42, checkpoint SHA-256 values, the input convention, and export semantics. The checker verifies log-probability conversion, corpus-membership classes, unique and complete epoch-1/epoch-5 pairs, and run/step coordinates. It does not recompute checkpoint inference, corpus counts, or the reranked “others” statistic from full-vocabulary logits.

## Redraw and rerun

~~~sh
python3 -m pip install -r requirements-figures.txt
python3 reproduce/plot_coverage.py --output outputs/coverage.png
~~~

This produces an independent visualization of the eight fits, with mean ± sample SD across seeds. The paper's original vector PDFs are preserved separately in [docs/figs](../docs/figs/manifest.json).

For GPU training prerequisites and the two baseline commands, see [TRAINING.md](../docs/TRAINING.md). Missing raw data and tokenizer assets are stated in [AVAILABILITY.md](../docs/AVAILABILITY.md).
