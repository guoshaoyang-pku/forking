# A short route through the evidence

This repository is organized around three questions. Each has an observable result, an implementation path, and a check that can be run independently.

| Question | Start with | Then inspect |
| --- | --- | --- |
| Does replay separate fitting from generalization? | [Replay trajectories](figs/replay.pdf) | Fixed replay iteration, train-loss logging, and fixed-validation batches in [train.py](../code/train.py) |
| What changes for individual predictions? | [Context endpoint figure](figs/context-examples.pdf) | [Endpoint CSV](../evidence/context-examples/context_examples_plotted_values.csv) and [checkpoint manifest](../evidence/context-examples/context_examples_manifest.json) |
| How does the gap relate to missing continuation coverage? | [Coverage figure](figs/coverage.pdf) | [Independent fit calculation](../reproduce/verify_coverage.py) and [observations](../evidence/coverage/online_points.csv) |

## Run the small checks first

~~~sh
python3 reproduce/verify_coverage.py
python3 reproduce/verify_contexts.py
python3 tools/check_release.py
~~~

The first two commands require no third-party packages. They reproduce derived-data analysis and validate export consistency. The third checks release file identity and local website integrity. For selected model-measurement behavior, install requirements-test.txt and run python3 -m pytest -q.

## Code map

| File | Responsibility |
| --- | --- |
| [code/train.py](../code/train.py) | Model, n-gram addressing/injection, replay iterator, optimizer construction, training and fixed evaluation |
| [code/ngram_freq.py](../code/ngram_freq.py) | Context keys, global frequency lookup, and exact-frequency loss aggregation |
| [code/diag_worker.py](../code/diag_worker.py) | Asynchronous CPU aggregation of diagnostic observations |
| [code/table_occupancy.py](../code/table_occupancy.py) | Table occupancy diagnostics |
| [scripts/run_training.py](../scripts/run_training.py) | Explicit baseline settings and input validation before a training run |

## Definitions that affect interpretation

**Online gap** is fixed-validation CE after the update minus current-batch training CE before that update, at the same logged step. A fixed training probe is a different measurement. Subtracting the no-gram control produces a net gap; neither should be silently substituted for raw online gap.

**Novel continuation** means a next token absent after a context in training; the context itself can be seen. A completely unseen context has no training-side loss. Token membership categories in the example figure describe the corpus, not separate model inputs or a train/validation switch.

**Local memory** uses fixed context-to-row addressing with learned table vectors. Same-context examples share the addressing rule. A finite hash table can have collisions; table size alone is not a proof of injectivity.

**Evidence scope:** three-seed descriptive coverage fits, single-seed context illustrations, and manuscript intervention panels support different kinds of inference. See [EVIDENCE.md](EVIDENCE.md) for exact boundaries and [AVAILABILITY.md](AVAILABILITY.md) for training assets that remain to be supplied.

The trainer is an exact source copy. An older comment above fixed_val_batches mentions a moving validation window; the implementation immediately below captures and reuses fixed batches. The separate probe dataset also preserves the training iterator. Read those operations as the implemented behavior.
