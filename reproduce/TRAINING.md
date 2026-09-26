# Train the baseline and its no-gram control

This is the smallest GPU training path for the fixed-replay setup. The two
configurations expose every active command-line setting. The no-gram control
changes only whether the bigram and trigram tables exist.

For a quick check of the released observations, use the CPU evidence commands
in the root README first. The training path below requires external inputs;
it is not a from-scratch data preparation pipeline.

## Inputs and environment

- Use a CUDA-enabled PyTorch installation, NumPy, and a GPU with BF16 support.
  Select the PyTorch wheel appropriate for your CUDA runtime. The launcher
  records the actual Python executable, PyTorch, NumPy, CUDA, and GPU.
- Place the eleven packed token shards in one directory and supply the
  corresponding frequency index. Exact filenames, sizes, SHA-256 checksums,
  and the train/validation split are in
  [the input manifest](../configs/data_manifest.json).
- These input files total **1,348,963,202 bytes**. The repository does not
  contain the raw corpus, token shards, tokenizer, frequency index, or weights.
  A public input download location and complete tokenizer/preparation
  provenance must be supplied separately. The checksums identify the baseline
  inputs; they do not replace that provenance or grant redistribution rights.
- The canonical batch is 72 sequences of length 2048, or 147,456 tokens per
  optimizer step. It requires a CUDA-capable GPU with BF16 support. A full
  run of this exported launcher has not yet been validated. A smaller GPU may
  require a separately documented batch/accumulation adaptation; changing
  the batch silently is not an exact reproduction.

The expected shard format is little-endian uint16, with 2,049 tokens per
BOS-aligned row. Training consumes the first 2,048 tokens as input and the
last 2,048 as targets. Vocabulary size alone does not identify a tokenizer.
The launcher accepts only input files matching the published byte identities.

## Check the inputs

These checks use only the Python standard library and read the input files.

```bash
python scripts/validate_data.py \
  --data-dir /path/to/tokenized \
  --freq-index /path/to/freq_index.npz
```

Success requires all twelve files to match. Missing files, incorrect sizes,
and incorrect hashes produce a nonzero exit status. Files are never changed.

## Inspect the command, then train

Run from the repository root. Without `--execute`, this prints the complete
command and starts no training; it does not assert that the inputs exist.

```bash
python scripts/run_training.py \
  --config configs/baseline.json \
  --data-dir /path/to/tokenized \
  --freq-index /path/to/freq_index.npz \
  --run-id baseline_s42_fd \
  --gpu 0
```

Append `--execute` to validate the inputs and GPU, then launch. The process
writes to `outputs/baseline_s42_fd_fixed/`. To train the matched no-gram
control, use `--config configs/nogram.json` and a new run ID such as
`nogram_s42_fd`. Use distinct run IDs for additional seeds or budgets. A
preexisting output directory is always rejected, including after an
interrupted run; retained logs make a failed attempt inspectable.

The launcher does not automatically choose an idle GPU. Run it on the GPU
allocated to your job. It makes only that GPU visible to the trainer.

## What is fixed

| Coordinate | Baseline |
| --- | --- |
| Backbone | 8 layers, 6 heads, width 768; LayerNorm, learned absolute positions, tied embeddings |
| Memory | One clean table per bigram/trigram branch, injected at the input |
| Capacity | 1,048,576 rows per table; width 768; one hash per branch |
| Backbone optimizer | AdamW, betas (0.8, 0.95), weight decay 0.1, learning rate 0.0006 |
| Table optimizer | RMSProp, no momentum, beta 0.99, learning rate 0.0768 |
| Schedule | Steps 1–100 warm from 0.25× to 1×; constant thereafter |
| Run | Seed 42, 1,000 steps, BF16, no compilation |
| Data | Fixed-order replay; train shard 1; disjoint validation shard IDs |
| Measurement | Fixed validation, online train loss, diagnostics every 10 steps |

Some dataclass and raw trainer CLI defaults are historical. Use the packaged
configurations rather than launching `code/train.py` with omitted arguments.
The configurations identify this minimal protocol; they do not claim to be
the complete configuration set for every figure in the paper.

## Read the outputs

- `launch.json` records the exact command, configuration, input verification,
  code hashes, and runtime versions.
- `train.log` contains the process output; it is retained after a failure.
- `train_log.jsonl` contains one row per validation step, with online
  `train_loss`, fixed `val_loss`, `gap`, `epoch`, and learning-rate multiplier.
- `freq_bin_loss.jsonl` and `exact_freq_loss.jsonl` contain the frequency
  diagnostics; `table_norm.jsonl` records table magnitudes.
- `summary.json` contains the trainer configuration and endpoint losses.
- `verification.json` is written only after successful process exit and
  artifact checks. It checks completed steps, finite losses, and
  `gap = val_loss - train_loss`. It does not require a preferred scientific
  result.

The online training loss is measured before the step's update on that
optimization batch; the fixed validation loss is measured after the update.
They have the same logged step, but are different samples at different
parameter timestamps. This gap must not be relabeled as a fixed-train probe
gap. Contexts absent from training have no training loss and no defined
train/validation gap.

The minimal configuration saves metrics without multi-gigabyte weights.
The original trainer supports optional analysis checkpoints, but they do not
include optimizer state and are not a resumable-training mechanism.
