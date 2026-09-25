# Minimal inference

A readable reference implementation rather than a production serving engine. The
model code covers the vision encoder and aligner, sliding-window plus compressed
sparse attention with its two-level indexer, engram n-gram lookups, MoE,
Hyper-Connections, and the DSpark forward path. Generation itself is plain
autoregressive sampling.

## Install

```bash
python -m pip install -r requirements.txt
```

## Convert Hugging Face weights

The runtime uses one converted checkpoint file per tensor-parallel rank. From
this directory:

```bash
export HF_CKPT_PATH=/path/to/DeepSeek-V4.1-Flash-HF
export SAVE_PATH=/path/to/DeepSeek-V4.1-Flash-TP8
export MP=8

python convert.py \
  --hf-ckpt-path "${HF_CKPT_PATH}" \
  --save-path "${SAVE_PATH}" \
  --model-parallel "${MP}" \
  --expert-dtype fp4 \
  --tokenizer-path "${HF_CKPT_PATH}"
```

Expert counts are inferred from the weight names, so they do not need to be
passed. `--tokenizer-path` points at whichever directory holds `tokenizer.json`
and `tokenizer_config.json`; they are copied into the converted checkpoint.

## Run the equivalent TXT and JSON examples

```bash
export CKPT_PATH=/path/to/DeepSeek-V4.1-Flash-TP8
export MP=8

INPUT_FILE=examples/example.txt ./run.sh
INPUT_FILE=examples/example_harmony.json ./run.sh
```

The two files express the same interleaved two-image prompt, so they produce
identical encoded prompts and input token IDs.

For interactive chat:

```bash
torchrun --nproc-per-node "${MP}" generate.py \
  --ckpt-path "${CKPT_PATH}" \
  --config config.json \
  --interactive \
  --temperature 0.6
```

For multi-node execution, pass the usual `torchrun --nnodes`, `--node-rank`,
`--master-addr`, and `--master-port` arguments before `generate.py`.

## Self-test

`model.py` builds a small model from the `ModelArgs` defaults and runs a prefill
plus 22 decode steps, exercising the real dense-fp8 / MoE-fp4 kernels. Weights
are uninitialized, so it checks shapes and kernel plumbing, not numerics:

```bash
python model.py
```
