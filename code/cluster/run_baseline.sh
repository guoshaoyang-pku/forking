#!/usr/bin/env bash
# Canonical clean-table baseline launcher for a new main-line experiment.
#
# Usage:
#   bash code/cluster/run_baseline.sh <gpu> <run_id> [steps]
#
# Non-table-size experiments use clean R_bigram = R_trigram = 2^20. R remains
# a scientific coordinate, so a table-size experiment must use a derivative
# launcher/command that names its alternate capacities.
set -euo pipefail

GPU="${1:?gpu id required}"
RUN_ID="${2:?run id required, without _fixed suffix}"
STEPS="${3:-1000}"
BIGRAM_R=1048576
TRIGRAM_R=1048576

if [ "$#" -gt 3 ]; then
  echo "usage: $0 <gpu> <run_id> [steps]" >&2
  exit 2
fi

case "$RUN_ID" in
  *_fixed)
    echo "run_id must not include _fixed; the launcher adds it exactly once" >&2
    exit 2
    ;;
esac
for value in "$STEPS"; do
  case "$value" in
    ''|*[!0-9]*)
      echo "steps must be a positive integer" >&2
      exit 2
      ;;
  esac
done
if [ "$STEPS" -lt 1 ]; then
  echo "steps must be a positive integer" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index.npz}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"

if [ ! -x "$PY" ]; then
  echo "Python interpreter not found or not executable: $PY" >&2
  exit 2
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "Tokenized data directory not found: $DATA_DIR" >&2
  exit 2
fi
if [ ! -f "$FREQ_INDEX" ]; then
  echo "Frequency index not found: $FREQ_INDEX" >&2
  exit 2
fi
if [ -e "$RESULT_DIR" ]; then
  echo "Refusing to overwrite existing run directory: $RESULT_DIR" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
echo "[baseline] run=$RUN_ID gpu=$GPU steps=$STEPS R_bigram=$BIGRAM_R R_trigram=$TRIGRAM_R"
echo "[baseline] output=$RESULT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" \
  --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" \
  --train_shards 1 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --seed 42 \
  --steps "$STEPS" \
  --dtype bf16 \
  --injection_position input \
  --enable_unigram 0 \
  --enable_bigram 1 \
  --enable_trigram 1 \
  --bigram_clean_table "$BIGRAM_R" \
  --trigram_clean_table "$TRIGRAM_R" \
  --n_layer 8 \
  --n_head 6 \
  --n_embd 768 \
  --vocab_size 8192 \
  --sequence_len 2048 \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --lr 0.0006 \
  --lr_schedule warmup_constant \
  --warmup_steps 100 \
  --table_optimizer rmsprop \
  --table_betas 0.0,0.99 \
  --table_lr_scale 128.0 \
  --val_interval 10 \
  --val_batches 4 \
  --freq_index "$FREQ_INDEX" \
  --freq_eval_interval 10 \
  --freq_eval_batches 4 \
  --exact_freq_eval_interval 10 \
  --table_norm_interval 10 \
  --fixed_train_probe 0 \
  > "$RESULT_DIR/train.log" 2>&1

test -s "$RESULT_DIR/summary.json"
test -s "$RESULT_DIR/train_log.jsonl"
echo "[baseline] complete: $RESULT_DIR"
