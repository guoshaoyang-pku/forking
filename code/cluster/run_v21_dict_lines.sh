#!/usr/bin/env bash
# v2.1 no-collision dictionary line (user directive 2026-10-17):
#   "朴素 no-collision：每个 ngram 留 1M slot，n=2~5，同时和分别都测"
# Dictionary mode: per order, the top-1M train contexts (by count) get UNIQUE
# rows of the R = 2^20 table; misses contribute exactly zero.  Same slots as
# the hash line (R = 2^20), so dict-vs-hash isolates the collision/isolation
# factor.  Standard v5 setting otherwise (agents.md §1): input injection,
# backbone LR 6e-4 warmup_constant(100), table RMSProp (0,0.99) x128, bf16,
# no compile, val/diag every 10 steps, multi-order freq index.
#
#   run_v21_dict_lines.sh <gpu> <mode:dict|hash|nogram> <orders|nogram> <seed> [steps] [device_batch]
#   orders: "5" | "2,3,4,5" | ...
#
# A100-40GB: export NGLAB_DEVICE_BATCH=18 (accum auto-compensates; mb18x4 is
# the verified A100 precedent).  Arms with >= 2 tables need device_batch 12
# (4 tables x (param + RMSProp state) + optimizer transients overflow 40GB at
# mb18).  H200: leave unset (72).  Total batch is ALWAYS 147,456 tokens: the
# microbatch size only splits the same optimizer step (identical gradients).
set -euo pipefail

GPU="${1:?gpu id}"
MODE="${2:?mode: dict | hash | nogram}"
ORDERS="${3:?orders, e.g. 5 or 2,3,4,5 or nogram}"
SEED="${4:?seed}"
STEPS="${5:-2000}"
DEVICE_BATCH="${6:-${NGLAB_DEVICE_BATCH:-72}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_orders2to8_s1.npz}"
R=1048576

if [[ "$MODE" == "nogram" || "$ORDERS" == "nogram" ]]; then
  TAG="nogram"
  MODE="nogram"
  TABLE_ARGS=(--enable_bigram 0 --enable_trigram 0)
  DICT_ARGS=()
else
  TAG="${MODE}_n${ORDERS//,/-}"
  TABLE_ARGS=(--ngram_orders "$ORDERS" --bigram_clean_table "$R"
              --trigram_clean_table "$R" --ngram_clean_table "$R")
  if [[ "$MODE" == "dict" ]]; then
    DICT_ARGS=(--ngram_dict_topk "$R")
  else
    DICT_ARGS=()
  fi
fi
RUN_ID="v21_${TAG}_s${SEED}_${STEPS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"

[[ -d "$DATA_DIR" ]] || { echo "data unavailable: $DATA_DIR" >&2; exit 2; }
[[ -f "$FREQ_INDEX" ]] || { echo "multi-order index unavailable: $FREQ_INDEX" >&2; exit 2; }
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"
echo "[v21] run=$RUN_ID gpu=$GPU mode=$MODE orders=$ORDERS seed=$SEED mb=$DEVICE_BATCH"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" \
  --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" \
  --train_shards 1 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --seed "$SEED" \
  --steps "$STEPS" \
  --dtype bf16 \
  --injection_position input \
  --enable_unigram 0 \
  "${TABLE_ARGS[@]}" \
  "${DICT_ARGS[@]}" \
  --n_layer 8 \
  --n_head 6 \
  --n_embd 768 \
  --vocab_size 8192 \
  --sequence_len 2048 \
  --device_batch_size "$DEVICE_BATCH" \
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
  --diag_workers "${NGLAB_DIAG_WORKERS:-4}" \
  > "$RESULT_DIR/train.log" 2>&1

test -s "$RESULT_DIR/summary.json"
echo "[v21] complete: $RESULT_DIR"
