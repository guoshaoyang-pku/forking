#!/usr/bin/env bash
# v2 n-gram order sweep (forking v2, item 2): gap-vs-frequency per order n.
# V5 standard setting (agents.md §1): input injection, clean single tables of
# R = 2^20 rows for EVERY order, backbone LR 6e-4 warmup_constant(100),
# table RMSProp (0,0.99) x128, bf16, no compile, val/diag every 10 steps.
# Only the set of over-encoded orders changes between arms.  Every arm logs
# freq bins / exact-f for all orders 2..8 from one multi-order index.
#
#   run_v2_order_sweep.sh <gpu> <orders|nogram> <seed> [steps] [extra train.py args]
#   orders: "5" | "2,3,4,5" | "nogram"
set -euo pipefail

GPU="${1:?gpu id}"
ORDERS="${2:?orders, e.g. 5 or 2,3,4,5 or nogram}"
SEED="${3:?seed}"
STEPS="${4:-2000}"
shift $(( $# < 4 ? $# : 4 ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_orders2to8_s1.npz}"
R=1048576

if [[ "$ORDERS" == "nogram" ]]; then
  TAG="nogram"
  TABLE_ARGS=(--enable_bigram 0 --enable_trigram 0)
else
  TAG="n${ORDERS//,/-}"
  TABLE_ARGS=(--ngram_orders "$ORDERS" --bigram_clean_table "$R"
              --trigram_clean_table "$R" --ngram_clean_table "$R")
fi
RUN_ID="v2ord_${TAG}_s${SEED}_${STEPS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"

[[ -d "$DATA_DIR" ]] || { echo "data unavailable: $DATA_DIR" >&2; exit 2; }
[[ -f "$FREQ_INDEX" ]] || { echo "multi-order index unavailable: $FREQ_INDEX" >&2; exit 2; }
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"
echo "[v2ord] run=$RUN_ID gpu=$GPU orders=$ORDERS seed=$SEED steps=$STEPS"

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
  --diag_workers "${NGLAB_DIAG_WORKERS:-4}" \
  "$@" \
  > "$RESULT_DIR/train.log" 2>&1

test -s "$RESULT_DIR/summary.json"
echo "[v2ord] complete: $RESULT_DIR"
