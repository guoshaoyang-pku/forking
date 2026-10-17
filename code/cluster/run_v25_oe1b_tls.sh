#!/usr/bin/env bash
# v2.5 OE selection at 1B (user 2026-10-18): 8L/768 (~50M non-embedding) x 1B ClimbMix
# unique tokens x E=3, per-order dictionary R / f limits.  V5 standard otherwise
# (input injection, RMSProp(0,0.99) x128, lr 6e-4 warmup_constant(100), bf16, no compile).
# freq index = 1B pool counts pruned to f>=30 (selection only); the final-step freq-bin
# diagnostics therefore mislabel f<30 contexts as novel and are NOT evidence.
#
#   run_v25_oe1b_tls.sh <gpu> <tag> <mode:nogram|hash|dict> <orders> <table_lr_scale> [per_order_spec] [epochs]
# Same as run_v25_oe1b.sh plus the table LR scale as a variable (separate file so the
# running round-1 launchers are never edited in place).
# e.g. run_v25_oe1b.sh 4 d1m dict 2,3,4,5
#      run_v25_oe1b.sh 5 r1m1m256k64k dict 2,3,4,5 "4:k=262144,5:k=65536"
set -euo pipefail
GPU="${1:?gpu}"; TAG="${2:?tag}"; MODE="${3:?mode}"; ORDERS="${4:-}"
TLS="${5:?table_lr_scale}"; SPEC="${6:-}"; EPOCHS="${7:-3}"
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_o2to5_1b.npz}"
SHARDS="$(cat "$ROOT/data/train_shards_1b.txt")"
RUN_ID="v25oe_${TAG}_tls${TLS}_p1b_s${NGLAB_SEED:-42}_e${EPOCHS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"

case "$MODE" in
  nogram) NG=(--enable_bigram 0 --enable_trigram 0 --enable_unigram 0) ;;
  hash)   NG=(--enable_unigram 0 --ngram_orders "$ORDERS" --bigram_clean_table 1048576
              --trigram_clean_table 1048576 --ngram_clean_table 1048576) ;;
  dict)   NG=(--enable_unigram 0 --ngram_orders "$ORDERS" --ngram_clean_table 1048576
              --ngram_dict_topk 1048576 --freq_index "$FREQ_INDEX")
          [[ -n "$SPEC" ]] && NG+=(--ngram_dict_per_order "$SPEC") ;;
  *) echo "mode: nogram|hash|dict" >&2; exit 2 ;;
esac

echo "[v25] run=$RUN_ID gpu=$GPU mode=$MODE orders=$ORDERS tls=$TLS spec=$SPEC E=$EPOCHS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" --out_dir "$OUT_DIR" --data_dir "$DATA_DIR" \
  --train_shards "$SHARDS" --train_max_tokens 1000000000 --train_epochs "$EPOCHS" \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 --seed "${NGLAB_SEED:-42}" \
  --dtype bf16 --injection_position input "${NG[@]}" \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size 72 --total_batch_size 147456 \
  --lr 0.0006 --lr_schedule warmup_constant --warmup_steps 100 \
  --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale "$TLS" \
  --val_interval 10 --val_batches 4 --table_norm_interval 50 --fixed_train_probe 0 \
  --freq_eval_interval 1000000 --exact_freq_eval_interval 1000000 \
  > "$RESULT_DIR/train.log" 2>&1
test -s "$RESULT_DIR/summary.json"
echo "[v25] complete: $RESULT_DIR"
