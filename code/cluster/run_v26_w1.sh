#!/usr/bin/env bash
# v26 W1 (teacher-free OE algorithm search, plan v2 2026-10-18): 8L/768 (~50M)
# x 1B ClimbMix unique x E=3, f>=256 selection, M=128k physical rows, table LR 4x.
# Routing arms: nogram | h1 | h2 | s1 | c1 | c2 | c3 (assign modes read an npz).
# exact-f256 reference = existing v25oe_f256_tls4 runs (not re-run).
# Differences from v25 tls launcher: assignment npz instead of dict_topk,
# fixed-train probe on (eval-mode Delta_train), diag still final-step only.
#
#   run_v26_w1.sh <gpu> <tag:h1|h2|s1|c1|c2|c3|nogram> <assign_npz|-> [table_lr_scale] [epochs]
set -euo pipefail
GPU="${1:?gpu}"; TAG="${2:?tag}"; ASSIGN="${3:?assign_npz_or_-}"
TLS="${4:-4}"; EPOCHS="${5:-3}"
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_o2to5_1b_f30.npz}"
SHARDS="$(cat "$ROOT/data/train_shards_1b.txt")"
HEADS=1
[[ "$TAG" == "h2" ]] && HEADS=2
RUNTAG="${NGLAB_RUNTAG:-}"
[[ -n "$RUNTAG" ]] && RUNTAG="_${RUNTAG}"
RUN_ID="v26w1_${TAG}${RUNTAG}_tls${TLS}_p1b_s${NGLAB_SEED:-42}_e${EPOCHS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"

case "$TAG" in
  nogram) NG=(--enable_bigram 0 --enable_trigram 0 --enable_unigram 0) ;;
  h1|h2|s1|c1|c2|c3)
      [[ -f "$ASSIGN" ]] || { echo "assign npz missing: $ASSIGN" >&2; exit 2; }
      NG=(--enable_unigram 0 --ngram_orders 2,3,4,5 --ngram_clean_table "${NGLAB_CLEAN_TABLE:-65536}" \
          --ngram_assign_file "$ASSIGN" --ngram_heads "$HEADS" \
          --freq_index "$FREQ_INDEX") ;;
  *) echo "tag: nogram|h1|h2|s1|c1|c2|c3" >&2; exit 2 ;;
esac

SAVE_ARGS=()
[[ "${NGLAB_SAVE_FINAL:-0}" == "1" ]] && SAVE_ARGS=(--save_final_model)

echo "[v26] run=$RUN_ID gpu=$GPU tag=$TAG assign=$ASSIGN tls=$TLS heads=$HEADS E=$EPOCHS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" --out_dir "$OUT_DIR" --data_dir "$DATA_DIR" \
  --train_shards "$SHARDS" --train_max_tokens 1000000000 --train_epochs "$EPOCHS" \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 --seed "${NGLAB_SEED:-42}" \
  --dtype bf16 --injection_position input "${NG[@]}" \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size 72 --total_batch_size 147456 \
  --lr 0.0006 --lr_schedule warmup_constant --warmup_steps 100 \
  --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale "$TLS" \
  --val_interval 10 --val_batches 4 --table_norm_interval 50 \
  --fixed_train_probe 4 --train_probe_mode uniform --probe_eval_interval 100 \
  --freq_eval_interval 1000000 --exact_freq_eval_interval 1000000 --diag_interval 1000000 \
  ${SAVE_ARGS[@]+"${SAVE_ARGS[@]}"} \
  > "$RESULT_DIR/train.log" 2>&1
test -s "$RESULT_DIR/summary.json"
echo "[v26] complete: $RESULT_DIR"
