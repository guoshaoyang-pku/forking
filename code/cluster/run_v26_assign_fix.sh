#!/usr/bin/env bash
# v2.6 assignment-structure decisive arms under the adamw+clip fixed recipe
# (user 2026-10-20; §99 opt-fix verdict + fork-summary-v3-comments-20261019 C3
# result). Two arms on the W1.5 zero-collision / occupancy-structure question:
#   exact974k = per-ngram row, zero collision (3/3 diverged under RMSProp W1.5;
#               retest now that adam+clip is the deployment recipe). tls 2.0.
#   perm512k  = C3@512k assignment shuffled WITHIN frequency bin (same codebook
#               shape/traffic, semantics killed). If PERM ~= C3 -> occupancy
#               structure carries the benefit; if C3 > PERM -> semantic
#               clustering itself is worth money. tls 1.0 (deployment recipe).
# TABLE (both arms): adamw betas 0.9,0.95 + grad clip 1.0 (the §99 fix).
#
#   run_v26_assign_fix.sh <gpu> <arm:exact974k|perm512k> [seed] [epochs] [tls]
set -euo pipefail
GPU="${1:?gpu}"; ARM="${2:?arm}"; SEED="${3:-42}"; EPOCHS="${4:-3}"
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_o2to5_1b.npz}"
SHARDS="$(cat "$ROOT/data/train_shards_1b.txt")"

NG_BASE=(--enable_unigram 0 --ngram_orders 2,3,4,5
         --ngram_clean_table 1048576
         --freq_index "$FREQ_INDEX")
case "$ARM" in
  exact974k)
    TLS_DEFAULT=2.0
    NG=("${NG_BASE[@]}" --ngram_assign_file "$ROOT/data/assignments/assign_exact974k.npz")
    ;;
  perm512k)
    TLS_DEFAULT=1.0
    NG=("${NG_BASE[@]}" --ngram_assign_file "$ROOT/data/assignments/assign_perm_512k.npz")
    ;;
  *) echo "arm: exact974k|perm512k" >&2; exit 2 ;;
esac
TLS="${5:-$TLS_DEFAULT}"
TABLE=(--table_optimizer adamw --table_betas 0.9,0.95
       --table_grad_clip 1.0 --table_lr_scale "$TLS")

RUN_ID="v26fix_${ARM}_adamc_tls${TLS%.*}_p1b_s${SEED}_e${EPOCHS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"

echo "[v26fix] run=$RUN_ID gpu=$GPU arm=$ARM seed=$SEED E=$EPOCHS tls=$TLS"
CUDA_VISIBLE_DEVICES="$GPU" setsid nohup "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" --out_dir "$OUT_DIR" --data_dir "$DATA_DIR" \
  --train_shards "$SHARDS" --train_max_tokens 1000000000 --train_epochs "$EPOCHS" \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 --seed "$SEED" \
  --dtype bf16 --injection_position input "${NG[@]}" \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size 72 --total_batch_size 147456 \
  --lr 0.0006 --lr_schedule warmup_constant --warmup_steps 100 \
  "${TABLE[@]}" \
  --val_interval 10 --val_batches 4 --table_norm_interval 50 --fixed_train_probe 0 \
  --freq_eval_interval 1000000 --exact_freq_eval_interval 1000000 --diag_interval 1000000 \
  > "$RESULT_DIR/train.log" 2>&1 &
echo "[v26fix] launched pid=$! dir=$RESULT_DIR"
