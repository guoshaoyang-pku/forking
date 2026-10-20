#!/usr/bin/env bash
# v26 D/N=20 invariant ladder (user 2026-10-20): hold D/N=20 and R/N=8
# (R = per-order exact rows), scale N in {35m,50m,104m,159m}; phenomenon
# invariance check: OE E1/E2/E3 benefit, forking gap trajectory, divergence.
# Pools are prefix/extension shards of the same ClimbMix corpus; selection
# counts come from the 1B-pool freq index (caveat registered in log).
#
#   run_v26_dn20.sh <gpu> <model:35m|50m|104m|159m> <arm:nogram|exact> [seed]
# env: NGLAB_PY, NGLAB_ROOT, NGLAB_FREQ_INDEX, NGLAB_ASSIGN_DIR
set -euo pipefail
GPU="${1:?gpu}"; MODEL="${2:?model}"; ARM="${3:?arm}"; SEED="${4:-42}"
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_o2to5_1b.npz}"
ASSIGN_DIR="${NGLAB_ASSIGN_DIR:-$ROOT/data/assignments}"
BASE22="1,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32"

TLS_OVERRIDE="${NGLAB_TLS:-}"
RK_OVERRIDE="${NGLAB_RK:-}"
case "$MODEL" in
  35m)  NL=6;  NE=640;  NH=5; RK=280;  DTOK=700000000;  DTAG=d0p7b; TLS=0.5; DB=72
        SHARDS="1,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26" ;;
  50m)  NL=8;  NE=768;  NH=6; RK=400;  DTOK=1000000000; DTAG=d1b;   TLS=1;   DB=72
        SHARDS="$BASE22" ;;
  104m) NL=10; NE=896;  NH=7; RK=832;  DTOK=2080000000; DTAG=d2p1b; TLS=1;   DB=72
        SHARDS="$BASE22,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60" ;;
  159m) NL=12; NE=1024; NH=8; RK=1272; DTOK=3180000000; DTAG=d3p2b; TLS=1;   DB=36
        SHARDS="$BASE22,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84" ;;
  *) echo "model: 35m|50m|104m|159m" >&2; exit 2 ;;
esac
[[ -n "$TLS_OVERRIDE" ]] && TLS="$TLS_OVERRIDE"
[[ -n "$RK_OVERRIDE" ]] && RK="$RK_OVERRIDE"

case "$ARM" in
  nogram)
    NG=(--enable_bigram 0 --enable_trigram 0 --enable_unigram 0) ;;
  exact)
    ASSIGN="$ASSIGN_DIR/assign_exact_r${RK}k.npz"
    [[ -f "$ASSIGN" ]] || { echo "assign npz missing: $ASSIGN (run dn20_make_exact_r.py)" >&2; exit 2; }
    CT=524288
    (( RK > 400 )) && CT=1048576
    (( RK > 832 )) && CT=2097152
    NG=(--enable_unigram 0 --ngram_orders 2,3,4,5 --ngram_clean_table "$CT"
        --ngram_assign_file "$ASSIGN" --freq_index "$FREQ_INDEX") ;;
  *) echo "arm: nogram|exact" >&2; exit 2 ;;
esac

TABLE=(--table_optimizer adamw --table_betas 0.9,0.95 --table_grad_clip 1.0
       --table_lr_scale "$TLS")
RUN_ID="v26dn_${MODEL}_${ARM}_r${RK}k_${DTAG}_tls${TLS}_s${SEED}_e3_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"

echo "[v26dn] run=$RUN_ID gpu=$GPU model=$MODEL arm=$ARM seed=$SEED D=$DTOK R/order=${RK}k tls=$TLS db=$DB"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" --out_dir "$OUT_DIR" --data_dir "$DATA_DIR" \
  --train_shards "$SHARDS" --train_max_tokens "$DTOK" --train_epochs 3 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 --seed "$SEED" \
  --dtype bf16 --injection_position input "${NG[@]}" \
  --n_layer "$NL" --n_head "$NH" --n_embd "$NE" --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size "$DB" --total_batch_size 147456 --val_device_batch_size 72 \
  --lr 0.0006 --lr_schedule warmup_constant --warmup_steps 100 \
  "${TABLE[@]}" \
  --val_interval 10 --val_batches 4 --table_norm_interval 50 --fixed_train_probe 0 \
  --freq_eval_interval 1000000 --exact_freq_eval_interval 1000000 --diag_interval 1000000 \
  > "$RESULT_DIR/train.log" 2>&1
test -s "$RESULT_DIR/summary.json"
echo "[v26dn] complete: $RESULT_DIR"
