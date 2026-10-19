#!/usr/bin/env bash
# v2.6 EOS-divergence optimizer-fix decisive arms (user 2026-10-19;
# divergence-problem-brief-20261019 §8): k256k@tls16 diverged 2/2 seeds under
# hand-written RMSProp(0,0.99) no-clip (s42 ~17100, s43 ~15310, §87). Two arms
# decompose the fix:
#   adamc  = table Adam(wd=0, betas 0.9/0.95, eps 1e-8) + grad clip 1.0 (the fix)
#   rclip  = RMSProp(0,0.99) unchanged + grad clip 1.0      (pure-clip control)
# Prediction A: both clean at E3 and E1 benefit kept -> optimizer hypothesis
# confirmed, fix becomes the deployment recipe. Prediction B: still diverges ->
# infra hypothesis upgraded, add per-step max|dRow|/min-v/hit-histogram probes.
#
#   run_v26_optfix.sh <gpu> <arm:adamc|rclip> [seed] [epochs]
set -euo pipefail
GPU="${1:?gpu}"; ARM="${2:?arm}"; SEED="${3:-42}"; EPOCHS="${4:-3}"
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_o2to5_1b.npz}"
SHARDS="$(cat "$ROOT/data/train_shards_1b.txt")"

NG=(--enable_unigram 0 --ngram_orders 2,3,4,5 --ngram_clean_table 1048576
    --ngram_dict_topk 1048576 --freq_index "$FREQ_INDEX"
    --ngram_dict_per_order "2:k=262144,3:k=262144,4:k=65536,5:k=16384")
case "$ARM" in
  adamc) TABLE=(--table_optimizer adamw --table_betas 0.9,0.95
                --table_grad_clip 1.0 --table_lr_scale 16.0) ;;
  rclip) TABLE=(--table_optimizer rmsprop --table_betas 0.0,0.99
                --table_grad_clip 1.0 --table_lr_scale 16.0) ;;
  *) echo "arm: adamc|rclip" >&2; exit 2 ;;
esac

RUN_ID="v26fix_k256k_${ARM}_tls16_p1b_s${SEED}_e${EPOCHS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"

echo "[v26fix] run=$RUN_ID gpu=$GPU arm=$ARM seed=$SEED E=$EPOCHS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
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
  > "$RESULT_DIR/train.log" 2>&1
test -s "$RESULT_DIR/summary.json"
echo "[v26fix] complete: $RESULT_DIR"
