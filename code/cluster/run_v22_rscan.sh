#!/usr/bin/env bash
# v2.2 table-capacity (R) scan on H200 — user directive 2026-10-17:
#   "关于词表大小：对于 n=2, 2~3, 2~4, 2~5, 2~6, 2~7 都扫10个点，1~之前的上界，可能是3e6"
# Hash clean tables, R rows per order (standard v5 table semantics; R < distinct
# is the collision regime — the sweep maps gap vs table capacity).  Standard
# setting otherwise (agents.md §1): input injection, backbone LR 6e-4
# warmup_constant(100), table RMSProp (0,0.99) x128, bf16, no compile,
# val/diag every 10 steps, multi-order freq index, per-token stats dumped at
# the final step (token_stats_step*_*.npz).
#
#   run_v22_rscan.sh <gpu> <orders> <R> [steps] [device_batch]
#   orders: "2" | "2,3" | "2,3,4" | "2,3,4,5" | "2,3,4,5,6" | "2,3,4,5,6,7"
#
# Memory guard (H200 141GB): per-order tables cost R*768*8 B (param + RMSProp
# state).  6 orders x R=3e6 ~= 110 GB + transients — tight but inside budget;
# if a config OOM occurs the queue's fail counter retires the LINE, not the GPU.
# Total batch is ALWAYS 147,456 tokens (microbatch only splits the same step).
set -euo pipefail

GPU="${1:?gpu id}"
ORDERS="${2:?orders, e.g. 2,3,4,5}"
R="${3:?table rows R}"
STEPS="${4:-2000}"
DEVICE_BATCH="${5:-${NGLAB_DEVICE_BATCH:-72}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_orders2to8_s1.npz}"

rtag() {
  case "$1" in
    1000000) echo 1m ;; 2000000) echo 2m ;; 3000000) echo 3m ;;
    100000) echo 100k ;; 300000) echo 300k ;;
    10000) echo 10k ;; 1000) echo 1k ;;
    *) echo "$1" ;;
  esac
}

TAG="n${ORDERS//,/-}"
RUN_ID="v22r_${TAG}_r$(rtag "$R")_s${NGLAB_SEED:-42}_${STEPS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"

# restart hygiene: an incomplete dir from a previous failed launch is moved
# aside so the requeued job can start clean (queue retries the same line).
mkdir -p "$OUT_DIR" "$ROOT/logs/failed_launch"
if [[ -d "$RESULT_DIR" && ! -s "$RESULT_DIR/summary.json" ]]; then
  mv "$RESULT_DIR" "$ROOT/logs/failed_launch/$(basename "$RESULT_DIR").relaunch.$(date +%s)"
fi
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }

# the multi-order index must exist before launch (built once per host)
WAITED=0
while [[ ! -f "$FREQ_INDEX" ]]; do
  (( WAITED >= 7200 )) && { echo "freq index never appeared: $FREQ_INDEX" >&2; exit 2; }
  [[ $WAITED -eq 0 ]] && echo "[v22r] waiting for freq index: $FREQ_INDEX"
  sleep 60; WAITED=$((WAITED + 60))
done

mkdir -p "$RESULT_DIR"
echo "[v22r] run=$RUN_ID gpu=$GPU orders=$ORDERS R=$R steps=$STEPS mb=$DEVICE_BATCH"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" \
  --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" \
  --train_shards 1 \
  --val_shards "${NGLAB_VAL_SHARDS:-2,3,4,5,6,7,8,9,10,6542}" \
  --seed "${NGLAB_SEED:-42}" \
  --steps "$STEPS" \
  --dtype bf16 \
  --injection_position input \
  --enable_unigram 0 \
  --ngram_orders "$ORDERS" \
  --bigram_clean_table "$R" \
  --trigram_clean_table "$R" \
  --ngram_clean_table "$R" \
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
echo "[v22r] complete: $RESULT_DIR"
