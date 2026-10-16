#!/usr/bin/env bash
# v2.3 training-data envelope on H200 — user directive 2026-10-17:
#   "关于训练数据：n=2, 2~3, 2~4, 2~5, 2~6, 2~7 都是同时开，词表都是1M，
#    1B的数据训练会很慢，1B取一个点，最少10M，总共插15个点即可"
# Pool = first P content tokens of the train shard stream (--train_max_tokens,
# nested-prefix).  Fixed 2000 steps (295M tokens trained) per point: pool 10M
# sees ~30 epochs (heavy forking), pool 1B sees 0.3 epochs (no repetition) —
# the envelope maps the forking gap against repetition.  NO freq index: the
# per-token dump (loss + per-order hashed context keys at the final step)
# carries the post-hoc gap-freq statistics; pool-relative f is computed
# offline against the same hash.
#
#   run_v23_dataenv.sh <gpu> <orders> <pool_tokens> [steps] [device_batch]
#   orders: "2" | "2,3" | "2,3,4" | "2,3,4,5" | "2,3,4,5,6" | "2,3,4,5,6,7"
#
# Train shard list: NGLAB_ENV_TRAIN_SHARDS (default "1"; extended pools append
# freshly tokenized shards, e.g. "1,67,68,..."; val stays 2-10+6542).
set -euo pipefail

GPU="${1:?gpu id}"
ORDERS="${2:?orders, e.g. 2,3,4,5}"
POOL="${3:?pool content tokens, e.g. 10000000}"
STEPS="${4:-2000}"
DEVICE_BATCH="${5:-${NGLAB_DEVICE_BATCH:-72}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"

ptag() {
  case "$1" in
    10000000) echo 10m ;; 14000000) echo 14m ;; 19000000) echo 19m ;;
    27000000) echo 27m ;; 37000000) echo 37m ;; 52000000) echo 52m ;;
    72000000) echo 72m ;; 100000000) echo 100m ;; 139000000) echo 139m ;;
    193000000) echo 193m ;; 268000000) echo 268m ;; 373000000) echo 373m ;;
    519000000) echo 519m ;; 722000000) echo 722m ;; 1000000000) echo 1b ;;
    *) echo "$1" ;;
  esac
}

TAG="n${ORDERS//,/-}"
TRAIN_SHARDS="${NGLAB_ENV_TRAIN_SHARDS:-1}"
RUN_ID="v23d_${TAG}_p$(ptag "$POOL")_s${NGLAB_SEED:-42}_${STEPS}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"

# restart hygiene: an incomplete dir from a previous failed launch is moved
# aside so the requeued job can start clean (queue retries the same line).
mkdir -p "$OUT_DIR" "$ROOT/logs/failed_launch"
if [[ -d "$RESULT_DIR" && ! -s "$RESULT_DIR/summary.json" ]]; then
  mv "$RESULT_DIR" "$ROOT/logs/failed_launch/$(basename "$RESULT_DIR").relaunch.$(date +%s)"
fi
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }

# warn (not fail) when the pool exceeds what the shard list actually holds
AVAIL=$("$PY" - "$DATA_DIR" "$TRAIN_SHARDS" <<'PYEOF'
import sys, os
d, shards = sys.argv[1], [int(x) for x in sys.argv[2].split(",") if x.strip()]
tot = 0
for s in shards:
    p = os.path.join(d, f"shard_{s:05d}.bin")
    if os.path.exists(p):
        tot += os.path.getsize(p) // 2 // 2048 * 2048
print(tot)
PYEOF
)
if (( POOL > AVAIL )); then
  echo "[v23d] WARNING: pool $POOL > available $AVAIL tokens from shards [$TRAIN_SHARDS]; using the full stream" >&2
fi

mkdir -p "$RESULT_DIR"
echo "[v23d] run=$RUN_ID gpu=$GPU orders=$ORDERS pool=$POOL avail=$AVAIL steps=$STEPS mb=$DEVICE_BATCH shards=[$TRAIN_SHARDS]"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" \
  --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" \
  --train_shards "$TRAIN_SHARDS" \
  --train_max_tokens "$POOL" \
  --val_shards "${NGLAB_VAL_SHARDS:-2,3,4,5,6,7,8,9,10,6542}" \
  --seed "${NGLAB_SEED:-42}" \
  --steps "$STEPS" \
  --dtype bf16 \
  --injection_position input \
  --enable_unigram 0 \
  --ngram_orders "$ORDERS" \
  --bigram_clean_table 1048576 \
  --trigram_clean_table 1048576 \
  --ngram_clean_table 1048576 \
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
  --table_norm_interval 10 \
  --fixed_train_probe 0 \
  > "$RESULT_DIR/train.log" 2>&1

test -s "$RESULT_DIR/summary.json"
echo "[v23d] complete: $RESULT_DIR"
