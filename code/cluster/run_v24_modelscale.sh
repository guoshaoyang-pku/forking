#!/usr/bin/env bash
# v2.4 model-size scaling on H200 — user directive 2026-10-17:
#   "对于模型大小的scaling，vanilla gpt 层数和宽度同时适当增大和缩小，
#    词表大小不动，保持1M，默认只带 trigram。从现在的大小，n=2~5同时存在，
#    1M词表，都扫描一下关于训练数据的scaling，10M~1B取15个点"
# Ladder (layers · width · heads, head_dim 128 fixed, tied embedding, vocab 8192):
#   4L/512d(4H) ~17M · 6L/640d(5H) ~35M · 8L/768d(6H) ~63M (base)
#   · 10L/896d(7H) ~104M · 12L/1024d(8H) ~159M
# ngram modes: tri = trigram only; n25 = orders 2,3,4,5 joint.  R = 1M rows per
# order, unchanged across sizes.  Data envelope = --train_max_tokens pool
# (nested prefix, same protocol as v2.3), fixed 2000 steps = 295M trained
# tokens per arm.  Backbone LR stays 6e-4 at every size (P1: model size is the
# only variable; no muP, no per-size LR retune).  Per-token stats dump at the
# final step is automatic.
#
#   run_v24_modelscale.sh <gpu> <layers> <heads> <embd> <mode:tri|n25> <pool> [steps] [device_batch]
set -euo pipefail

GPU="${1:?gpu id}"
NL="${2:?n_layers}"
NH="${3:?n_head}"
NE="${4:?n_embd}"
MODE="${5:?mode: tri | n25}"
POOL="${6:?pool content tokens, e.g. 10000000}"
STEPS="${7:-2000}"
DEVICE_BATCH="${8:-${NGLAB_DEVICE_BATCH:-72}}"
EPOCHS="${9:-0}"
# Epoch-anchored protocol (2026-10-17 user directive): EPOCHS>0 fixes the number
# of passes over the pool and derives steps = ceil(E * pool / total_batch);
# run_id carries an e{E} tag instead of the step count.
if (( EPOCHS > 0 )); then
  STAG="e${EPOCHS}"
  STEP_ARGS=(--train_epochs "$EPOCHS")
else
  STAG="${STEPS}"
  STEP_ARGS=(--steps "$STEPS")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_orders2to8_s1.npz}"

(( NE % NH == 0 )) || { echo "n_embd $NE not divisible by n_head $NH" >&2; exit 2; }

case "$MODE" in
  tri) ORDERS="3"; MTAG="tri" ;;
  n25) ORDERS="2,3,4,5"; MTAG="n25" ;;
  *) echo "mode must be tri | n25" >&2; exit 2 ;;
esac

ptag() {
  case "$1" in
    10000000) echo 10m ;; 14000000) echo 14m ;; 19000000) echo 19m ;;
    27000000) echo 27m ;; 37000000) echo 37m ;; 49692672) echo 49m ;;
    52000000) echo 52m ;;
    72000000) echo 72m ;; 100000000) echo 100m ;; 139000000) echo 139m ;;
    193000000) echo 193m ;; 268000000) echo 268m ;; 373000000) echo 373m ;;
    519000000) echo 519m ;; 722000000) echo 722m ;; 1000000000) echo 1b ;;
    *) echo "$1" ;;
  esac
}

RUN_ID="v24ms_l${NL}h${NH}d${NE}_${MTAG}_p$(ptag "$POOL")_s${NGLAB_SEED:-42}_${STAG}_fd"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"

# restart hygiene: an incomplete dir from a previous failed launch is moved
# aside so the requeued job can start clean (queue retries the same line).
mkdir -p "$OUT_DIR" "$ROOT/logs/failed_launch"
if [[ -d "$RESULT_DIR" && ! -s "$RESULT_DIR/summary.json" ]]; then
  mv "$RESULT_DIR" "$ROOT/logs/failed_launch/$(basename "$RESULT_DIR").relaunch.$(date +%s)"
fi
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }

if [[ "$MODE" == "n25" ]]; then
  # the data envelope's freq rows need the multi-order index (built once per host)
  WAITED=0
  while [[ ! -f "$FREQ_INDEX" ]]; do
    (( WAITED >= 7200 )) && { echo "freq index never appeared: $FREQ_INDEX" >&2; exit 2; }
    [[ $WAITED -eq 0 ]] && echo "[v24ms] waiting for freq index: $FREQ_INDEX"
    sleep 60; WAITED=$((WAITED + 60))
  done
  FREQ_ARGS=(--freq_index "$FREQ_INDEX" --freq_eval_interval 10 --freq_eval_batches 4
             --exact_freq_eval_interval 10)
else
  # tri mode keeps the trigram freq rows too (index covers orders 2-8)
  FREQ_ARGS=(--freq_index "$FREQ_INDEX" --freq_eval_interval 10 --freq_eval_batches 4
             --exact_freq_eval_interval 10)
fi

# warn (not fail) when the pool exceeds what shard 1 actually holds
AVAIL=$("$PY" - "$DATA_DIR" <<'PYEOF'
import sys, os
d = sys.argv[1]
p = os.path.join(d, "shard_00001.bin")
print(os.path.getsize(p) // 2 // 2048 * 2048 if os.path.exists(p) else 0)
PYEOF
)
if (( POOL > AVAIL )); then
  echo "[v24ms] WARNING: pool $POOL > available $AVAIL tokens from shard 1; using the full stream" >&2
fi

mkdir -p "$RESULT_DIR"
echo "[v24ms] run=$RUN_ID gpu=$GPU L${NL}H${NH}d${NE} mode=$MODE pool=$POOL avail=$AVAIL steps=$STEPS mb=$DEVICE_BATCH"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" \
  --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" \
  --train_shards 1 \
  --train_max_tokens "$POOL" \
  --val_shards "${NGLAB_VAL_SHARDS:-2,3,4,5,6,7,8,9,10,6542}" \
  --seed "${NGLAB_SEED:-42}" \
  "${STEP_ARGS[@]}" \
  --dtype bf16 \
  --injection_position input \
  --enable_unigram 0 \
  --ngram_orders "$ORDERS" \
  --trigram_clean_table 1048576 \
  --ngram_clean_table 1048576 \
  --n_layer "$NL" \
  --n_head "$NH" \
  --n_embd "$NE" \
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
  "${FREQ_ARGS[@]}" \
  > "$RESULT_DIR/train.log" 2>&1

test -s "$RESULT_DIR/summary.json"
echo "[v24ms] complete: $RESULT_DIR"
