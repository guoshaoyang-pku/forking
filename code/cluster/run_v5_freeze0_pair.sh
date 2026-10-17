#!/usr/bin/env bash
# experiment-log §79 · freeze0 pair: does a frozen random table still fork?
#
# Arms (v5 standard setting, ONLY variable = --freeze_table):
#   frz0_ctrl_s42    standard input injection, learnable table (expect ~1.867)
#   frz0_table_s42   + --freeze_table (tables stay at N(0,0.02) init, never updated)
#
# Competing predictions (§79): H-user partition-fork vs H-sharpen no-fork.
#
# Usage:
#   GPU_ID=0 bash code/cluster/run_v5_freeze0_pair.sh
set -euo pipefail

GPU_ID=${GPU_ID:-0}
STEPS=${STEPS:-1000}
DATA_DIR=${NGLAB_DATA_DIR:-data/tokenized}
OUT_DIR=${NGLAB_OUT_DIR:-data/runs_fixed}
CODE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$CODE_DIR"

echo "== md5 (must match the committed versions before launch) =="
md5 code/train.py code/ngram_freq.py 2>/dev/null || md5sum code/train.py code/ngram_freq.py

echo "== GPU check =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "Using GPU $GPU_ID (assign one card per agent; do not share)."

COMMON_ARGS=(
  --injection_position input
  --bigram_clean_table 1048576
  --trigram_clean_table 1048576
  --table_optimizer rmsprop
  --table_betas 0.0,0.99
  --table_lr_scale 128.0
  --lr 0.0006
  --lr_schedule warmup_constant
  --warmup_steps 100
  --seed 42
  --steps "$STEPS"
  --device_batch_size 72
  --total_batch_size 147456
  --val_interval 10
  --data_dir "$DATA_DIR"
  --train_shards 1
  --val_shards 2,3,4,5,6,7,8,9,10,6542
  --dtype bf16
  --out_dir "$OUT_DIR"
)

echo "== arm F0: control (learnable table) =="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 code/train.py \
  --run_id frz0_ctrl_s42 "${COMMON_ARGS[@]}"

echo "== arm F1: frozen table from step 0 =="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 code/train.py \
  --run_id frz0_table_s42 "${COMMON_ARGS[@]}" --freeze_table

echo "== pair summary =="
python3 - "$OUT_DIR" <<'EOF'
import json, sys
out_dir = sys.argv[1]
for arm in ["frz0_ctrl_s42", "frz0_table_s42"]:
    try:
        summary = json.load(open(f"{out_dir}/{arm}/summary.json"))
        print(f"{arm}: final_gap {summary['final_gap']:+.4f}  "
              f"train {summary['final_train_loss']:.4f}  val {summary['final_val_loss']:.4f}")
    except FileNotFoundError:
        print(f"{arm}: MISSING")
EOF
echo "DONE. Compare with experiment-log §79 predictions."
