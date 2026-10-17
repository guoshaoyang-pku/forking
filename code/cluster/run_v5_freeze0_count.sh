#!/usr/bin/env bash
# experiment-log §80 · count-init frozen arm: real-signal-but-unlearnable table.
# Golf-equivalent replica: rows = count-weighted mean of init-wte rows of each
# hash bucket (pure train-text frequency statistics), then frozen via
# --freeze_table. Single variable vs frz0_table_s42 (§79): table content
# (random -> real statistics).
#
# Usage:
#   GPU_ID=0 bash code/cluster/run_v5_freeze0_count.sh
set -euo pipefail

GPU_ID=${GPU_ID:-0}
STEPS=${STEPS:-1000}
DATA_DIR=${NGLAB_DATA_DIR:-data/tokenized}
OUT_DIR=${NGLAB_OUT_DIR:-data/runs_fixed}
FREQ_NPZ=${NGLAB_FREQ_NPZ:-data/freq_index.npz}
CODE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$CODE_DIR"

echo "== md5 (must match the committed versions before launch) =="
md5 code/train.py code/ngram_freq.py 2>/dev/null || md5sum code/train.py code/ngram_freq.py

echo "== freq index =="
ls -la "$FREQ_NPZ"

echo "== GPU check =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "Using GPU $GPU_ID (assign one card per agent; do not share)."

echo "== arm FC: count-init frozen table =="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 code/train.py \
  --run_id frz0_count_s42 \
  --injection_position input \
  --bigram_clean_table 1048576 \
  --trigram_clean_table 1048576 \
  --table_optimizer rmsprop \
  --table_betas 0.0,0.99 \
  --table_lr_scale 128.0 \
  --lr 0.0006 \
  --lr_schedule warmup_constant \
  --warmup_steps 100 \
  --seed 42 \
  --steps "$STEPS" \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --val_interval 10 \
  --data_dir "$DATA_DIR" \
  --train_shards 1 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --dtype bf16 \
  --out_dir "$OUT_DIR" \
  --freq_index "$FREQ_NPZ" \
  --table_init_count "$FREQ_NPZ" \
  --freeze_table

echo "== summary =="
python3 - "$OUT_DIR" <<'EOF'
import json, sys
out_dir = sys.argv[1]
try:
    summary = json.load(open(f"{out_dir}/frz0_count_s42/summary.json"))
    print(f"frz0_count_s42: final_gap {summary['final_gap']:+.4f}  "
          f"train {summary['final_train_loss']:.4f}  val {summary['final_val_loss']:.4f}")
except FileNotFoundError:
    print("frz0_count_s42: MISSING")
EOF
echo "DONE. Compare with §79 (frz0_table 0.064 / frz0_ctrl 2.702) and §80 predictions."
