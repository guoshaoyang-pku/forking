#!/usr/bin/env bash
# plan-9 S0 engineering pilot (experiment-log §81)
#   G4/X4/S0 = 16R/top-4/d_e768, OE on/off, 100M tokens x 1 pass x seed 42
#   data = manifest_full163 nested prefix (token-exact via --train_max_tokens)
#
# Acceptance (plan-9 section 8): no NaN, assignment conservation, no permanent
# dead experts, sane loss, complete moe_diag/panel fields, tokens/s + peak
# memory in the ledger. NOT for paper conclusions.
#
# Usage:  GPU_ID=0 bash code/cluster/run_p9_s0.sh
set -euo pipefail

GPU_ID=${GPU_ID:-0}
PYTHON=${PYTHON:-python3}
DATA_DIR=${NGLAB_DATA_DIR:-data/manifest_full163}
OUT_DIR=${NGLAB_OUT_DIR:-data/runs_fixed}
CODE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$CODE_DIR"

echo "== md5 (must match committed versions) =="
md5 code/train.py code/moe.py code/ngram_freq.py 2>/dev/null || \
  md5sum code/train.py code/moe.py code/ngram_freq.py

echo "== GPU check =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "Using GPU $GPU_ID"

TRAIN_SHARDS="$(seq -s, 0 157)"

COMMON_ARGS=(
  --injection_position input
  --lr 0.0006
  --lr_schedule warmup_constant
  --warmup_steps 100
  --seed 42
  --device_batch_size 72
  --total_batch_size 147456
  --data_dir "$DATA_DIR"
  --train_shards "$TRAIN_SHARDS"
  --val_shards 6542
  --train_max_tokens 100000000
  --train_epochs 1
  --val_interval 10
  --dtype bf16
  --out_dir "$OUT_DIR"
  --moe_g 4
  --moe_x 4
)

echo "== arm S0-OEON =="
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON code/train.py \
  --run_id p9_moe_g4_x4_s0_e16_k4_de768_oeon_u100m_r1_seed42 \
  --enable_bigram 1 --enable_trigram 1 \
  --bigram_clean_table 1048576 --trigram_clean_table 1048576 \
  --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 128.0 \
  "${COMMON_ARGS[@]}"

echo "== arm S0-OEOFF =="
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON code/train.py \
  --run_id p9_moe_g4_x4_s0_e16_k4_de768_oeoff_u100m_r1_seed42 \
  --enable_bigram 0 --enable_trigram 0 \
  --bigram_clean_table 0 --trigram_clean_table 0 \
  "${COMMON_ARGS[@]}"

echo "== S0 summary =="
$PYTHON - "$OUT_DIR" <<'EOF'
import json, sys, math
out = sys.argv[1]
ids = ["p9_moe_g4_x4_s0_e16_k4_de768_oeon_u100m_r1_seed42",
       "p9_moe_g4_x4_s0_e16_k4_de768_oeoff_u100m_r1_seed42"]
for rid in ids:
    try:
        s = json.load(open(f"{out}/{rid}/summary.json"))
        moe = s["moe"]
        # dead experts / load stats from the last moe_diag line
        last = None
        with open(f"{out}/{rid}/moe_diag.jsonl") as f:
            for line in f:
                last = json.loads(line)
        deads = [last[f"l{i}"]["dead_experts"] for i in range(8)]
        loads = [last[f"l{i}"]["load_entropy_norm"] for i in range(8)]
        nan_free = all(math.isfinite(x) for x in
                       [s["final_train_loss"], s["final_val_loss"]])
        print(f"{rid.split('_de768_')[1]}:")
        print(f"  train {s['final_train_loss']:.4f} val {s['final_val_loss']:.4f} "
              f"gap {s['final_gap']:+.4f} nan_free={nan_free}")
        print(f"  aux {moe['final_aux_raw']:.4f} z {moe['final_z_raw']:.4f} "
              f"tokens/s {moe['tokens_per_s']:.0f} peakGB {moe['peak_mem_gb']:.1f}")
        print(f"  dead experts per layer (last block) {deads}")
        print(f"  load entropy norm per layer {[round(x,3) for x in loads]}")
    except FileNotFoundError as e:
        print(f"{rid}: MISSING ({e})")
EOF
echo "DONE. Acceptance per plan-9 section 8; do NOT start the main 2x2 without user sign-off."
