#!/usr/bin/env bash
# plan-8 G6 · fast-path bench: 4 arms on ONE H200, v5 standard setting.
#
# Arms:
#   A fastpath_eager_base_s42      eager bf16 (control)
#   B fastpath_eager_pc_s42        + offline row index (pure-gather forward)
#   C fastpath_compile_pc_s42      + torch.compile
#   D fastpath_compile_pc_fp8_s42  + torchao fp8 backbone (needs torchao)
#
# Acceptance (plan-8 §3): C >= 1.5x A throughput (expectation, measure first);
# B final gap == A final gap within 1e-6 (bit-identical fast path);
# D must pass the separate bf16 pairing protocol before entering mainline.
#
# Usage:
#   GPU_ID=0 bash code/cluster/run_v5_fastpath_bench.sh
#   STEPS=300 (default)  # bench only; not a scientific run
set -euo pipefail

GPU_ID=${GPU_ID:-0}
STEPS=${STEPS:-300}
DATA_DIR=${NGLAB_DATA_DIR:-data/tokenized}
OUT_DIR=${NGLAB_OUT_DIR:-data/runs_fixed}
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_DIR"

echo "== md5 (must match the committed versions before launch) =="
md5 train.py ngram_freq.py precompute_row_index.py 2>/dev/null || md5sum train.py ngram_freq.py precompute_row_index.py

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

echo "== offline row index for shard 1 (skipped if present) =="
if [ ! -f "$DATA_DIR/shard_00001.brows.u32.bin" ]; then
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 precompute_row_index.py \
    --data_dir "$DATA_DIR" --shards 1 \
    --vocab_size 8192 \
    --bigram_clean_table 1048576 \
    --trigram_clean_table 1048576 \
    --sequence_len 2048 --token_dtype u16
else
  echo "rows already present: $DATA_DIR/shard_00001.{b,t}rows.u32.bin"
fi

echo "== arm A: eager base =="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
  --run_id fastpath_eager_base_s42 "${COMMON_ARGS[@]}"

echo "== arm B: eager + precomputed rows =="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
  --run_id fastpath_eager_pc_s42 "${COMMON_ARGS[@]}" --row_index

echo "== arm C: compile + precomputed rows =="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
  --run_id fastpath_compile_pc_s42 "${COMMON_ARGS[@]}" --row_index --compile

echo "== arm D: compile + precomputed rows + fp8 backbone =="
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
  --run_id fastpath_compile_pc_fp8_s42 "${COMMON_ARGS[@]}" --row_index --compile --fp8_torchao

echo "== bench summary =="
python3 - "$OUT_DIR" "$STEPS" <<'EOF'
import json, sys
out_dir, steps = sys.argv[1], int(sys.argv[2])
arms = ["fastpath_eager_base_s42", "fastpath_eager_pc_s42",
        "fastpath_compile_pc_s42", "fastpath_compile_pc_fp8_s42"]
rows = []
for arm in arms:
    try:
        log = [json.loads(l) for l in open(f"{out_dir}/{arm}/train_log.jsonl")]
        summary = json.load(open(f"{out_dir}/{arm}/summary.json"))
    except FileNotFoundError:
        print(f"{arm}: MISSING"); continue
    t = log[-1]["elapsed_s"]
    rows.append((arm, steps / t, summary["final_gap"], t))
base = next((r for r in rows if r[0] == arms[0]), None)
for arm, sps, gap, t in rows:
    speedup = f"{base[1] / sps:.2f}x" if base else "-"
    print(f"{arm:34s} {sps:8.1f} steps/s  ({speedup})  final_gap {gap:+.4f}  wall {t:.0f}s")
EOF
echo "NOTE: compile arms include one-time compile warmup in wall time."
echo "DONE. Compare with plan-8 §3 acceptance."
