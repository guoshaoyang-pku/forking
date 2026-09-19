#!/usr/bin/env bash
# 20x corpus, three-pass matched multi-layer memory checkpoint runs.
# Layer ids are zero-based: 1,4 = 1-based blocks 2 and 5.
set -euo pipefail
POSITION="\${1:?usage: run_20x_multilayer_memory.sh v|y|engram gpu [run_suffix]}"
GPU="\${2:?usage: run_20x_multilayer_memory.sh v|y|engram gpu [run_suffix]}"
SUFFIX="\${3:-ckpt}"
case "$POSITION" in v|y|engram) ;; *) echo "position must be v, y, or engram" >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
ROOT="\${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="\${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA="\${NGLAB_DATA_DIR:-$ROOT/data/tokenized_epoch20}"
OUT="\${NGLAB_OUT_DIR:-$ROOT/data/runs_scaling}"
RID="s1v5_128_ep20_\${POSITION}_L14_20p0xL4_3ep_\${SUFFIX}"
RESULT="$OUT/\${RID}_fixed"
[[ -x "$PY" ]] || { echo "python unavailable: $PY" >&2; exit 2; }
[[ -d "$DATA" ]] || { echo "data unavailable: $DATA" >&2; exit 2; }
for sid in $(seq 0 23); do
  shard="$DATA/shard_$(printf '%05d' "$sid").bin"
  [[ -s "$shard" ]] || { echo "missing shard $sid" >&2; exit 2; }
done
[[ ! -e "$RESULT" ]] || { echo "refusing existing run: $RESULT" >&2; exit 2; }
mkdir -p "$RESULT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "\${RID}_fixed" --out_dir "$OUT" \
  --data_dir "$DATA" --train_shards 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --val_shards 21,22,23 --seed 42 --steps 20220 --dtype bf16 \
  --injection_position "$POSITION" --inject_layers 1,4 \
  --enable_unigram 0 --enable_bigram 1 --enable_trigram 1 \
  --bigram_clean_table 1048576 --trigram_clean_table 1048576 \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size 72 --total_batch_size 147456 --lr 0.0006 \
  --lr_schedule warmup_constant --warmup_steps 100 \
  --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 128.0 \
  --epoch_batches 6740 --val_interval 10 --val_steps 6740,13480,20220 \
  --val_batches 4 --table_norm_interval 10 --save_final_model --save_epoch_checkpoints \
  > "$RESULT/train.log" 2>&1
test -s "$RESULT/summary.json"
test -s "$RESULT/final_model.pt"
test -s "$RESULT/epoch1_model.pt"
test -s "$RESULT/epoch2_model.pt"
echo "[20x-$POSITION-L14] complete: $RESULT"
