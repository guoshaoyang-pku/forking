#!/usr/bin/env bash
set -euo pipefail
ROOT="${NGLAB_ROOT:-/data4/guoshaoyang/ngram-gap-lab}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
BASE="${NGLAB_BASE:-$ROOT/data/runs_scaling/sftbase_ep20_tri_ckpt_fixed/final_model.pt}"
TAG="${NGLAB_OOD_TAG:-magicoder}"
DATA="${NGLAB_OOD_DATA_DIR:-$ROOT/data/ood_tokenized_magicoder}"
OUT="${NGLAB_OOD_OUT:-$ROOT/data/runs_sft_ood}"
WAIT_SECS="${NGLAB_BASE_WAIT_SECS:-172800}"
mkdir -p "$OUT"
start=$(date +%s)
while [ ! -s "$BASE" ] || [ ! -s "$DATA/shard_00000.bin" ] || [ ! -s "$DATA/shard_00001.bin" ]; do
  now=$(date +%s)
  if [ $((now - start)) -ge "$WAIT_SECS" ]; then echo "[sft-ood] timed out waiting for base/data" >&2; exit 1; fi
  sleep 30
done
sha256sum "$BASE" > "$OUT/base_checkpoint.sha256"
run_one() {
  local gpu="$1" arm="$2"; local extra=()
  if [ "$arm" = frozen ]; then extra+=(--freeze_table); fi
  local rid="sft_ood_${TAG}_e5_${arm}_s42"; local run_dir="$OUT/$rid"
  if [ -e "$run_dir" ]; then echo "[sft-ood] refusing to overwrite $run_dir" >&2; return 2; fi
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$rid" --out_dir "$OUT" --data_dir "$DATA" \
    --train_shards 0 --val_shards 1 --seed 42 --steps 1665 --dtype bf16 \
    --injection_position input --enable_unigram 0 --enable_bigram 0 --enable_trigram 1 \
    --bigram_clean_table 0 --trigram_clean_table 1048576 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --total_batch_size 147456 --lr 0.0006 \
    --lr_schedule warmup_constant --warmup_steps 100 \
    --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 128.0 \
    --epoch_batches 333 --val_interval 333 --val_steps 333,666,999,1332,1665 --val_batches 4 \
    --table_norm_interval 333 --init_model "$BASE" --save_final_model \
    "${extra[@]}" > "$run_dir/train.log" 2>&1 &
  echo "[sft-ood] launched $rid on GPU $gpu (pid $!)"
}
run_one "${NGLAB_OOD_GPU_TRAINABLE:-0}" trainable
run_one "${NGLAB_OOD_GPU_FROZEN:-7}" frozen
wait
echo "[sft-ood] both OOD arms completed"
