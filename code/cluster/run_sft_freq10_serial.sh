#!/usr/bin/env bash
# Re-run the §57 continued-pretraining stress test with dense v10 loss logs.
# Six arms are deliberately serial on one GPU; no existing run is overwritten.
set -euo pipefail
ROOT="\${NGLAB_ROOT:-/data4/guoshaoyang/ngram-gap-lab}"
PY="\${NGLAB_PY:-\$ROOT/.venv/bin/python}"
GPU="\${NGLAB_GPU:-1}"
BASE="\${NGLAB_BASE:-\$ROOT/data/runs_scaling/sftbase_ep20_tri_ckpt_fixed/final_model.pt}"
ID_DATA="\${NGLAB_ID_DATA:-\$ROOT/data/tokenized_epoch20}"
MAGICODER_DATA="\${NGLAB_MAGICODER_DATA:-\$ROOT/data/ood_tokenized_magicoder}"
OPENR1_DATA="\${NGLAB_OPENR1_DATA:-\$ROOT/data/ood_tokenized_openr1_10shards}"
ID_OUT="\${NGLAB_ID_OUT:-\$ROOT/data/runs_sft_freq10}"
MAGICODER_OUT="\${NGLAB_MAGICODER_OUT:-\$ROOT/data/runs_sft_ood_freq10}"
OPENR1_OUT="\${NGLAB_OPENR1_OUT:-\$ROOT/data/runs_sft_ood_openr1_freq10}"
for path in "\$BASE" "\$ID_DATA" "\$MAGICODER_DATA" "\$OPENR1_DATA"; do
  test -e "\$path" || { echo "[freq10] missing required path: \$path" >&2; exit 1; }
done
mkdir -p "\$ID_OUT" "\$MAGICODER_OUT" "\$OPENR1_OUT"
sha256sum "\$BASE" > "\$ID_OUT/base_checkpoint.sha256"
sha256sum "\$BASE" > "\$MAGICODER_OUT/base_checkpoint.sha256"
sha256sum "\$BASE" > "\$OPENR1_OUT/base_checkpoint.sha256"
sha256sum "\$ROOT/code/train.py" "\$ROOT/code/ngram_freq.py" | tee "\$ID_OUT/code_hashes.sha256" > "\$MAGICODER_OUT/code_hashes.sha256"
cp "\$ID_OUT/code_hashes.sha256" "\$OPENR1_OUT/code_hashes.sha256"
run_one() {
  local out="\$1" rid="\$2" data="\$3" train_shards="\$4" val_shards="\$5" epoch_batches="\$6" freeze="\$7"
  local run_dir="\$out/\$rid"
  if [ -e "\$run_dir" ]; then echo "[freq10] refusing to overwrite \$run_dir" >&2; exit 2; fi
  mkdir -p "\$run_dir"
  local extra=()
  if [ "\$freeze" = 1 ]; then extra+=(--freeze_table); fi
  echo "[freq10] start \$rid on GPU \$GPU \$(date -Is)"
  CUDA_VISIBLE_DEVICES="\$GPU" "\$PY" -u "\$ROOT/code/train.py" \\
    --run_id "\$rid" --out_dir "\$out" --data_dir "\$data" \\
    --train_shards "\$train_shards" --val_shards "\$val_shards" \\
    --seed 42 --steps 1665 --dtype bf16 \\
    --injection_position input --enable_unigram 0 --enable_bigram 0 --enable_trigram 1 \\
    --bigram_clean_table 0 --trigram_clean_table 1048576 \\
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \\
    --device_batch_size 72 --total_batch_size 147456 \\
    --lr 0.0006 --lr_schedule warmup_constant --warmup_steps 100 \\
    --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 128.0 \\
    --epoch_batches "\$epoch_batches" --val_interval 10 --freq_eval_interval 10 \\
    --exact_freq_eval_interval 10 --val_batches 4 --table_norm_interval 10 \\
    --init_model "\$BASE" --save_final_model "\${extra[@]}" > "\$run_dir/train.log" 2>&1
  test -s "\$run_dir/summary.json" || { echo "[freq10] missing summary for \$rid" >&2; exit 3; }
  test -s "\$run_dir/train_log.jsonl" || { echo "[freq10] missing train log for \$rid" >&2; exit 3; }
  test -s "\$run_dir/final_model.pt" || { echo "[freq10] missing final model for \$rid" >&2; exit 3; }
  echo "[freq10] done \$rid \$(date -Is)"
}
run_one "\$ID_OUT" "sft_id_tri_e5_freq10_trainable_s42" "\$ID_DATA" 21 22,23 333 0
run_one "\$ID_OUT" "sft_id_tri_e5_freq10_frozen_s42" "\$ID_DATA" 21 22,23 333 1
run_one "\$MAGICODER_OUT" "sft_ood_magicoder_e5_freq10_trainable_s42" "\$MAGICODER_DATA" 0 1 333 0
run_one "\$MAGICODER_OUT" "sft_ood_magicoder_e5_freq10_frozen_s42" "\$MAGICODER_DATA" 0 1 333 1
run_one "\$OPENR1_OUT" "sft_ood_openr1_10shards_e5_freq10_trainable_s42" "\$OPENR1_DATA" 0 1 333 0
run_one "\$OPENR1_OUT" "sft_ood_openr1_10shards_e5_freq10_frozen_s42" "\$OPENR1_DATA" 0 1 333 1
echo "[freq10] all six runs completed \$(date -Is)"
