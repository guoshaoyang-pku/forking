#!/usr/bin/env bash
# plan-9 FLAME family (Step 2, user 2026-10-19): 2x2 {Dense, Dense+OE} x
# {FLAME-MoE, FLAME-MoE+OE} at FLAME Small (12L/512D/16H, layer0 dense
# SwiGLU 2736, MoE layers 64R/top-6 + 2 shared units of d_e=352), standard
# residual. Same 1B ClimbMix pool x E=3 protocol as v25 (shards 1,11-23,25-32;
# val 2-10+6542; batch 147456 tok; lr 6e-4 warmup_constant(100); bf16; seed
# 42 default). Dense = P2 exact-active twin (--flame_dense_twin, width 8*d_e).
# OE = n=2..5 joint, k256k per-order selection (262144/262144/65536/16384),
# table RMSProp(0,0.99) @1x -- the Step-1 recipe (k256k@1x: best stable E3 net,
# selection arms never diverged at <=4x across 12 arm-seeds, §89).
#
#   run_v26_flame_family.sh <gpu> <arm:dense|dense_oe|moe|moe_oe> [mode:E3|<steps>] [device_batch]
# e.g. run_v26_flame_family.sh 0 moe E3            # full 1B x 3 epochs
#      run_v26_flame_family.sh 0 moe 60            # 60-step smoke (same pool/order)
#      run_v26_flame_family.sh 3 moe_oe E3 24      # microbatch 24 (accum 3), val pinned to 72
set -euo pipefail
GPU="${1:?gpu}"; ARM="${2:?arm}"; MODE="${3:-E3}"; DBATCH="${4:-72}"
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
SEED="${NGLAB_SEED:-42}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_o2to5_1b.npz}"
SHARDS="$(cat "$ROOT/data/train_shards_1b.txt")"

case "$ARM" in
  dense|moe)     OE=0 ;;
  dense_oe|moe_oe) OE=1 ;;
  *) echo "arm: dense|dense_oe|moe|moe_oe" >&2; exit 2 ;;
esac
FLAME=(--flame_profile small)
case "$ARM" in
  dense|dense_oe) FLAME+=(--flame_dense_twin) ;;
esac

if [[ "$OE" == 1 ]]; then
  if [[ -n "${NGLAB_OE_ASSIGN:-}" ]]; then
    # assign-driven OE (e.g. exact974k zero-collision / w2resid routing):
    # exact-R table sizing via train.py cfg._assign_R0; heads=1.
    NG=(--enable_unigram 0 --ngram_orders 2,3,4,5 --ngram_clean_table 1048576
        --ngram_assign_file "$NGLAB_OE_ASSIGN" --freq_index "$FREQ_INDEX")
    RUN_ID_SUFFIX="${NGLAB_OE_SUFFIX:-assign}"
  else
    NG=(--enable_unigram 0 --ngram_orders 2,3,4,5 --ngram_clean_table 1048576
        --ngram_dict_topk 1048576 --freq_index "$FREQ_INDEX"
        --ngram_dict_per_order "2:k=262144,3:k=262144,4:k=65536,5:k=16384")
    RUN_ID_SUFFIX="adamc"
  fi
  # EOS-divergence fix (user 2026-10-19, divergence-problem-brief §8): table
  # optimizer = Adam(wd=0, betas 0.9/0.95, eps 1e-8) + global grad clip 1.0,
  # replacing hand-written RMSProp(0,0.99) whose dense-grad v-decay amplifies
  # sparse-row hit displacement up to 10x lr_t.
  TABLE=(--table_optimizer adamw --table_betas 0.9,0.95 --table_grad_clip 1.0
         --table_lr_scale 1.0)
else
  NG=(--enable_bigram 0 --enable_trigram 0 --enable_unigram 0)
  TABLE=()
fi

if [[ "$MODE" == "E3" ]]; then
  SCHED=(--train_max_tokens 1000000000 --train_epochs 3)
  RUN_ID="v26fl_${ARM}_p1b_s${SEED}_e3_fd"
else
  SCHED=(--train_max_tokens 1000000000 --steps "$MODE")
  RUN_ID="v26fl_${ARM}_p1b_s${SEED}_smoke${MODE}_fd"
fi
[[ "$OE" == 1 ]] && RUN_ID="${RUN_ID/e3_fd/e3${RUN_ID_SUFFIX}_fd}"
[[ "$DBATCH" == 72 ]] || RUN_ID="${RUN_ID}_db${DBATCH}"
BATCH=(--device_batch_size "$DBATCH" --total_batch_size 147456
       --val_device_batch_size 72)
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"

echo "[v26fl] run=$RUN_ID gpu=$GPU arm=$ARM mode=$MODE seed=$SEED"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" --out_dir "$OUT_DIR" --data_dir "$DATA_DIR" \
  --train_shards "$SHARDS" "${SCHED[@]}" \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 --seed "$SEED" \
  --dtype bf16 --injection_position input "${NG[@]}" \
  --vocab_size 8192 --sequence_len 2048 \
  "${BATCH[@]}" \
  --lr 0.0006 --lr_schedule warmup_constant --warmup_steps 100 \
  "${FLAME[@]}" "${TABLE[@]}" \
  --val_interval 10 --val_batches 4 --table_norm_interval 50 --fixed_train_probe 0 \
  --freq_eval_interval 1000000 --exact_freq_eval_interval 1000000 --diag_interval 1000000 \
  > "$RESULT_DIR/train.log" 2>&1
test -s "$RESULT_DIR/summary.json"
echo "[v26fl] complete: $RESULT_DIR"
