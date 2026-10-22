#!/usr/bin/env bash
# v27 launcher (2026-10-22): same model ladder / arms / optimizer as run_v26_mh.sh, but
#  * data = clean pool data/tokenized_v27 (code/tools/v27_clean_pool.py): val is the
#    288 evaluated chunks of shard 2 (+224 spare chunks); every train chunk sharing any
#    32-token span with them is removed, shards holding val copies are dropped. New
#    shards must pass the same filter before they enter manifest.json (10B plan).
#  * D = NGLAB_DN x N_nominal unique tokens (any D/N, default 20). The shard list is the
#    shortest prefix of manifest.train_order covering D (nested prefixes: all D/N
#    points share one data ordering).
#  * NGLAB_DB / NGLAB_VDB override device / val micro-batch (A100-40GB). The optimizer
#    batch (147,456 tok) and the evaluated val chunks (288) are unchanged.
#
#   run_v27.sh <gpu> <model> <arm:nogram|exact|mh> [seed]
# env: NGLAB_PY NGLAB_ROOT NGLAB_DN NGLAB_RK NGLAB_TLS NGLAB_EPOCHS NGLAB_DB NGLAB_VDB
#      NGLAB_SUBSET NGLAB_SUBSET_PASSES NGLAB_HEADS NGLAB_KK NGLAB_TAG
set -euo pipefail
GPU="${1:?gpu}"; MODEL="${2:?model}"; ARM="${3:?arm}"; SEED="${4:-42}"
ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
DATA_DIR="${NGLAB_DATA_DIR_V27:-$ROOT/data/tokenized_v27}"
OUT_DIR="$ROOT/data/runs_fixed"
FREQ_INDEX="${NGLAB_FREQ_INDEX:-$ROOT/data/freq_index_o2to5_1b_f30.npz}"
ASSIGN_DIR="${NGLAB_ASSIGN_DIR:-$ROOT/data/assignments}"

E="${NGLAB_EPOCHS:-3}"
case "$MODEL" in
  17m)  NL=4;  NE=512;  NH=4; RK=136;  NN=17000000;  TLS=1;   DB=72 ;;
  35m)  NL=6;  NE=640;  NH=5; RK=280;  NN=35000000;  TLS=0.5; DB=72 ;;
  50m)  NL=8;  NE=768;  NH=6; RK=400;  NN=50000000;  TLS=1;   DB=72 ;;
  104m) NL=10; NE=896;  NH=7; RK=832;  NN=104000000; TLS=1;   DB=72 ;;
  159m) NL=12; NE=1024; NH=8; RK=1272; NN=159000000; TLS=1;   DB=36 ;;
  200m) NL=12; NE=1152; NH=9; RK=1600; NN=200000000; TLS=1;   DB=12 ;;
  *) echo "model: 17m|35m|50m|104m|159m|200m" >&2; exit 2 ;;
esac
TLS="${NGLAB_TLS:-$TLS}"
RK="${NGLAB_RK:-$RK}"
DB="${NGLAB_DB:-$DB}"
VDB="${NGLAB_VDB:-72}"
(( 288 % VDB == 0 )) || { echo "NGLAB_VDB must divide 288" >&2; exit 2; }
VB=$(( 288 / VDB ))

DN="${NGLAB_DN:-20}"
DTOK=$("$PY" -c "print(int(round(float('$DN') * $NN)))")
DTAG="dn${DN}"
MAN="$DATA_DIR/manifest.json"
[[ -f "$MAN" ]] || { echo "missing $MAN (build with code/tools/v27_clean_pool.py)" >&2; exit 2; }
SHARDS=$("$PY" "$ROOT/code/tools/v27_prefix.py" "$MAN" "$DTOK")

case "$ARM" in
  nogram)
    NG=(--enable_bigram 0 --enable_trigram 0 --enable_unigram 0) ;;
  exact)
    ASSIGN="$ASSIGN_DIR/assign_exact_r${RK}k.npz"
    [[ -f "$ASSIGN" ]] || { echo "assign npz missing: $ASSIGN (run dn20_make_exact_r.py)" >&2; exit 2; }
    CT=524288
    (( RK > 400 )) && CT=1048576
    (( RK > 832 )) && CT=2097152
    NG=(--enable_unigram 0 --ngram_orders 2,3,4,5 --ngram_clean_table "$CT"
        --ngram_assign_file "$ASSIGN" --freq_index "$FREQ_INDEX") ;;
  mh)
    H="${NGLAB_HEADS:?NGLAB_HEADS}"; KK="${NGLAB_KK:?NGLAB_KK (k contexts/order)}"
    ASSIGN="$ASSIGN_DIR/assign_mh${H}_k${KK}k_r${RK}k.npz"
    [[ -f "$ASSIGN" ]] || { echo "assign npz missing: $ASSIGN (run code/tools/mh_make_assign.py)" >&2; exit 2; }
    CT=524288
    (( RK > 400 )) && CT=1048576
    (( RK > 832 )) && CT=2097152
    NG=(--enable_unigram 0 --ngram_orders 2,3,4,5 --ngram_clean_table "$CT"
        --ngram_assign_file "$ASSIGN" --ngram_heads "$H" --freq_index "$FREQ_INDEX")
    ARM="mh${H}k${KK}k" ;;
  *) echo "arm: nogram|exact|mh" >&2; exit 2 ;;
esac

SUB=()
TAG="${NGLAB_TAG:-}"
if [[ -n "${NGLAB_SUBSET:-}" ]]; then
  E=1
  SP="${NGLAB_SUBSET_PASSES:-3}"
  SUB=(--replay_subset_frac "$NGLAB_SUBSET" --replay_subset_passes "$SP"
       --fixed_train_probe 4 --train_probe_mode subset --probe_eval_interval 50)
  TAG="_sub${NGLAB_SUBSET}x${SP}${TAG}"
fi
TABLE=(--table_optimizer adamw --table_betas 0.9,0.95 --table_grad_clip 1.0
       --table_lr_scale "$TLS")
RUN_ID="v27_${MODEL}_${ARM}_r${RK}k_${DTAG}_tls${TLS}_s${SEED}_e${E}${TAG}"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }
mkdir -p "$RESULT_DIR"

echo "[v27] run=$RUN_ID gpu=$GPU model=$MODEL arm=$ARM seed=$SEED D=$DTOK shards=$(echo "$SHARDS" | tr , '\n' | wc -l) R/order=${RK}k tls=$TLS db=$DB vdb=$VDB"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" --out_dir "$OUT_DIR" --data_dir "$DATA_DIR" \
  --train_shards "$SHARDS" --train_max_tokens "$DTOK" --train_epochs "$E" \
  --val_shards 2 --seed "$SEED" \
  --dtype bf16 --injection_position input "${NG[@]}" \
  --n_layer "$NL" --n_head "$NH" --n_embd "$NE" --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size "$DB" --total_batch_size 147456 --val_device_batch_size "$VDB" \
  --lr 0.0006 --lr_schedule warmup_constant --warmup_steps 100 \
  "${TABLE[@]}" \
  --val_interval 10 --val_batches "$VB" --table_norm_interval 50 --fixed_train_probe 0 \
  --freq_eval_interval 1000000 --exact_freq_eval_interval 1000000 --diag_interval 1000000 \
  "${SUB[@]}" \
  > "$RESULT_DIR/train.log" 2>&1
test -s "$RESULT_DIR/summary.json"
echo "[v27] complete: $RESULT_DIR"
