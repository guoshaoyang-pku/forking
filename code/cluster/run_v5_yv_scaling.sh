#!/usr/bin/env bash
# y/v scaling wave (2026-09-06): same three axes as run_v5_s1_three_axis.sh
# (input arm), now for injection_position y and v.
#
# Groups (S1_GROUP):
#   tbl_y / tbl_v  : bigram single-table R sweep (31 R points, input grid reused)
#   ep_y  / ep_v   : bigram single-table epoch-length sweep (12 x 3ep + 1xL4_10ep)
#   ml             : multi-layer main curves, both tables R=2^20, 2000 steps
#                    (y/v x {1,7}/{1,3,5,7}; single-layer {1} = existing inj_fd arms)
#
# Usage: S1_GROUP=tbl_y NGLAB_PY=python3 bash run_v5_yv_scaling.sh <gpu> [gpu...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
VAL_SHARDS="2,3,4,5,6,7,8,9,10,6542"
GROUP="${S1_GROUP:?set S1_GROUP to tbl_y, tbl_v, ep_y, ep_v, or ml}"
GPUS=("$@")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "usage: S1_GROUP=<group> NGLAB_PY=python3 $0 <gpu> [gpu...]" >&2
  exit 2
fi

TABLE_ROWS=(16000 22000 30000 41000 56000 76000 104000 142000 194000 265000 362000 494000 675000 922000 1259000 1719000 2000000 2347000)
SMALL_TABLE_ROWS=(10000 4642 2154 1000 464 215 100 46 22 10 5 2 1)
SPECS=()

case "$GROUP" in
  tbl_y|tbl_v)
    POS="${GROUP#tbl_}"
    for rows in "${TABLE_ROWS[@]}" "${SMALL_TABLE_ROWS[@]}"; do
      SPECS+=("s1v5_128_${POS}_tbl_bi1_R${rows}|1000|--injection_position ${POS} --enable_bigram 1 --enable_trigram 0 --bigram_clean_table ${rows} --trigram_clean_table 0 --val_steps 337,674,1000")
    done
    ;;
  ep_y|ep_v)
    POS="${GROUP#ep_}"
    EPOCH_MULTS=(0p125 0p1667 0p25 0p3333 0p5 0p6667 0p75 1p0 1p25 1p5 1p75 2p0)
    EPOCH_BATCHES=(42 56 84 112 168 224 253 337 421 506 590 674)
    for i in "${!EPOCH_BATCHES[@]}"; do
      batches="${EPOCH_BATCHES[$i]}"
      multiplier="${EPOCH_MULTS[$i]}"
      steps=$((batches * 3))
      SPECS+=("s1v5_128_${POS}_ep_bi_${multiplier}xL4_3ep|${steps}|--injection_position ${POS} --epoch_batches ${batches} --enable_bigram 1 --enable_trigram 0 --bigram_clean_table 1048576 --trigram_clean_table 0 --val_steps ${batches},$((batches * 2)),${steps}")
    done
    SPECS+=(
      "s1v5_128_${POS}_ep_bi_1xL4_10ep|3370|--injection_position ${POS} --epoch_batches 337 --enable_bigram 1 --enable_trigram 0 --bigram_clean_table 1048576 --trigram_clean_table 0 --val_steps 337,674,1011,1348,1685,2022,2359,2696,3033,3370"
    )
    ;;
  ml)
    for POS in y v; do
      SPECS+=("mlv5_${POS}_L17_fd|2000|--injection_position ${POS} --inject_layers 1,7")
      SPECS+=("mlv5_${POS}_L1357_fd|2000|--injection_position ${POS} --inject_layers 1,3,5,7")
    done
    ;;
  epfix)
    # Corrected >1x epoch arms: real multi-shard train sets. The original ep
    # group could not exceed the 1x shard (nested-prefix), so 1.25x-2x silently
    # cycled the same 1x shard (extra passes, not longer epochs).
    # Spec: run_id|steps|train_shards|freq_index|epoch_batches
    # Batches are exact shard-union floors: 1+62=421, 1+61=505, 1+60+62=589, 1+2=670.
    declare -A EPF_SHARDS=( [1p25x]="1,62" [1p5x]="1,61" [1p75x]="1,60,62" [2x]="1,2" )
    declare -A EPF_IDX=( [1p25x]="freq_index_train1_25x" [1p5x]="freq_index_train1_5x" \
                         [1p75x]="freq_index_train1_75x" [2x]="freq_index_train2x_fine" )
    declare -A EPF_BATCHES=( [1p25x]=421 [1p5x]=505 [1p75x]=589 [2x]=670 )
    for POS in y v input; do
      for m in 1p25x 1p5x 1p75x 2x; do
        b="${EPF_BATCHES[$m]}"
        pos_flag=""
        if [[ "$POS" != "input" ]]; then
          pos_flag="--injection_position ${POS} --enable_bigram 1 --enable_trigram 0 --bigram_clean_table 1048576 --trigram_clean_table 0"
        fi
        SPECS+=("s1v5_128_${POS}_epf_${m}xL4_3ep|$((b * 3))|${EPF_SHARDS[$m]}|${EPF_IDX[$m]}|--epoch_batches ${b} ${pos_flag} --val_steps ${b},$((b * 2)),$((b * 3))")
      done
    done
    ;;
  *)
    echo "unknown S1_GROUP=$GROUP" >&2
    exit 2
    ;;
esac

run_one() {
  local gpu="$1"
  local spec="$2"
  local run_id steps extra result_dir
  local shards="1" freq=""
  local -a F
  IFS='|' read -ra F <<< "$spec"
  if [[ "${#F[@]}" -eq 5 ]]; then
    run_id="${F[0]}"; steps="${F[1]}"; shards="${F[2]}"; freq="${F[3]}"; extra="${F[4]}"
  else
    run_id="${F[0]}"; steps="${F[1]}"; extra="${F[2]}"
  fi
  result_dir="$OUT_DIR/${run_id}_fixed"
  if [[ -f "$result_dir/summary.json" ]]; then
    echo "[yv-$GROUP] skip complete $run_id"
    return 0
  fi
  [[ ! -e "$result_dir" ]] || {
    echo "[yv-$GROUP] refusing partial directory $result_dir" >&2
    return 2
  }
  echo "[yv-$GROUP] $run_id gpu=$gpu steps=$steps shards=$shards freq=$freq"
  if [[ -n "$freq" ]]; then
    NGLAB_FREQ_INDEX="$OUT_DIR/../${freq}.npz" \
    NGLAB_PY="$PY" NGLAB_OUT_DIR="$OUT_DIR" bash "$SCRIPT_DIR/run_v5_clean.sh" \
      "$gpu" "$run_id" "$shards" "$VAL_SHARDS" "$steps" \
      $extra
  else
    NGLAB_PY="$PY" NGLAB_OUT_DIR="$OUT_DIR" bash "$SCRIPT_DIR/run_v5_clean.sh" \
      "$gpu" "$run_id" "$shards" "$VAL_SHARDS" "$steps" \
      $extra
  fi
}

active=0
slot=0
gpu_count="${#GPUS[@]}"
pids=()
for spec in "${SPECS[@]}"; do
  while [[ "$active" -ge "$gpu_count" ]]; do
    wait "${pids[0]}" || echo "[yv-$GROUP] a run failed; continuing" >&2
    pids=("${pids[@]:1}")
    active=$((active - 1))
  done
  run_one "${GPUS[$slot]}" "$spec" &
  pids+=("$!")
  active=$((active + 1))
  slot=$(( (slot + 1) % gpu_count ))
done
while [[ "$active" -gt 0 ]]; do
  wait "${pids[0]}" || echo "[yv-$GROUP] a run failed; continuing" >&2
  pids=("${pids[@]:1}")
  active=$((active - 1))
done
echo "[yv-$GROUP] complete"
