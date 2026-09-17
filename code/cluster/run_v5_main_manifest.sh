#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
GROUP="${V5_GROUP:?set V5_GROUP to inj, inj_freq10, dose, dose_freq10, epoch, causal, causal_refresh, mask_high_refresh, rho, long, s1_epoch, s1_frequency, module_arms, or module_arms_epoch}"
GPUS=("$@")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "usage: V5_GROUP=<group> NGLAB_PY=python3 $0 <gpu> [gpu...]" >&2
  exit 2
fi

case "$GROUP" in
  inj)
    SPECS=(
      "nglab1x_input_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab1x_y_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position y"
      "nglab1x_v_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position v"
      "nglab1x_nogram_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--enable_bigram 0 --enable_trigram 0"
    )
    ;;
  inj_seed43)
    SPECS=(
      "nglab1x_input_v5_s43|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 43"
      "nglab1x_y_v5_s43|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 43 --injection_position y"
      "nglab1x_v_v5_s43|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 43 --injection_position v"
      "nglab1x_nogram_v5_s43|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 43 --enable_bigram 0 --enable_trigram 0"
    )
    ;;
  inj_seed44)
    SPECS=(
      "nglab1x_input_v5_s44|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 44"
      "nglab1x_y_v5_s44|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 44 --injection_position y"
      "nglab1x_v_v5_s44|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 44 --injection_position v"
      "nglab1x_nogram_v5_s44|1|2,3,4,5,6,7,8,9,10,6542|2000|--seed 44 --enable_bigram 0 --enable_trigram 0"
    )
    ;;
  inj_freq10)
    SPECS=(
      "nglab1x_input_v5_freq10_r1|1|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab1x_y_v5_freq10|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position y"
      "nglab1x_v_v5_freq10|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position v"
      "nglab1x_nogram_v5_freq10|1|2,3,4,5,6,7,8,9,10,6542|2000|--enable_bigram 0 --enable_trigram 0"
    )
    ;;
  inj_fd)
    # fast-diag rerun of the main 4 arms: bf16 diagnostic forwards + async CPU
    # aggregation. Training dynamics identical; diagnostics shift only at
    # bf16-noise level. Baseline family: *_v5_128x_freq10_fixed.
    SPECS=(
      "nglab1x_input_v5_128x_freq10_fd|1|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab1x_y_v5_128x_freq10_fd|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position y"
      "nglab1x_v_v5_128x_freq10_fd|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position v"
      "nglab1x_nogram_v5_128x_freq10_fd|1|2,3,4,5,6,7,8,9,10,6542|2000|--enable_bigram 0 --enable_trigram 0"
    )
    ;;
  dose)
    SPECS=(
      "nglab0_25x_input_v5|62|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab0_5x_input_v5|60|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab0_75x_input_v5|63|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab1_5x_input_v5|1,61|3,4,5,6,7,8,9,10,6542|2000|"
      "nglab2x_input_v5|1,2|3,4,5,6,7,8,9,10,6542|2000|"
      "nglab2_5x_input_v5|1,2,64|4,5,6,7,8,9,10,6542|2000|"
      "nglab3x_input_v5|1,2,3|4,5,6,7,8,9,10,6542|2000|"
      "nglab4x_input_v5|1,2,3,4|5,6,7,8,9,10,6542|2000|"
      "nglab5x_input_v5|1,2,3,4,5|6,7,8,9,10,6542|2000|"
      "nglab6x_input_v5|1,2,3,4,5,6|7,8,9,10,6542|2000|"
      "nglab8x_input_v5|1,2,3,4,5,6,7,8|9,10,6542|2000|"
    )
    ;;
  dose_freq10)
    SPECS=(
      "nglab0_25x_input_v5_freq10|62|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab0_5x_input_v5_freq10|60|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab0_75x_input_v5_freq10|63|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab1_5x_input_v5_freq10|1,61|3,4,5,6,7,8,9,10,6542|2000|"
      "nglab2x_input_v5_freq10|1,2|3,4,5,6,7,8,9,10,6542|2000|"
      "nglab2_5x_input_v5_freq10|1,2,64|4,5,6,7,8,9,10,6542|2000|"
      "nglab3x_input_v5_freq10|1,2,3|4,5,6,7,8,9,10,6542|2000|"
      "nglab4x_input_v5_freq10|1,2,3,4|5,6,7,8,9,10,6542|2000|"
      "nglab5x_input_v5_freq10|1,2,3,4,5|6,7,8,9,10,6542|2000|"
      "nglab6x_input_v5_freq10|1,2,3,4,5,6|7,8,9,10,6542|2000|"
      "nglab8x_input_v5_freq10|1,2,3,4,5,6,7,8|9,10,6542|2000|"
    )
    ;;
  epoch)
    SPECS=(
      "nglab0_25x_e5_v5|62|2,3,4,5,6,7,8,9,10,6542|420|"
      "nglab0_5x_e5_v5|60|2,3,4,5,6,7,8,9,10,6542|840|"
      "nglab0_75x_e5_v5|63|2,3,4,5,6,7,8,9,10,6542|1260|"
      "nglab1x_e5_v5|1|2,3,4,5,6,7,8,9,10,6542|1685|"
      "nglab1_5x_e5_v5|1,61|3,4,5,6,7,8,9,10,6542|2525|"
      "nglab2x_e5_v5|1,2|3,4,5,6,7,8,9,10,6542|3350|"
      "nglab2_5x_e5_v5|1,2,64|4,5,6,7,8,9,10,6542|4190|"
      "nglab3x_e5_v5|1,2,3|4,5,6,7,8,9,10,6542|5000|"
      "nglab4x_e5_v5|1,2,3,4|5,6,7,8,9,10,6542|6700|"
    )
    ;;
  causal)
    SPECS=(
      "nglab1x_freeze_table_e1_v5|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention freeze_table --intervention_epoch 1"
      "nglab1x_freeze_backbone_e1_v5|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention freeze_backbone --intervention_epoch 1"
    )
    ;;
  causal_refresh)
    SPECS=(
      "causalv5c_none|1|2,3,4,5,6,7,8,9,10,6542|1000|"
      "causalv5c_freeze_table_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention freeze_table --intervention_epoch 1"
      "causalv5c_freeze_backbone_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention freeze_backbone --intervention_epoch 1"
      "causalv5c_hash_reseed_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention hash_reseed --intervention_epoch 1"
      "causalv5c_hash_reseed_e1e2|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention hash_reseed --intervention_epochs 1,2"
      "causalv5c_mask_low_f200_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 200"
      "causalv5c_mask_high_f200_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 200"
      "causalv5m_mask_low_le0_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 0 --intervention_low_inclusive 1"
      "causalv5m_mask_low_le1_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 1 --intervention_low_inclusive 1"
      "causalv5m_mask_low_le2_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 2 --intervention_low_inclusive 1"
      "causalv5m_mask_low_le4_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 4 --intervention_low_inclusive 1"
      "causalv5m_mask_low_le8_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 8 --intervention_low_inclusive 1"
    )
    ;;
  mask_high_refresh)
    # f >= t inclusive semantics (current code); replaces the historical f > t scan
    SPECS=()
    for t in 1 2 5 10 25 50 100 400 800 1600 3200 6400 12800; do
      SPECS+=("causalv5m2_mask_high_t${t}_e1|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold ${t}")
    done
    ;;
  causal_dynamics)
    # A(t,passes) dynamics probe, 2026-08-30: 6-pass (2022-step) horizon.
    # Single variable per arm vs causalv5m3_none_2022; input arm, both tables,
    # 128x standard from run_v5_clean.sh defaults; extra flags win (argparse last).
    SPECS=(
      "causalv5m3_none_2022|1|2,3,4,5,6,7,8,9,10,6542|2022|"
      "causalv5m3_freeze_backbone_e1|1|2,3,4,5,6,7,8,9,10,6542|2022|--intervention freeze_backbone --intervention_epoch 1"
      "causalv5m3_freeze_backbone_e2|1|2,3,4,5,6,7,8,9,10,6542|2022|--intervention freeze_backbone --intervention_epoch 2"
      "causalv5m3_freeze_backbone_e3|1|2,3,4,5,6,7,8,9,10,6542|2022|--intervention freeze_backbone --intervention_epoch 3"
      "causalv5m3_freeze_table_e2|1|2,3,4,5,6,7,8,9,10,6542|2022|--intervention freeze_table --intervention_epoch 2"
      "causalv5m3_wd0|1|2,3,4,5,6,7,8,9,10,6542|2022|--weight_decay 0.0"
      "causalv5m3_wd03|1|2,3,4,5,6,7,8,9,10,6542|2022|--weight_decay 0.3"
    )
    ;;
  rho)
    SPECS=(
      "nglab1x_input_rho_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--fixed_train_probe 4 --probe_eval_interval 10"
      "nglab2x_input_rho_v5|1,2|3,4,5,6,7,8,9,10,6542|2000|--fixed_train_probe 4 --probe_eval_interval 10"
    )
    ;;
  long)
    SPECS=(
      "nglab1x_nogram_long_v5|1|2,3,4,5,6,7,8,9,10,6542|8000|--enable_bigram 0 --enable_trigram 0"
    )
    ;;
  s1_epoch)
    SPECS=(
      "s1v5_L1_both_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 42"
      "s1v5_L2_both_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 84"
      "s1v5_L3_both_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 168"
      "s1v5_L4_both_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337"
      "s1v5_L1_nogram_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 42 --enable_bigram 0 --enable_trigram 0"
      "s1v5_L2_nogram_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 84 --enable_bigram 0 --enable_trigram 0"
      "s1v5_L3_nogram_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 168 --enable_bigram 0 --enable_trigram 0"
      "s1v5_L4_nogram_fs|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0"
    )
    ;;
  s1_frequency)
    SPECS=(
      "s1v5_freq_bigram|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 1 --enable_trigram 0"
      "s1v5_freq_trigram|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 0 --enable_trigram 1"
      "s1v5_freq_both|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337"
      "s1v5_freq_nogram|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0"
    )
    ;;
  module_arms)
    # Four-way module ablation (native / +bigram / +trigram / both) at the
    # current 128x standard, for the logits rank-distribution figure in the
    # paper draft (2026-09-16). Same coordinates as the s1_frequency family;
    # only the module arms vary. First wave (2026-09-16) completed without
    # checkpoints; the _ckpt wave adds --save_final_model for the offline
    # logits eval (code/tools/eval_logits_rank.py).
    SPECS=(
      "s1v5_128_marm_nogram|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0"
      "s1v5_128_marm_bigram|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 1 --enable_trigram 0"
      "s1v5_128_marm_trigram|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 0 --enable_trigram 1"
      "s1v5_128_marm_both|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337"
      "s1v5_128_marm_nogram_ckpt|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0 --save_final_model"
      "s1v5_128_marm_bigram_ckpt|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 1 --enable_trigram 0 --save_final_model"
      "s1v5_128_marm_trigram_ckpt|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --enable_bigram 0 --enable_trigram 1 --save_final_model"
      "s1v5_128_marm_both_ckpt|1|2,3,4,5,6,7,8,9,10,6542|1000|--epoch_batches 337 --save_final_model"
    )
    ;;
  module_arms_epoch)
    # Epoch-boundary checkpoints for the same four module arms: stop at the
    # end of epoch 1 (337 steps) and epoch 2 (674 steps) with
    # --save_final_model, so eval_logits_rank.py can score the rank
    # distribution at each epoch boundary. The existing _ckpt wave
    # (step 1000) is the epoch-3 boundary. Shard-1 fixed replay + seed 42
    # make the first 337/674 steps bit-identical to the 1000-step runs, so
    # the three boundaries are the same trajectories sampled at e1/e2/e3.
    SPECS=(
      "s1v5_128_marm_nogram_e1ckpt|1|2,3,4,5,6,7,8,9,10,6542|337|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0 --save_final_model"
      "s1v5_128_marm_bigram_e1ckpt|1|2,3,4,5,6,7,8,9,10,6542|337|--epoch_batches 337 --enable_bigram 1 --enable_trigram 0 --save_final_model"
      "s1v5_128_marm_trigram_e1ckpt|1|2,3,4,5,6,7,8,9,10,6542|337|--epoch_batches 337 --enable_bigram 0 --enable_trigram 1 --save_final_model"
      "s1v5_128_marm_both_e1ckpt|1|2,3,4,5,6,7,8,9,10,6542|337|--epoch_batches 337 --save_final_model"
      "s1v5_128_marm_nogram_e2ckpt|1|2,3,4,5,6,7,8,9,10,6542|674|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0 --save_final_model"
      "s1v5_128_marm_bigram_e2ckpt|1|2,3,4,5,6,7,8,9,10,6542|674|--epoch_batches 337 --enable_bigram 1 --enable_trigram 0 --save_final_model"
      "s1v5_128_marm_trigram_e2ckpt|1|2,3,4,5,6,7,8,9,10,6542|674|--epoch_batches 337 --enable_bigram 0 --enable_trigram 1 --save_final_model"
      "s1v5_128_marm_both_e2ckpt|1|2,3,4,5,6,7,8,9,10,6542|674|--epoch_batches 337 --save_final_model"
    )
    ;;
  *)
    echo "unknown V5_GROUP=$GROUP" >&2
    exit 2
    ;;
esac

run_one() {
  local gpu="$1"
  local spec="$2"
  local run_id train_shards val_shards steps extra result_dir
  IFS='|' read -r run_id train_shards val_shards steps extra <<< "$spec"
  result_dir="$OUT_DIR/${run_id}_fixed"
  if [[ -f "$result_dir/summary.json" ]]; then
    echo "[v5-$GROUP] skip complete $run_id"
    return 0
  fi
  [[ ! -e "$result_dir" ]] || {
    echo "[v5-$GROUP] refusing partial directory $result_dir" >&2
    return 2
  }
  echo "[v5-$GROUP] $run_id gpu=$gpu steps=$steps"
  NGLAB_PY="$PY" NGLAB_OUT_DIR="$OUT_DIR" bash "$SCRIPT_DIR/run_v5_clean.sh" "$gpu" "$run_id" \
    "$train_shards" "$val_shards" "$steps" $extra
}

active=0
slot=0
gpu_count="${#GPUS[@]}"
pids=()
for spec in "${SPECS[@]}"; do
  while [[ "$active" -ge "$gpu_count" ]]; do
    if ! wait "${pids[0]}"; then
      echo "[v5-$GROUP] a run failed; continuing remaining queue" >&2
    fi
    pids=("${pids[@]:1}")
    active=$((active - 1))
  done
  run_one "${GPUS[$slot]}" "$spec" &
  pids+=("$!")
  active=$((active + 1))
  slot=$(( (slot + 1) % gpu_count ))
done
while [[ "$active" -gt 0 ]]; do
  if ! wait "${pids[0]}"; then
    echo "[v5-$GROUP] a run failed; continuing remaining queue" >&2
  fi
  pids=("${pids[@]:1}")
  active=$((active - 1))
done
echo "[v5-$GROUP] complete"