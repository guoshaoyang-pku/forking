#!/usr/bin/env bash
# Long-horizon context-masking runs (2026-09-22 user request, experiment-log
# §62 follow-up): extend the both-arm ctxmask triple to 4 epochs (1348 steps)
# so the epoch-1-masked trial's aftermath stays visible through epochs 2-4
# (fig7 redesign: "epoch2 的过程我想看 epoch1 被冻结住的那个 trial 的后续").
#
# Identical setting to run_ctxmask_ab.sh (same shards/seed/optimizer/tables);
# ONLY the horizon changes -- same trial, run longer:
#   *_ctxmask_ctrl_l4 : 1348 steps, intervention=none, A/B eval only
#   *_ctxmask_e1_l4   : 1348 steps, mask A from epoch-2 boundary (intervention_epoch 1)
#   *_ctxmask_e2_l4   : 1348 steps, mask A from epoch-3 boundary (intervention_epoch 2)
#
# Usage: NGLAB_OUT_DIR=<runs_scaling> bash code/cluster/run_ctxmask_ab_long.sh <gpu>
set -euo pipefail

GPU="${1:?gpu id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
export NGLAB_OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_scaling}"

BOTH=""
COMMON="--epoch_batches 337 --fixed_train_probe 4"
MASK="--intervention random_context_loss_mask --save_epoch_checkpoints --save_final_model"
CTRL="--context_eval_ab 1"
STEPS=1348

for kind in ctrl e1 e2; do
  case "$kind" in
    ctrl) EXTRA="$CTRL" ;;
    e1)   EXTRA="$MASK --intervention_epoch 1" ;;
    e2)   EXTRA="$MASK --intervention_epoch 2" ;;
  esac
  run_id="s1v5_128_marm_both_ctxmask_${kind}_l4"
  if [[ -f "$NGLAB_OUT_DIR/${run_id}_fixed/summary.json" ]]; then
    echo "[ctxmask-long] skip complete $run_id"
    continue
  fi
  echo "[ctxmask-long] $run_id gpu=$GPU steps=$STEPS"
  NGLAB_PY="$PY" bash "$SCRIPT_DIR/run_v5_clean.sh" "$GPU" "$run_id" \
    1 2,3,4,5,6,7,8,9,10,6542 "$STEPS" $COMMON $BOTH $EXTRA
done
echo "[ctxmask-long] gpu=$GPU queue done"
