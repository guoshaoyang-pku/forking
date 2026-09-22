#!/usr/bin/env bash
# Context-masking (A/B) experiment launcher, v5 clean-table marm baseline.
# Reproduces the collaborator's random-context loss-masking runs and adds the
# A/B context-gap eval (context_gap.jsonl) used for paper fig7/fig7*.
#
# Per arm (nogram/bigram/trigram/both), three runs from scratch
# (shard-1 fixed replay + seed 42 => identical trajectory to the marm runs
# up to the intervention boundary):
#   *_ctxmask_ctrl : 1011 steps, intervention=none, A/B eval only
#   *_ctxmask_e1   : 674 steps, loss-mask set A from epoch 2 (intervention_epoch 1)
#   *_ctxmask_e2   : 1011 steps, loss-mask set A from epoch 3 (intervention_epoch 2)
# Mask runs save epoch boundary checkpoints (--save_epoch_checkpoints) and
# final models for the offline KL eval (code/tools/eval_context_kl.py).
#
# Usage: NGLAB_OUT_DIR=<runs_scaling> bash code/cluster/run_ctxmask_ab.sh <gpu>
set -euo pipefail

GPU="${1:?gpu id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
export NGLAB_OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_scaling}"

NOGRAM="--enable_bigram 0 --enable_trigram 0"
BIGRAM="--enable_bigram 1 --enable_trigram 0"
TRIGRAM="--enable_bigram 0 --enable_trigram 1"
BOTH=""
COMMON="--epoch_batches 337 --fixed_train_probe 4"
MASK="--intervention random_context_loss_mask --save_epoch_checkpoints --save_final_model"
CTRL="--context_eval_ab 1"

case "$GPU" in
  3)
    SPEC_IDS=(both_ctrl both_e1 both_e2 nogram_ctrl) ;;
  5)
    SPEC_IDS=(nogram_e1 nogram_e2 bigram_ctrl bigram_e1) ;;
  6)
    SPEC_IDS=(bigram_e2 trigram_ctrl trigram_e1 trigram_e2) ;;
  *)
    echo "unknown GPU assignment: $GPU (edit run_ctxmask_ab.sh)" >&2
    exit 2 ;;
esac

for spec in "${SPEC_IDS[@]}"; do
  arm="${spec%_*}"; kind="${spec##*_}"
  case "$arm" in
    nogram) ARMFLAGS="$NOGRAM" ;;
    bigram) ARMFLAGS="$BIGRAM" ;;
    trigram) ARMFLAGS="$TRIGRAM" ;;
    both) ARMFLAGS="$BOTH" ;;
  esac
  case "$kind" in
    ctrl) STEPS=1011; EXTRA="$CTRL" ;;
    e1)   STEPS=674;  EXTRA="$MASK --intervention_epoch 1" ;;
    e2)   STEPS=1011; EXTRA="$MASK --intervention_epoch 2" ;;
  esac
  run_id="s1v5_128_marm_${arm}_ctxmask_${kind}"
  if [[ -f "$NGLAB_OUT_DIR/${run_id}_fixed/summary.json" ]]; then
    echo "[ctxmask] skip complete $run_id"
    continue
  fi
  echo "[ctxmask] $run_id gpu=$GPU steps=$STEPS"
  NGLAB_PY="$PY" bash "$SCRIPT_DIR/run_v5_clean.sh" "$GPU" "$run_id" \
    1 2,3,4,5,6,7,8,9,10,6542 "$STEPS" $COMMON $ARMFLAGS $EXTRA
done
echo "[ctxmask] gpu=$GPU queue done"
