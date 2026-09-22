#!/usr/bin/env bash
# Row-mode context-masking (A/B) launcher — collision-private subspaces.
#
# Type-mode ctxmask masks 2% of distinct trigram TYPES, but the clean table
# hashes 19.0M types onto 1M rows (~19 types/row), so a "masked" row keeps
# being trained by its ~18 unmasked colliding types (type-mode A flattened
# only ~25%, see experiment-log §62). Row mode masks 2% of table ROWS via the
# exact clean-table hash, so every context landing on a masked row is excluded
# from the loss: the masked subspace is private by construction. This tests
# whether the collaborator's flat masked-A curve corresponds to row masking.
#
# Runs (from scratch, shard-1 fixed replay, seed 42, identical trajectory to
# the marm/ctxmask runs up to the intervention boundary):
#   *_rowmask_ctrl : 1011 steps, intervention=none, row-mode A/B eval only
#   *_rowmask_e1   : 674 steps, loss-mask set A from epoch 2
#   *_rowmask_e2   : 1011 steps, loss-mask set A from epoch 3
#
# Minimal discriminating set for fig7/fig7*: both triad (fig7) + nogram mask
# runs (fig7* masked-epoch KL). nogram_ctrl is skipped: with no table, row
# selection is just another random 2% of contexts, and the type-mode nogram
# curves/KL already provide the reference.
#
# Usage: NGLAB_OUT_DIR=<runs_scaling> bash code/cluster/run_rowmask_ab.sh <gpu>
set -euo pipefail

GPU="${1:?gpu id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
export NGLAB_OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_scaling}"

NOGRAM="--enable_bigram 0 --enable_trigram 0"
BOTH=""
COMMON="--epoch_batches 337 --fixed_train_probe 4 --context_set_mode row"
MASK="--intervention random_context_loss_mask --save_epoch_checkpoints --save_final_model"
CTRL="--context_eval_ab 1"

case "$GPU" in
  1)
    SPEC_IDS=(both_ctrl both_e1 both_e2) ;;
  3)
    SPEC_IDS=(nogram_e1 nogram_e2) ;;
  *)
    echo "unknown GPU assignment: $GPU (edit run_rowmask_ab.sh)" >&2
    exit 2 ;;
esac

for spec in "${SPEC_IDS[@]}"; do
  arm="${spec%_*}"; kind="${spec##*_}"
  case "$arm" in
    nogram) ARMFLAGS="$NOGRAM" ;;
    both) ARMFLAGS="$BOTH" ;;
  esac
  case "$kind" in
    ctrl) STEPS=1011; EXTRA="$CTRL" ;;
    e1)   STEPS=674;  EXTRA="$MASK --intervention_epoch 1" ;;
    e2)   STEPS=1011; EXTRA="$MASK --intervention_epoch 2" ;;
  esac
  run_id="s1v5_128_marm_${arm}_rowmask_${kind}"
  if [[ -f "$NGLAB_OUT_DIR/${run_id}_fixed/summary.json" ]]; then
    echo "[rowmask] skip complete $run_id"
    continue
  fi
  echo "[rowmask] $run_id gpu=$GPU steps=$STEPS"
  NGLAB_PY="$PY" bash "$SCRIPT_DIR/run_v5_clean.sh" "$GPU" "$run_id" \
    1 2,3,4,5,6,7,8,9,10,6542 "$STEPS" $COMMON $ARMFLAGS $EXTRA
done
echo "[rowmask] gpu=$GPU queue done"
