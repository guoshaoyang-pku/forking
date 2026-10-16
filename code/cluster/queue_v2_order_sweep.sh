#!/usr/bin/env bash
# Per-host GPU queue for the v2 order sweep.  One loop per listed GPU pops the
# next "<orders> <seed> [steps]" line from the job file under flock and runs
# run_v2_order_sweep.sh; a GPU is only used when it holds < 2 GiB.
#
#   queue_v2_order_sweep.sh <job_file> <gpu,gpu,...>
set -uo pipefail

JOBS="${1:?job file}"
GPUS="${2:?comma gpu list}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
export TMPDIR="${TMPDIR:-$ROOT/tmp}"

pop_job() {
  flock "$JOBS.lock" bash -c '
    line=$(grep -m1 -v "^\s*\(#\|$\)" "$0" || true)
    [[ -n "$line" ]] && { grep -vxF -- "$line" "$0" > "$0.tmp" || true; mv "$0.tmp" "$0"; }
    printf "%s" "$line"' "$JOBS"
}

gpu_loop() {
  local gpu="$1" line
  while true; do
    until (( $(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) < 2048 )); do
      sleep 60
    done
    line="$(pop_job)"
    [[ -z "$line" ]] && { echo "[queue] gpu $gpu: no jobs left"; return; }
    echo "[queue] $(date '+%F %T') gpu $gpu start: $line"
    # shellcheck disable=SC2086
    bash "$SCRIPT_DIR/run_v2_order_sweep.sh" "$gpu" $line \
      && echo "[queue] $(date '+%F %T') gpu $gpu done: $line" \
      || echo "[queue] $(date '+%F %T') gpu $gpu FAILED: $line"
  done
}

for gpu in ${GPUS//,/ }; do
  gpu_loop "$gpu" >> "$LOG_DIR/queue_gpu${gpu}.log" 2>&1 &
  sleep 5
done
wait
