#!/usr/bin/env bash
# Per-host GPU queue for the v2.2 R-scan and v2.3 data-envelope H200 sweeps.
# One loop per listed GPU pops the next line from the job file under flock and
# runs it; a GPU is only used when it holds < 2 GiB (yields to other tenants).
#
#   queue_sweeps.sh <job_file> <gpu,gpu,...>
#   job line:  "<runner> <orders> <R|pool> [steps] [device_batch]"
#              runner: run_v22_rscan.sh | run_v23_dataenv.sh
#              e.g.  "run_v22_rscan.sh 2,3,4,5 1000000"
#                    "run_v23_dataenv.sh 2,3 10000000"
#
# Failure policy: a failed line is requeued up to MAX_FAILS times total
# (config-caused OOMs must not loop forever, and must not retire a healthy
# GPU).  Failed lines are counted in <job_file>.fails.log.
set -uo pipefail

JOBS="${1:?job file}"
GPUS="${2:?comma gpu list}"
MAX_FAILS="${NGLAB_MAX_FAILS:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOG_DIR="$ROOT/logs"
FAILS_LOG="${JOBS}.fails.log"
mkdir -p "$LOG_DIR"
touch "$FAILS_LOG"
export TMPDIR="${TMPDIR:-$ROOT/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

pop_job() {
  flock "$JOBS.lock" bash -c '
    line=$(grep -m1 -v "^\s*\(#\|$\)" "$0" || true)
    [[ -n "$line" ]] && { grep -vxF -- "$line" "$0" > "$0.tmp" || true; mv "$0.tmp" "$0"; }
    printf "%s" "$line"' "$JOBS"
}

requeue() {
  flock "$JOBS.lock" bash -c 'printf "%s\n" "$1" >> "$0"' "$JOBS" "$1"
}

fail_count() {
  grep -cxF -- "$1" "$FAILS_LOG" 2>/dev/null || true
}

gpu_loop() {
  local gpu="$1" line runner fails
  while true; do
    until (( $(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) < 2048 )); do
      sleep 60
    done
    line="$(pop_job)"
    [[ -z "$line" ]] && { echo "[queue] gpu $gpu: no jobs left"; return; }
    echo "[queue] $(date '+%F %T') gpu $gpu start: $line"
    read -r runner r_args <<< "$line"
    # shellcheck disable=SC2086
    if bash "$SCRIPT_DIR/$runner" "$gpu" $r_args; then
      echo "[queue] $(date '+%F %T') gpu $gpu done: $line"
    else
      fails=$(fail_count "$line")
      if (( fails + 1 >= MAX_FAILS )); then
        echo "[queue] $(date '+%F %T') gpu $gpu FAILED (attempt $((fails + 1))/$MAX_FAILS), giving up: $line"
        flock "$FAILS_LOG.lock" bash -c 'printf "%s\n" "$1" >> "$0"' "$FAILS_LOG" "$line"
      else
        flock "$FAILS_LOG.lock" bash -c 'printf "%s\n" "$1" >> "$0"' "$FAILS_LOG" "$line"
        requeue "$line"
        echo "[queue] $(date '+%F %T') gpu $gpu failed (attempt $((fails + 1))/$MAX_FAILS), requeued: $line"
      fi
    fi
  done
}

for gpu in ${GPUS//,/ }; do
  gpu_loop "$gpu" >> "$LOG_DIR/queue_sweeps_gpu${gpu}.log" 2>&1 &
  sleep 2
done
wait
