#!/usr/bin/env bash
# Per-host GPU queue for the v2.1 no-collision dictionary line.
# One loop per listed GPU pops the next "<mode> <orders|nogram> <seed> [steps]"
# line from the job file under flock and runs run_v21_dict_lines.sh; a GPU is
# only used when it holds < 2 GiB (yields to other tenants, e.g. the nightly
# RL jobs on the A100 fleet).
#
#   queue_v21_dict_lines.sh <job_file> <gpu,gpu,...>
#   job line:  "dict 2,3,4,5 42"  |  "hash 5 42"  |  "keephash 2,3,4,5 42"  |  "nogram nogram 42"
#              optional 4th field = device batch (A100 arms with >=2 tables
#              need 12; unset = NGLAB_DEVICE_BATCH or 72)
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

requeue() {
  flock "$JOBS.lock" bash -c 'printf "%s\n" "$1" >> "$0"' "$JOBS" "$1"
}

gpu_loop() {
  local gpu="$1" line t0 run_dir
  while true; do
    until (( $(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits) < 2048 )); do
      sleep 60
    done
    line="$(pop_job)"
    [[ -z "$line" ]] && { echo "[queue] gpu $gpu: no jobs left"; return; }
    echo "[queue] $(date '+%F %T') gpu $gpu start: $line"
    t0=$SECONDS
    # shellcheck disable=SC2086
    if bash "$SCRIPT_DIR/run_v21_dict_lines.sh" "$gpu" $line; then
      echo "[queue] $(date '+%F %T') gpu $gpu done: $line"
    elif (( SECONDS - t0 < 180 )); then
      # launch-time failure = unhealthy GPU: hand the job back, retire the GPU
      read -r j_mode j_orders j_seed j_steps j_mb <<< "$line"
      if [[ "$j_mode" == "nogram" || "$j_orders" == "nogram" ]]; then
        j_tag="nogram"
      else
        j_tag="${j_mode}_n${j_orders//,/-}"
      fi
      run_dir="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}/v21_${j_tag}_s${j_seed}_${j_steps:-2000}_fd_fixed"
      mkdir -p "$LOG_DIR/failed_launch"
      [[ -d "$run_dir" && ! -s "$run_dir/summary.json" ]] \
        && mv "$run_dir" "$LOG_DIR/failed_launch/$(basename "$run_dir").gpu$gpu.$(date +%s)"
      requeue "$line"
      echo "[queue] $(date '+%F %T') gpu $gpu launch failure, requeued '$line', retiring gpu"
      return
    else
      echo "[queue] $(date '+%F %T') gpu $gpu FAILED after $((SECONDS - t0))s: $line"
    fi
  done
}

for gpu in ${GPUS//,/ }; do
  gpu_loop "$gpu" >> "$LOG_DIR/queue_v21_gpu${gpu}.log" 2>&1 &
  sleep 5
done
wait
