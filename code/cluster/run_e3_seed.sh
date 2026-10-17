#!/usr/bin/env bash
# Seed-carrying job wrapper for queue_sweeps.sh: the queue only passes
# positionals, so per-line seeds go through this shim which sets NGLAB_SEED
# and forwards to the real launcher.
#
#   queue line:  "run_e3_seed.sh <seed> <launcher> <launcher args...>"
#   queue invokes: bash run_e3_seed.sh <gpu> <seed> <launcher> <args...>
set -euo pipefail
GPU="${1:?gpu id}"
SEED="${2:?seed}"
RUNNER="${3:?launcher name, e.g. run_v23_dataenv.sh}"
shift 3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGLAB_SEED="$SEED" bash "$SCRIPT_DIR/$RUNNER" "$GPU" "$@"
