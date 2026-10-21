#!/usr/bin/env bash
# Teacher-residual distillation launcher (pro5k6, 8x RTX PRO 5000 72GB).
#
# Four TP=2 workers, one per GPU pair.  Shards are interleaved across workers so
# every worker sees a comparable document mix.  Each worker stops once it has
# consumed TOK_BUDGET_PER_WORKER project tokens; total target = 10B tokens.
#
#   ./run_w3_distill_prod.sh start     # launch all four workers
#   ./run_w3_distill_prod.sh status    # progress snapshot
#   ./run_w3_distill_prod.sh stop      # SIGTERM (workers save their part first)
set -uo pipefail

HOST="${HOST:-pro5kvrdma6}"
NDATA="${NDATA:-/data}"
TAG="${TAG:-w3p}"
NWORK="${NWORK:-4}"
TOK_PER_WORKER="${TOK_PER_WORKER:-2.5e9}"
SHARD_START="${SHARD_START:-0}"
SHARD_END="${SHARD_END:-210}"
LAYERS="${LAYERS:-0,10,25,40}"

cmd="${1:-status}"
case "$cmd" in
start)
  for w in $(seq 0 $((NWORK - 1))); do
    g0=$((w * 2)); g1=$((w * 2 + 1))
    ssh "$HOST" "cd $NDATA && CUDA_VISIBLE_DEVICES=$g0,$g1 OE_CAP_LAYERS=$LAYERS \
      setsid nohup python3 $NDATA/distill_prod.py \
        --worker $w --num_workers $NWORK \
        --shard_start $SHARD_START --shard_end $SHARD_END \
        --token_budget $TOK_PER_WORKER --tag $TAG \
        > $NDATA/distill_w${w}.log 2>&1 < /dev/null & echo launched w$w on GPU $g0,$g1"
  done
  ;;
status)
  ssh "$HOST" "for f in $NDATA/distill_w*.log; do \
      echo \"== \$f\"; grep -E 'worker|diag' \$f | tail -4; done; \
    echo '== parts'; ls -lh $NDATA/distill_out/ 2>/dev/null | tail -8; \
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
  ;;
stop)
  ssh "$HOST" "pkill -TERM -f distill_prod.py; sleep 5; pgrep -f distill_prod.py | wc -l"
  ;;
*)
  echo "usage: $0 {start|status|stop}" >&2
  exit 2
  ;;
esac
