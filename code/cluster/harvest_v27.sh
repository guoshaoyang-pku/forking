#!/bin/bash
# Harvest v27 train_log.jsonl + summary.json from all hosts to Mac.
# usage: harvest_v27.sh [host ...]   (default: all)
set -u
ROOT=/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/data/harvest/v27_a100
mkdir -p "$ROOT"
HOSTS=${@:-"a100_perm a100_t1 a100_t1_2 a100_t1_3 a100_t1_4 360-2"}
for h in $HOSTS; do
  case $h in
    a100_perm) RDIR=/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed ;;
    360-2)     RDIR=/data/home/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed ;;
    *)         RDIR=/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed ;;
  esac
  echo "== $h =="
  rsync -az --timeout=60 \
    --include='v27_*/' --include='v27_*/train_log.jsonl' --include='v27_*/summary.json' \
    --exclude='*' \
    "$h:$RDIR/" "$ROOT/$h/" 2>&1 | tail -1
done
echo HARVEST_DONE
