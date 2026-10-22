#!/bin/bash
# Harvest v27 train_log.jsonl + summary.json (+ subset probe + table norms) from all hosts to Mac.
# usage: harvest_v27.sh [host ...]   (default: all)
# tione hosts have no rsync -> tar-over-ssh.
set -u
ROOT=/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/.tmp/harvest/v27_a100
mkdir -p "$ROOT"
HOSTS=${@:-"a100_perm a100_t1 a100_t1_2 a100_t1_3 a100_t1_4 360-2 ophis-gpu"}
FILES="train_log.jsonl summary.json fixed_train_loss.jsonl table_norm.jsonl"
for h in $HOSTS; do
  case $h in
    a100_perm) RDIR=/data2/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed ;;
    360-2)     RDIR=/data/home/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed ;;
    ophis-gpu) RDIR=/data1/guoshaoyang/ngram-gap-lab-v27/data/runs_fixed ;;
    *)         RDIR=/home/tione/guoshaoyang/ngram-gap-lab/data/runs_fixed ;;
  esac
  mkdir -p "$ROOT/$h"
  ssh -o ConnectTimeout=20 "$h" "cd $RDIR && for d in v27_*/; do for f in $FILES; do [ -f \"\$d\$f\" ] && echo \"\$d\$f\"; done; done | tar czf - -T -" \
    2>/dev/null | tar xzf - -C "$ROOT/$h/" 2>/dev/null
  echo "== $h: $(ls -d "$ROOT/$h"/v27_* 2>/dev/null | wc -l) runs"
done
echo HARVEST_DONE
