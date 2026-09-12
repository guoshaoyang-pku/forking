# Remote preflight - 2026-09-13

Read-only checks for the overnight queue. No GPU run was started.

| host | project path | GPU state at check | code/train.py MD5 | code/ngram_freq.py MD5 | run_baseline.sh MD5 |
|---|---|---|---|---|---|
| local | /Users/guoshaoyang/Desktop/workdir/ngram-gap-lab | n/a | 29965c1157908df0d5473e3416e09b0d | 3baba85c9b18c433a564c3654f63f2b3 | c004acc11b73a5f1aa0a043bd6926f58 |
| ophis-gpu | /data4/guoshaoyang/ngram-gap-lab | occupied on several cards | 29965c1157908df0d5473e3416e09b0d | 3baba85c9b18c433a564c3654f63f2b3 | c004acc11b73a5f1aa0a043bd6926f58 |
| 360-1 | /data/home/guoshaoyang/ngram-gap-lab | occupied on several cards | 1ca8ebb86a493517e688e3570c605be2 | dd2e92f8416cba96b4fcb678941cd90e | c004acc11b73a5f1aa0a043bd6926f58 |
| 360-2 | /data/home/guoshaoyang/ngram-gap-lab | all 8 cards reported 0 MiB used | 1ca8ebb86a493517e688e3570c605be2 | dd2e92f8416cba96b4fcb678941cd90e | c004acc11b73a5f1aa0a043bd6926f58 |

## Decision

T8/T9 remain queued, despite 360-2 being idle. The code mismatch violates the cross-machine md5 requirement in agents.md P4. A future run requires authorized code sync, explicit commit/version choice, a second md5 check, train/val overlap check, run registration, and GPU ownership.

No files were copied, no large artifacts were moved, and no process was stopped.
