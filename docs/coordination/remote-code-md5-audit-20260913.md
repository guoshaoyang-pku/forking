# Remote code MD5 audit · 2026-09-13

Read-only SSH comparison for the T8/T9 candidate hosts. No code was copied or changed.

| host | train.py | ngram_freq.py | run_baseline.sh | run_v5_clean.sh | result |
|---|---|---|---|---|---|
| local authority | 29965c1157908df0d5473e3416e09b0d | 3baba85c9b18c433a564c3654f63f2b3 | 0cfc37a520f789e5b47b7998a624e0fd | 1c1ad9c2043b75a1e962ecd0d5f5628c | reference |
| ophis-gpu | 29965c1157908df0d5473e3416e09b0d | 3baba85c9b18c433a564c3654f63f2b3 | c004acc11b73a5f1aa0a043bd6926f58 | 1c1ad9c2043b75a1e962ecd0d5f5628c | train/frequency match; baseline launcher differs |
| 360-1 | 1ca8ebb86a493517e688e3570c605be2 | dd2e92f8416cba96b4fcb678941cd90e | c004acc11b73a5f1aa0a043bd6926f58 | 1c1ad9c2043b75a1e962ecd0d5f5628c | all core files differ from local |
| 360-2 | 1ca8ebb86a493517e688e3570c605be2 | dd2e92f8416cba96b4fcb678941cd90e | c004acc11b73a5f1aa0a043bd6926f58 | 1c1ad9c2043b75a1e962ecd0d5f5628c | all core files differ from local |

## Decision

The remote candidate runs cannot be promoted to current local SSOT evidence from endpoint summaries alone. 360-1 and 360-2 share one code revision with each other, but it differs from the local authority in train.py and ngram_freq.py; ophis matches the local core Python files but uses a different baseline launcher. This is a provenance gate, not proof that any run is invalid. Before a new current run, sync an approved commit and record post-sync md5; before historical backfill, retain host-scoped labels and compare the exact launcher/config semantics.

The audit does not authorize code synchronization, branch changes, GPU execution, or push.
